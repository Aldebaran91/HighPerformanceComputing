
// NVidia only supports OpenCL 1.2
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <chrono>

int main(int argc, const char** argv)
{
	cl_int err = CL_SUCCESS;

	cl_uint benchmarkSize = 1024*64;
	cl_uint INSIZE = 1024 * 1024;
	cl_uint WORKGROUP_COUNT = 0;
	
	size_t maxWorkGroupSize;
	const std::string KERNEL_FILE = "kernel.cl";
	std::vector<cl::Platform> all_platforms;
	std::vector<cl::Device> devices;
	cl::Platform platform;
	cl::Context context;
	cl::Program program;

	// OPENCL INIT
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0)
	{
		std::cout << " No platforms found. Check OpenCL installation!\n";
		return err;
	}
	else if (all_platforms.size() == 1)
	{
		platform = all_platforms[0];
	}
	else
	{
		platform = all_platforms[1];
	}

	cl_context_properties properties[] =
	{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
	context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

	devices = context.getInfo<CL_CONTEXT_DEVICES>();

	// use device[1] because that's a GPU; device[0] is the CPU
	cl::Device default_device = devices[0];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	maxWorkGroupSize = default_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	// Bit to Byte /1024 and uint 8bits
	long max = default_device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / 1024 / 8;
	
	std::cout << "Enter size of array (MAX " << max << ", MIN 16): ";
	std::cin >> benchmarkSize;

	if (benchmarkSize < 16 || benchmarkSize > 1048576)
	{
		std::cout << "Invalid size" << std::endl;
		return -1;
	}
	INSIZE = benchmarkSize + benchmarkSize % maxWorkGroupSize;
	WORKGROUP_COUNT = INSIZE / maxWorkGroupSize;
	if (WORKGROUP_COUNT == 0)
		WORKGROUP_COUNT = 1;
	
	try
	{
		// load and build the kernel
		std::ifstream sourceFile(KERNEL_FILE);
		if (!sourceFile)
		{
			std::cout << "kernel source file " << KERNEL_FILE << " not found!" << std::endl;
			return 1;
		}
		std::string sourceCode(
			std::istreambuf_iterator<char>(sourceFile),
			(std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
		program = cl::Program(context, source);
		program.build({ default_device });

		auto TIME_START = std::chrono::high_resolution_clock::now();

		// create buffers on device (allocate space on GPU)
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * INSIZE);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * INSIZE);
		cl::Buffer buffer_T(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * INSIZE);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * maxWorkGroupSize);
		cl::Buffer buffer_CC(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * maxWorkGroupSize);

		cl_uint* input = (cl_uint*)calloc(INSIZE, sizeof(cl_uint));
		cl_uint* output = (cl_uint*)calloc(INSIZE, sizeof(cl_uint));
		// More allocated than needed. Software developer was to lazy.
		cl_uint* cBlockSum1 = (cl_uint*)calloc(INSIZE, sizeof(cl_uint));
		cl_uint* cBlockSum2 = (cl_uint*)calloc(INSIZE, sizeof(cl_uint));

		for (cl_uint i = 0; i < INSIZE; ++i)
		{
			input[i] = i;
			output[i] = 0;
			cBlockSum1[i] = 0;
			cBlockSum2[i] = 0;
		}

		// create a queue (a queue of commands that the GPU will execute)
		cl::CommandQueue queue(context, default_device);

		// push write commands to queue
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(cl_uint) * INSIZE, input);

		// RUN ZE KERNEL
		cl::Kernel kernel(program, "blelloch", &err);
		// Input
		kernel.setArg(0, buffer_A);
		// Output
		kernel.setArg(1, buffer_B);
		// Local temp
		kernel.setArg(2, maxWorkGroupSize * sizeof(cl_uint), NULL);
		// Block sum
		kernel.setArg(3, buffer_C);

		// Set range for local and global
		cl::NDRange global(maxWorkGroupSize);
		cl::NDRange local(16);
		cl::NDRange global_work_size(INSIZE);
		
		// Execute kernel for each work group
		for (int i = 0; i < WORKGROUP_COUNT; i++) {
			queue.enqueueNDRangeKernel(kernel, i * maxWorkGroupSize, global, local);
		}
		
		// Read result
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, sizeof(cl_uint) * INSIZE, output);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(cl_uint) * maxWorkGroupSize, cBlockSum1);

		if (INSIZE >= 1024) {
			// Create new kernel for block-scan-sum
			cl::Kernel kernel1(program, "blelloch", &err);
			// Input, contains last item of each work group
			kernel1.setArg(0, buffer_CC);
			// Output
			kernel1.setArg(1, buffer_C);
			// Local temp
			kernel1.setArg(2, maxWorkGroupSize * sizeof(cl_uint), NULL);
			// Recycling of buffer_B, results will be discarded
			kernel1.setArg(3, buffer_B);

			// Write block-scan-sum result into buffer
			queue.enqueueWriteBuffer(buffer_CC, CL_TRUE, 0, sizeof(cl_uint) * maxWorkGroupSize, cBlockSum1);
			// Execute kernel
			queue.enqueueNDRangeKernel(kernel1, 0, maxWorkGroupSize, local);
			// Read result
			queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(cl_uint) * maxWorkGroupSize, cBlockSum2);


			// Create kernel for adding values from block-scan-sum to output
			cl::Kernel kernel2(program, "add", &err);
			// Input of whole array
			kernel2.setArg(0, buffer_A);
			// Block-scan-sum values
			kernel2.setArg(1, buffer_C);
			// Final result
			kernel2.setArg(2, buffer_B);

			// Write output and block-scan-sum to buffer
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(cl_uint) * INSIZE, output);
			queue.enqueueWriteBuffer(buffer_C, CL_TRUE, 0, sizeof(cl_uint) * maxWorkGroupSize, cBlockSum2);

			cl::NDRange local1(1);
			// Execute kernel for each work group
			for (int i = 0; i < WORKGROUP_COUNT; i++) {
				queue.enqueueNDRangeKernel(kernel2, i * maxWorkGroupSize, global, local1);
			}

			// Read endresult
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, sizeof(cl_uint) * INSIZE, output);
		}

		auto TIME_END = std::chrono::high_resolution_clock::now();
		std::cout << "Total time: " <<
			std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count() <<
			"ms" <<
			std::endl;
			
		bool wrongValue = false;
		cl_uint b = 0;
		for (int i = 0; i < benchmarkSize; ++i)
		{
			//std::cout << cBlockSum2[i] << std::endl;
			if (b != output[i]) {
				wrongValue = true;
				std::cout << "ERROR - EXPECTED " << b << " but got " << output[i] << " at position " << i << std::endl;
			}
			b += i;
		}

		if (!wrongValue) {
			std::cout << "All calculated values are correct!" << std::endl;
		}

		std::cin.get();
	}
	catch (cl::Error err)
	{
		std::string s;
		program.getBuildInfo(default_device, CL_PROGRAM_BUILD_LOG, &s);
		std::cout << s << std::endl;
		program.getBuildInfo(default_device, CL_PROGRAM_BUILD_OPTIONS, &s);
		std::cout << s << std::endl;

		std::cerr
			<< "ERROR: "
			<< err.what()
			<< "("
			<< err.err()
			<< ")"
			<< std::endl;
	}
	std::string str;
	std::getline(std::cin, str);

	return err;
}