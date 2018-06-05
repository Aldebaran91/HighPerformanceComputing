
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

std::vector<int> GenerateRandomInput(int size)
{
	srand(time(NULL));

	std::vector<int> result;
	for (int i = 0; i < size; ++i)
	{
		//result.push_back(rand() % 10);
		result.push_back(i);
	}

	return result;
}

int main(int argc, const char** argv)
{
	cl_int err = CL_SUCCESS;
	const long INSIZE = 1024 * 1024;
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
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * INSIZE);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * INSIZE);
		cl::Buffer buffer_T(context, CL_MEM_READ_WRITE, sizeof(int) * INSIZE);

		long* input = (long*)calloc(INSIZE, sizeof(long));
		long* output = (long*)calloc(INSIZE, sizeof(long));
		long* temp = (long*)calloc(INSIZE, sizeof(long));
		for (unsigned long i = 0; i < INSIZE; ++i)
		{
			input[i] = i;
			output[i] = 0;
			temp[i] = 0;
		}

		// create a queue (a queue of commands that the GPU will execute)
		cl::CommandQueue queue(context, default_device);

		// push write commands to queue
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*INSIZE, input);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*INSIZE, output);
		queue.enqueueWriteBuffer(buffer_T, CL_TRUE, 0, sizeof(int)*INSIZE, temp);

		// RUN ZE KERNEL
		cl::Kernel kernel(program, "blelloch", &err);
		kernel.setArg(0, buffer_A);
		kernel.setArg(1, buffer_B);
		kernel.setArg(2, maxWorkGroupSize);
		kernel.setArg(3, maxWorkGroupSize * sizeof(int), NULL);

		cl::NDRange global(maxWorkGroupSize);
		cl::NDRange local(16);
			
		for (int i = 0; i < 1024; i++) {
			queue.enqueueNDRangeKernel(kernel, i * 1024, global, local);
		}
		
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*INSIZE, output);

		auto TIME_END = std::chrono::high_resolution_clock::now();

		std::cout << "Total time: " <<
			std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count() <<
			"ms" <<
			std::endl;

		std::cout << "INPUT\n" << std::endl;

		/*for (int i = 0; i < INSIZE; ++i)
		{
		std::cout << input[i] << " | ";
		}*/

		std::cout << "\n\nOUTPUT\n";

		long b = 0;
		for (int i = 0; i < INSIZE; ++i)
		{
			if (b != output[i]) {
				std::cout << b << std::endl;
				std::cout << i << " " << output[i] << " \n ";
			}
			b += i;			
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

	return err;
}