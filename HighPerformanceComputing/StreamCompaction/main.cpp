
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

// CONST
const std::string KERNEL_FILE = "kernel.cl";
const int SIZE_BLOCK = 16;
const int SIZE_WG = 1024;

// GLOBAL VARS
cl_int err = CL_SUCCESS;

std::vector<cl::Platform> all_platforms;
std::vector<cl::Device> devices;
cl::Device default_device;
cl::Platform platform;
cl::Context context;
cl::Program program;

// FUNCTION HEADER
std::vector<int> stream_compaction_GPU(std::vector<int> input);
std::vector<int> strComGPU_Step1_Filter(std::vector<int> input, const int threshold, const std::string predicateKernel);
std::vector<int> strComGPU_Step2_PrefixSum(std::vector<int> input);
std::vector<int> strComGPU_Step3_Scatter(std::vector<int> input, std::vector<int> addr, std::vector<int> mask);



void Errorhandling(cl::Error err)
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

std::vector<int> generateRandomInput(int size)
{
	srand(time(NULL));
	std::vector<int> result;
	for (int i = 0; i < size; ++i)
	{
		result.push_back(rand() % 10);
	}
	return result;
}

int main(int argc, const char** argv)
{

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
	context = cl::Context(CL_DEVICE_TYPE_ALL, properties);

	devices = context.getInfo<CL_CONTEXT_DEVICES>();

	default_device = devices[0];

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
		program.build(devices);

		// SEQUENTIAL

		// GPU
		std::vector<int> input = generateRandomInput(32);
		auto output = stream_compaction_GPU(input);
	}
	catch (cl::Error err)
	{
		Errorhandling(err);
	}
	




	//cl_int err = CL_SUCCESS;
	////funkt atm nur mit bis zu 1024 elemente... ka warum
	////const int INSIZE = 1024;
	//const int INSIZE = 32;
	//const int THRESHOLD = 5;
	//const std::string KERNEL_FILE = "kernel.cl";
	//std::vector<cl::Platform> all_platforms;
	//cl::Platform::get(&all_platforms);
	//cl::Program program;




	//cl::Platform default_platform = all_platforms[0];
	//std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	//// get default device (CPUs, GPUs) of the default platform
	//std::vector<cl::Device> all_devices;
	//default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	//if (all_devices.size() == 0) {
	//	std::cout << " No devices found. Check OpenCL installation!\n";
	//	exit(1);
	//}

	//// use device[1] because that's a GPU; device[0] is the CPU
	//cl::Device default_device = all_devices[0];
	//std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	//// a context is like a "runtime link" to the device and platform;
	//// i.e. communication is possible
	//cl::Context context({ default_device });

	//try
	//{
	//	// load and build the kernel
	//	std::ifstream sourceFile(KERNEL_FILE);
	//	if (!sourceFile)
	//	{
	//		std::cout << "kernel source file " << KERNEL_FILE << " not found!" << std::endl;
	//		return 1;
	//	}
	//	std::string sourceCode(
	//		std::istreambuf_iterator<char>(sourceFile),
	//		(std::istreambuf_iterator<char>()));
	//	cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
	//	program = cl::Program(context, source);
	//	program.build({ default_device });


	//	int size[1] = { INSIZE };

	//	// create buffers on device (allocate space on GPU)
	//	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * INSIZE);
	//	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * INSIZE);

	//	int input[INSIZE];
	//	int output[INSIZE];
	//	srand(time(NULL));
	//	for (int i = 0; i < INSIZE; ++i)
	//	{
	//		input[i] = rand() % 10;
	//		output[i] = 0;
	//	}

	//	// create a queue (a queue of commands that the GPU will execute)
	//	cl::CommandQueue queue(context, default_device);

	//	// push write commands to queue
	//	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*INSIZE, input);
	//	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*INSIZE, output);

	//	// RUN ZE KERNEL
	//	cl::Kernel kernel(program, "blelloch_scan", &err);
	//	kernel.setArg(0, buffer_A);
	//	kernel.setArg(1, buffer_B);
	//	kernel.setArg(2, INSIZE * sizeof(int), NULL);
	//	kernel.setArg(3, INSIZE * sizeof(int), NULL);
	//	kernel.setArg(4, INSIZE);

	//	cl::NDRange global(INSIZE);
	//	cl::NDRange local(16);
	//	queue.enqueueNDRangeKernel(kernel, 0, global, local);


	//	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*INSIZE, output);

	//	std::cout << "INPUT\n" << std::endl;

	//	for (int i = 0; i < INSIZE; ++i)
	//	{
	//		std::cout << input[i] << " | ";
	//	}

	//	std::cout << "\n\nOUTPUT\n";

	//	for (int i = 0; i < INSIZE; ++i)
	//	{
	//		std::cout << output[i] << " \n ";
	//	}

	//	std::cin.get();
	//}
	//catch (cl::Error err)
	//{
	//	std::string s;
	//	program.getBuildInfo(default_device, CL_PROGRAM_BUILD_LOG, &s);
	//	std::cout << s << std::endl;
	//	program.getBuildInfo(default_device, CL_PROGRAM_BUILD_OPTIONS, &s);
	//	std::cout << s << std::endl;

	//	std::cerr
	//		<< "ERROR: "
	//		<< err.what()
	//		<< "("
	//		<< err.err()
	//		<< ")"
	//		<< std::endl;
	//}

	//return err;
}

std::vector<int> stream_compaction_GPU(std::vector<int> input)
{
	// !! ask prof !!
	// since handling everything inside one kernel doesn't work... split it
	// probably slower but i can't figure out why this won't work otherwise
	// !! ask prof !!

	//Filter - gets condition vector
	std::vector<int> filterResult = strComGPU_Step1_Filter(input, 5, "predicateKernel_greater");

	//Scan - build prefix sum for condition vector
	std::vector<int> filterAddresses = strComGPU_Step2_PrefixSum(filterResult);

	//Scatter
	std::vector<int> scatterResult = strComGPU_Step3_Scatter(input, filterAddresses, filterResult);

	return scatterResult;

}

std::vector<int> strComGPU_Step1_Filter(std::vector<int> input, const int threshold, const std::string predicateKernel)
{

	std::vector<int> result(input.size());

	try
	{
		cl::CommandQueue queue(context, default_device, 0, &err);
		// create buffers on device (allocate space on GPU)
		cl::Buffer buffer_INPUT(context, CL_MEM_READ_ONLY, sizeof(cl_int) * input.size());
		cl::Buffer buffer_OUTPUT(context, CL_MEM_READ_WRITE, sizeof(cl_int) * input.size());

		// push write commands to queue
		queue.enqueueWriteBuffer(buffer_INPUT, CL_TRUE, 0, sizeof(cl_int) * input.size(), &input[0]);

		cl::Kernel kernel(program, predicateKernel.c_str(), &err);

		kernel.setArg(0, buffer_INPUT);
		kernel.setArg(1, buffer_OUTPUT);
		kernel.setArg(2, threshold);

		cl::NDRange global(input.size());

		queue.enqueueNDRangeKernel(kernel, 0, global);

		cl::Event readBufferEvent;
		queue.enqueueReadBuffer(buffer_OUTPUT, CL_TRUE, 0, sizeof(cl_int) * input.size(), &result[0], NULL, &readBufferEvent);
		readBufferEvent.wait();
	}
	catch (cl::Error err)
	{
		Errorhandling(err);
	}

	return result;

}

std::vector<int> strComGPU_Step2_PrefixSum(std::vector<int> input)
{
	const std::string KERNEL = "blelloch";
	std::vector<int> result(input.size());

	try
	{
		cl::CommandQueue queue(context, default_device, 0, &err);
		// create buffers on device (allocate space on GPU)
		cl::Buffer buffer_INPUT(context, CL_MEM_READ_ONLY, sizeof(cl_int) * input.size());
		cl::Buffer buffer_OUTPUT(context, CL_MEM_READ_WRITE, sizeof(cl_int) * input.size());

		// push write commands to queue
		queue.enqueueWriteBuffer(buffer_INPUT, CL_TRUE, 0, sizeof(cl_int) * input.size(), &input[0]);

		cl::Kernel kernel(program, KERNEL.c_str(), &err);

		kernel.setArg(0, buffer_INPUT);
		kernel.setArg(1, buffer_OUTPUT);
		kernel.setArg(2, cl::LocalSpaceArg(cl::Local(sizeof(cl_int) * SIZE_BLOCK)));
		kernel.setArg(3, SIZE_BLOCK);

		cl::NDRange global(input.size());
		cl::NDRange local(SIZE_BLOCK);

		queue.enqueueNDRangeKernel(kernel, 1, global, local);

		cl::Event readBufferEvent;
		queue.enqueueReadBuffer(buffer_OUTPUT, CL_TRUE, 0, sizeof(cl_int) * input.size(), &result[0], NULL, &readBufferEvent);
		readBufferEvent.wait();

	}
	catch (cl::Error err)
	{
		Errorhandling(err);
	}

	return result;
}

std::vector<int> strComGPU_Step3_Scatter(std::vector<int> input, std::vector<int> addr, std::vector<int> mask)
{
	const std::string KERNEL = "scatter";
	std::vector<int> result(input.size());

	try
	{
		cl::CommandQueue queue(context, default_device, 0, &err);
		// create buffers on device (allocate space on GPU)
		cl::Buffer buffer_INPUT(context, CL_MEM_READ_ONLY, sizeof(cl_int) * input.size());
		cl::Buffer buffer_OUTPUT(context, CL_MEM_READ_WRITE, sizeof(cl_int) * input.size());
		cl::Buffer buffer_ADDR(context, CL_MEM_READ_ONLY, sizeof(cl_int) * addr.size());
		cl::Buffer buffer_MASK(context, CL_MEM_READ_WRITE, sizeof(cl_int) * mask.size());

		// push write commands to queue
		queue.enqueueWriteBuffer(buffer_INPUT, CL_TRUE, 0, sizeof(cl_int) * input.size(), &input[0]);
		queue.enqueueWriteBuffer(buffer_ADDR, CL_TRUE, 0, sizeof(cl_int) * addr.size(), &addr[0]);
		queue.enqueueWriteBuffer(buffer_MASK, CL_TRUE, 0, sizeof(cl_int) * mask.size(), &mask[0]);

		cl::Kernel kernel(program, KERNEL.c_str(), &err);

		kernel.setArg(0, buffer_INPUT);
		kernel.setArg(1, buffer_ADDR);
		kernel.setArg(2, buffer_MASK);
		kernel.setArg(3, buffer_OUTPUT);

		cl::NDRange global(input.size());

		queue.enqueueNDRangeKernel(kernel, 0, global);

		cl::Event readBufferEvent;
		queue.enqueueReadBuffer(buffer_OUTPUT, CL_TRUE, 0, sizeof(cl_int) * input.size(), &result[0], NULL, &readBufferEvent);
		readBufferEvent.wait();
	}
	catch (cl::Error err)
	{
		Errorhandling(err);
	}

	return result;
}