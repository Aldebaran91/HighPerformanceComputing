
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
#include "main.h"

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
	const int INSIZE = 1024;
	const std::string KERNEL_FILE = "kernel.cl";
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	cl::Program program;


	if (all_platforms.size() == 0)
	{
		std::cout << " No platforms found. Check OpenCL installation!\n";
		return err;
	}

	cl::Platform default_platform = all_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	// get default device (CPUs, GPUs) of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}

	// use device[1] because that's a GPU; device[0] is the CPU
	cl::Device default_device = all_devices[0];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	// a context is like a "runtime link" to the device and platform;
	// i.e. communication is possible
	cl::Context context({ default_device });

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


		int size[1] = { INSIZE };

		// create buffers on device (allocate space on GPU)
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * INSIZE);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * INSIZE);
		cl::Buffer buffer_T(context, CL_MEM_READ_WRITE, sizeof(int) * INSIZE);

		int input[INSIZE];
		int output[INSIZE];
		int temp[INSIZE];
		for (int i = 0; i < INSIZE; ++i)
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
		kernel.setArg(2, INSIZE);
		kernel.setArg(3, INSIZE * sizeof(int), NULL);

		cl::NDRange global(INSIZE);
		cl::NDRange local(16);
		queue.enqueueNDRangeKernel(kernel, 0, global, local);


		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*INSIZE, output);

		std::cout << "INPUT\n" << std::endl;

		/*for (int i = 0; i < INSIZE; ++i)
		{
			std::cout << input[i] << " | ";
		}*/

		std::cout << "\n\nOUTPUT\n";

		for (int i = 0; i < INSIZE; ++i)
		{
			std::cout << output[i] << " \n ";
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

//cl_int err = CL_SUCCESS;
//const cl_uint INPUT_SIZE = 64;
//const std::string KERNEL_FILE = "kernel.cl";
//std::vector<cl::Device> devices;
//cl::Program program;
//
////input.assign({ 3, 1, 7, 0, 4, 1, 6, 3 });
//
//int *input = new int[INPUT_SIZE];
//int *output = new int[INPUT_SIZE];
//for (int i = 0; i < INPUT_SIZE; ++i)
//{
//	input[i] = i;
//	output[i] = 0;
//}
//
//try
//{
//	// get available platforms ( NVIDIA, Intel, AMD,...)
//	std::vector<cl::Platform> platforms;
//	cl::Platform::get(&platforms);
//	if (platforms.size() == 0) {
//		std::cout << "No OpenCL platforms available!\n";
//		return 1;
//	}
//
//	// create a context and get available devices
//	int platFormNr = 0;
//	cl::Platform platform = platforms[platFormNr]; // on a different machine, you may have to select a different platform!
//	cl_context_properties properties[] =
//	{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[platFormNr])(), 0 };
//	cl::Context context(CL_DEVICE_TYPE_ALL, properties);
//
//	devices = context.getInfo<CL_CONTEXT_DEVICES>();
//
//
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
//	program.build(devices);
//
//	//create kernels
//	cl::CommandQueue queue(context, devices[0], 0, &err);
//
//	//buffers
//	cl::Buffer buffer_input = cl::Buffer(context, CL_MEM_READ_WRITE, INPUT_SIZE * sizeof(int));
//	cl::Buffer buffer_output = cl::Buffer(context, CL_MEM_READ_WRITE, INPUT_SIZE * sizeof(int));
//
//	// fill buffers
//	queue.enqueueWriteBuffer(
//		buffer_input, // which buffer to write to
//		CL_TRUE, // block until command is complete
//		0, // offset
//		INPUT_SIZE * sizeof(int), // size of write
//		input); // pointer to input
//	queue.enqueueWriteBuffer(buffer_output, CL_TRUE, 0, INPUT_SIZE * sizeof(int), output);
//
//	cl::Kernel kernel(program, "blelloch", &err);
//	kernel.setArg(0, buffer_input);
//	kernel.setArg(1, buffer_output);
//	kernel.setArg(2, 16);
//	kernel.setArg(3, INPUT_SIZE, NULL);
//
//	// launch add kernel
//	// Run the kernel on specific ND range
//	cl::NDRange global(INPUT_SIZE);
//	cl::NDRange local(256);
//	std::cout << "nvidia Scan Sum" << std::endl;
//	queue.enqueueNDRangeKernel(kernel, 0, global, local);
//
//	// read back result
//	queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, INPUT_SIZE * sizeof(int), output);