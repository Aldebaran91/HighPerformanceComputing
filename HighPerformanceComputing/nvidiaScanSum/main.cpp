// opencltest.cpp : an example for using OpenCL with C++
// requires cl.hpp, the C++ bindings for OpenCL v 1.2
// https://www.khronos.org/registry/cl/api/1.2/cl.hpp
// author: Eugen Jiresch
//

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

int main(int argc, char **argv) {
	const std::string KERNEL_FILE = "kernel.cl";
	cl_int err = CL_SUCCESS;
	cl::Program program;
	std::vector<cl::Device> devices;

	try {
		std::vector<int> input, output;
		input.assign({ 3, 1, 7, 0, 4, 1, 6, 3 });
		output.resize(input.size());

		// get available platforms ( NVIDIA, Intel, AMD,...)
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			std::cout << "No OpenCL platforms available!\n";
			return 1;
		}

		// create a context and get available devices
		int platFormNr = 1;
		cl::Platform platform = platforms[platFormNr]; // on a different machine, you may have to select a different platform!
		cl_context_properties properties[] =
		{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[platFormNr])(), 0 };
		cl::Context context(CL_DEVICE_TYPE_ALL, properties);

		devices = context.getInfo<CL_CONTEXT_DEVICES>();

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
		//create kernels
		cl::Kernel kernel(program, "scan", &err);
		cl::Event event;
		cl::CommandQueue queue(context, devices[0], 0, &err);

		// input buffers
		cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, input.size() * sizeof(int));
		cl::Buffer bufferB = cl::Buffer(context, CL_MEM_WRITE_ONLY, output.size() * sizeof(int));

		// fill buffers
		queue.enqueueWriteBuffer(
			bufferA, // which buffer to write to
			CL_TRUE, // block until command is complete
			0, // offset
			input.size() * sizeof(int), // size of write
			&input[0]); // pointer to input
		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, input.size() * sizeof(int), &input[0]);

		cl::Kernel addKernel(program, "scan", &err);
		addKernel.setArg(0, bufferA);
		addKernel.setArg(1, bufferB);
		addKernel.setArg(2, input.size(), NULL);
		addKernel.setArg(3, input.size());

		// launch add kernel
		// Run the kernel on specific ND range
		cl::NDRange global(input.size());
		cl::NDRange local(input.size());
		std::cout << "nvidia Scan Sum" << std::endl;
		queue.enqueueNDRangeKernel(addKernel, 0, global, local);

		// read back result
		queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, output.size() * sizeof(int), &output[0]);
		std::cout << "Reading result" << std::endl;
	}
	catch (cl::Error err) {
		// error handling
		// if the kernel has failed to compile, print the error log
		std::string s;
		program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &s);
		std::cout << s << std::endl;
		program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_OPTIONS, &s);
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