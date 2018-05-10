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
#include "tga.h"

int main(int argc, char **argv) {
	const std::string KERNEL_FILE = "kernel.cl";
	cl_int err = CL_SUCCESS;
	cl::Program program;
	std::vector<cl::Device> devices;

	try {

		// get available platforms ( NVIDIA, Intel, AMD,...)
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			std::cout << "No OpenCL platforms available!\n";
			return 1;
		}

		// create a context and get available devices
		int platFormNr = 0;
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
		cl::Kernel kernel(program, "image_rotate", &err);
		cl::Event event;
		cl::CommandQueue queue(context, devices[0], 0, &err);
		tga::TGAImage image, imageOutput;
		const char* filename = "1024.tga";

		bool loaded = tga::LoadTGA(&image, filename);
		imageOutput.imageData.resize(image.imageData.size());
		imageOutput.bpp = image.bpp;
		imageOutput.height = image.height;
		imageOutput.type = image.type;
		imageOutput.width = image.width;

		// input buffers
		cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, image.imageData.size() * sizeof(unsigned char));
		cl::Buffer bufferB = cl::Buffer(context, CL_MEM_WRITE_ONLY, image.imageData.size() * sizeof(unsigned char));

		// fill buffers
		queue.enqueueWriteBuffer(
			bufferA, // which buffer to write to
			CL_TRUE, // block until command is complete
			0, // offset
			image.imageData.size() * sizeof(unsigned char), // size of write 
			&image.imageData[0]); // pointer to input
		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, image.imageData.size() * sizeof(unsigned char), &imageOutput.imageData[0]);

		cl::Kernel addKernel(program, "image_rotate", &err);
		addKernel.setArg(0, bufferA);
		addKernel.setArg(1, bufferB);
		addKernel.setArg(2, (int)image.width);
		addKernel.setArg(3, (int)image.height);
		addKernel.setArg(4, 0);
		addKernel.setArg(5, 0);

		// launch add kernel
		// Run the kernel on specific ND range
		cl::NDRange global(1024);
		cl::NDRange local(1); //make sure local range is divisible by global range
		cl::NDRange offset(0);
		std::cout << "Calling 'image_rotate' kernel" << std::endl;
		queue.enqueueNDRangeKernel(addKernel, offset, global, local);

		// read back result
		queue.enqueueReadBuffer(bufferB, CL_TRUE, 0,
			imageOutput.imageData.size() * sizeof(unsigned char), &imageOutput.imageData[0]);

		tga::saveTGA(imageOutput, "output.tga");

		std::cout << "Exported image!";
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

	std::string str;
	std::getline(std::cin, str);
	return err;
}
