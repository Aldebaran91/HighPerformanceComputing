
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
#include <math.h>

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
struct PrefixSumResult
{
	std::vector<int> result;
	std::vector<int> groupSums;
};
std::vector<int> stream_compaction_GPU(std::vector<int> input);
std::vector<int> strComGPU_Step1_Filter(std::vector<int> input, const int threshold, const std::string predicateKernel);
std::vector<int> strComGPU_Step2_PrefixSum(std::vector<int> input);
std::vector<int> strComGPU_Step3_Scatter(std::vector<int> input, std::vector<int> addr, std::vector<int> mask);
PrefixSumResult CalcPrefixSum(std::vector<int> input);
std::vector<int> ApplyGroupSums(std::vector<int> input, std::vector<int> groupSums);



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
		std::vector<int> input = generateRandomInput(1024);
		auto output = stream_compaction_GPU(input);
	}
	catch (cl::Error err)
	{
		Errorhandling(err);
	}
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
	PrefixSumResult sumResult_Base = CalcPrefixSum(input);
	PrefixSumResult sumResult_GroupSums = CalcPrefixSum(sumResult_Base.groupSums);

	std::vector<int> result = ApplyGroupSums(sumResult_Base.result, sumResult_GroupSums.result);
	return result;
}

PrefixSumResult CalcPrefixSum(std::vector<int> input)
{
	const std::string KERNEL = "blelloch";
	PrefixSumResult sumResult;
	sumResult.result = std::vector<int>(input.size());
	sumResult.groupSums = std::vector<int>(ceil(input.size() / SIZE_WG));

	try
	{
		cl::CommandQueue queue(context, default_device, 0, &err);
		// create buffers on device (allocate space on GPU)
		cl::Buffer buffer_INPUT(context, CL_MEM_READ_ONLY, sizeof(cl_int) * input.size());
		cl::Buffer buffer_OUTPUT(context, CL_MEM_READ_WRITE, sizeof(cl_int) * input.size());
		cl::Buffer buffer_GROUPSUMS(context, CL_MEM_READ_WRITE, sizeof(cl_int) * ceil(input.size() / SIZE_WG));

		// push write commands to queue
		queue.enqueueWriteBuffer(buffer_INPUT, CL_TRUE, 0, sizeof(cl_int) * input.size(), &input[0]);

		cl::Kernel kernel(program, KERNEL.c_str(), &err);

		kernel.setArg(0, buffer_INPUT);
		kernel.setArg(1, buffer_OUTPUT);
		kernel.setArg(2, buffer_GROUPSUMS);
		kernel.setArg(3, cl::LocalSpaceArg(cl::Local(sizeof(cl_int) * SIZE_BLOCK)));
		kernel.setArg(4, SIZE_BLOCK);

		cl::NDRange global(input.size());
		cl::NDRange local(SIZE_BLOCK);

		queue.enqueueNDRangeKernel(kernel, 1, global, local);

		cl::Event readBufferEvent_output;
		queue.enqueueReadBuffer(buffer_OUTPUT, CL_TRUE, 0, sizeof(cl_int) * input.size(), &sumResult.result[0], NULL, &readBufferEvent_output);
		readBufferEvent_output.wait();
		cl::Event readBufferEvent_groupSums;
		queue.enqueueReadBuffer(buffer_GROUPSUMS, CL_TRUE, 0, sizeof(cl_int) * (input.size() / SIZE_WG), &sumResult.groupSums[0], NULL, &readBufferEvent_groupSums);
		readBufferEvent_groupSums.wait();

	}
	catch (cl::Error err)
	{
		Errorhandling(err);
	}

	return sumResult;
}

std::vector<int> ApplyGroupSums(std::vector<int> input, std::vector<int> groupSums)
{
	const std::string KERNEL = "ApplyGroupSums";
	std::vector<int> result(input.size());

	try
	{
		cl::CommandQueue queue(context, default_device, 0, &err);
		// create buffers on device (allocate space on GPU)
		cl::Buffer buffer_INPUT(context, CL_MEM_READ_ONLY, sizeof(cl_int) * input.size());
		cl::Buffer buffer_OUTPUT(context, CL_MEM_READ_WRITE, sizeof(cl_int) * input.size());
		cl::Buffer buffer_SUMS(context, CL_MEM_READ_ONLY, sizeof(cl_int) * groupSums.size());

		// push write commands to queue
		queue.enqueueWriteBuffer(buffer_INPUT, CL_TRUE, 0, sizeof(cl_int) * input.size(), &input[0]);
		queue.enqueueWriteBuffer(buffer_SUMS, CL_TRUE, 0, sizeof(cl_int) * groupSums.size(), &groupSums[0]);

		cl::Kernel kernel(program, KERNEL.c_str(), &err);

		kernel.setArg(0, buffer_INPUT);
		kernel.setArg(1, buffer_SUMS);
		kernel.setArg(2, buffer_OUTPUT);

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