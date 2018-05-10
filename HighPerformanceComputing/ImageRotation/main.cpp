// Author: Markus Schordan, 2011.

#include "CL/cl.h"
#include <malloc.h>
#include <iostream>
#include <string>
#include <sstream>

// list of error codes from "CL/cl.h"
std::string cl_errorstring(cl_int err) {
	switch (err) {
	case CL_SUCCESS:                          return std::string("Success");
	case CL_DEVICE_NOT_FOUND:                 return std::string("Device not found");
	case CL_DEVICE_NOT_AVAILABLE:             return std::string("Device not available");
	case CL_COMPILER_NOT_AVAILABLE:           return std::string("Compiler not available");
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return std::string("Memory object allocation failure");
	case CL_OUT_OF_RESOURCES:                 return std::string("Out of resources");
	case CL_OUT_OF_HOST_MEMORY:               return std::string("Out of host memory");
	case CL_PROFILING_INFO_NOT_AVAILABLE:     return std::string("Profiling information not available");
	case CL_MEM_COPY_OVERLAP:                 return std::string("Memory copy overlap");
	case CL_IMAGE_FORMAT_MISMATCH:            return std::string("Image format mismatch");
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return std::string("Image format not supported");
	case CL_BUILD_PROGRAM_FAILURE:            return std::string("Program build failure");
	case CL_MAP_FAILURE:                      return std::string("Map failure");
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:     return std::string("Misaligned sub buffer offset");
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return std::string("Exec status error for events in wait list");
	case CL_INVALID_VALUE:                    return std::string("Invalid value");
	case CL_INVALID_DEVICE_TYPE:              return std::string("Invalid device type");
	case CL_INVALID_PLATFORM:                 return std::string("Invalid platform");
	case CL_INVALID_DEVICE:                   return std::string("Invalid device");
	case CL_INVALID_CONTEXT:                  return std::string("Invalid context");
	case CL_INVALID_QUEUE_PROPERTIES:         return std::string("Invalid queue properties");
	case CL_INVALID_COMMAND_QUEUE:            return std::string("Invalid command queue");
	case CL_INVALID_HOST_PTR:                 return std::string("Invalid host pointer");
	case CL_INVALID_MEM_OBJECT:               return std::string("Invalid memory object");
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return std::string("Invalid image format descriptor");
	case CL_INVALID_IMAGE_SIZE:               return std::string("Invalid image size");
	case CL_INVALID_SAMPLER:                  return std::string("Invalid sampler");
	case CL_INVALID_BINARY:                   return std::string("Invalid binary");
	case CL_INVALID_BUILD_OPTIONS:            return std::string("Invalid build options");
	case CL_INVALID_PROGRAM:                  return std::string("Invalid program");
	case CL_INVALID_PROGRAM_EXECUTABLE:       return std::string("Invalid program executable");
	case CL_INVALID_KERNEL_NAME:              return std::string("Invalid kernel name");
	case CL_INVALID_KERNEL_DEFINITION:        return std::string("Invalid kernel definition");
	case CL_INVALID_KERNEL:                   return std::string("Invalid kernel");
	case CL_INVALID_ARG_INDEX:                return std::string("Invalid argument index");
	case CL_INVALID_ARG_VALUE:                return std::string("Invalid argument value");
	case CL_INVALID_ARG_SIZE:                 return std::string("Invalid argument size");
	case CL_INVALID_KERNEL_ARGS:              return std::string("Invalid kernel arguments");
	case CL_INVALID_WORK_DIMENSION:           return std::string("Invalid work dimension");
	case CL_INVALID_WORK_GROUP_SIZE:          return std::string("Invalid work group size");
	case CL_INVALID_WORK_ITEM_SIZE:           return std::string("Invalid work item size");
	case CL_INVALID_GLOBAL_OFFSET:            return std::string("Invalid global offset");
	case CL_INVALID_EVENT_WAIT_LIST:          return std::string("Invalid event wait list");
	case CL_INVALID_EVENT:                    return std::string("Invalid event");
	case CL_INVALID_OPERATION:                return std::string("Invalid operation");
	case CL_INVALID_GL_OBJECT:                return std::string("Invalid OpenGL object");
	case CL_INVALID_BUFFER_SIZE:              return std::string("Invalid buffer size");
	case CL_INVALID_MIP_LEVEL:                return std::string("Invalid mip-map level");
	case CL_INVALID_GLOBAL_WORK_SIZE:         return std::string("Invalid gloal work size");
	case CL_INVALID_PROPERTY:                 return std::string("Invalid property");
	default:                                  return std::string("Unknown error code");
	}
}

void handle_clerror(cl_int err) {
	if (err != CL_SUCCESS) {
		std::cerr << "OpenCL Error: " << cl_errorstring(err) << std::string(".") << std::endl;
		exit(EXIT_FAILURE);
	}
}

void print_separation_line(std::string s, int empty = 0) {
	for (int i = empty; i < 0; i++) {
		std::cout << std::endl;
	}
	int length = 79;
	int n = length / s.size();
	for (int i = 0; i < n; i++) {
		std::cout << s;
	}
	int rest = length % s.size();
	for (int i = 0; i < rest; i++) {
		std::cout << s[i];
	}
	std::cout << std::endl;
	for (int i = 0; i < empty; i++) {
		std::cout << std::endl;
	}
}

void print_name(std::string s, int fieldwidth) {
	std::cout.setf(std::ios::left);
	std::cout.width(fieldwidth);
	std::cout << s << ": ";
}
void print_pname(std::string s) {
	print_name(s, 20);
}
void print_dname(std::string s) {
	print_name(s, 31);
}

int main(int argc, char **argv) {
	cl_int err;
	cl_uint numPlatforms;
	cl_platform_id* platformIds;
	cl_uint numDevices;
	cl_device_id* deviceIds;

	print_separation_line("=");
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	handle_clerror(err);
	std::cout << "Number of platforms: " << numPlatforms << std::endl;
	platformIds = (cl_platform_id*)calloc(numPlatforms, sizeof(cl_platform_id));
	err = clGetPlatformIDs(numPlatforms, platformIds, NULL);
	handle_clerror(err);
	for (unsigned int i = 0; i < numPlatforms; i++) {
		print_separation_line("=");
		std::cout << "PLATFORM:" << i << std::endl;
		print_separation_line("=");
		print_pname("Platform ID"); std::cout << platformIds[i] << std::endl;
		cl_platform_id id = platformIds[i];
		size_t size;
		err = clGetPlatformInfo(id, CL_PLATFORM_VENDOR, 0, NULL, &size);
		handle_clerror(err);
		char* vendor = (char*)calloc(size, sizeof(char));
		err = clGetPlatformInfo(id, CL_PLATFORM_VENDOR, size, vendor, NULL);
		handle_clerror(err);
		print_pname("Vendor name"); std::cout << vendor << std::endl;

		err = clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, NULL, &size);
		handle_clerror(err);
		char* pname = (char*)calloc(size, sizeof(char));
		err = clGetPlatformInfo(id, CL_PLATFORM_NAME, size, pname, NULL);
		handle_clerror(err);
		print_pname("Platform name"); std::cout << pname << std::endl;

		err = clGetPlatformInfo(id, CL_PLATFORM_VERSION, 0, NULL, &size);
		handle_clerror(err);
		char* version = (char*)calloc(size, sizeof(char));
		err = clGetPlatformInfo(id, CL_PLATFORM_VERSION, size, version, NULL);
		handle_clerror(err);
		print_pname("Platform version"); std::cout << version << std::endl;

		err = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		handle_clerror(err);
		print_pname("Number of devices"); std::cout << numDevices << std::endl;
		deviceIds = (cl_device_id*)calloc(numDevices, sizeof(cl_platform_id));
		err = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, numDevices, deviceIds, NULL);
		handle_clerror(err);

		err = clGetPlatformInfo(id, CL_PLATFORM_EXTENSIONS, 0, NULL, &size);
		handle_clerror(err);
		char* extensions = (char*)calloc(size, sizeof(char));
		err = clGetPlatformInfo(id, CL_PLATFORM_EXTENSIONS, size, extensions, NULL);
		handle_clerror(err);
		print_pname("Platform extensions"); std::cout << extensions << std::endl;

		for (unsigned int j = 0; j < numDevices; j++) {
			print_separation_line("-");
			std::cout << "DEVICE:" << j << "  [PLATFORM:" << i << "]" << std::endl;
			print_separation_line("-");
			print_dname("  Device ID"); std::cout << deviceIds[j] << std::endl;
			cl_device_id deviceId = deviceIds[j];

			cl_uint maxComputeUnits = 0;
			err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);
			handle_clerror(err);
			print_dname("  Max compute units"); std::cout << maxComputeUnits << std::endl;

			cl_uint maxWorkItemDimensions = 0;
			err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWorkItemDimensions, NULL);
			handle_clerror(err);
			print_dname("  Max work item dimensions"); std::cout << maxWorkItemDimensions << " ";

			size_t* workItemSizes;
			workItemSizes = (size_t*)calloc(maxWorkItemDimensions, sizeof(size_t));
			err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemDimensions * sizeof(size_t), workItemSizes, NULL);
			handle_clerror(err);
			std::cout << std::endl;

			size_t maxWorkGroupSize = 0;
			err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
			handle_clerror(err);
			print_dname("  Max work group size"); std::cout << maxWorkGroupSize << std::endl;

			cl_int deviceVendorId = 0;
			err = clGetDeviceInfo(deviceId, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &deviceVendorId, NULL);
			handle_clerror(err);
			print_dname("  Device vendor id"); std::cout << deviceVendorId << std::endl;

			cl_uint maxClockFrequency = 0;
			err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &maxClockFrequency, NULL);
			handle_clerror(err);
			print_dname("  Max clock frequency"); std::cout << maxClockFrequency << std::endl;

			cl_uint deviceAddressBits = 0;
			err = clGetDeviceInfo(deviceId, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &deviceAddressBits, NULL);
			handle_clerror(err);
			print_dname("  Device address bits"); std::cout << deviceAddressBits << std::endl;

			cl_ulong deviceMaxMemAllocSize = 0;
			err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &deviceMaxMemAllocSize, NULL);
			handle_clerror(err);
			print_dname("  Device max memory alloc size"); std::cout << deviceMaxMemAllocSize << std::endl;
		}
		print_separation_line("-", 1);
	}
	print_separation_line("=");

	std::string str;
	std::getline(std::cin, str);
	return err;
}
