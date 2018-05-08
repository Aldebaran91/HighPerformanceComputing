using System;
using System.Collections.Generic;
using System.Linq;

namespace ConsoleApp1
{
    internal class Program
    {
        public static void WriteText(string text, ConsoleColor color, bool writeLine = false)
        {
            Console.ForegroundColor = color;
            if (writeLine)
                Console.WriteLine(text);
            else
                Console.Write(text);
            Console.ResetColor();
        }

        public static void WriteTableRow(bool isHead, int[] columnWidth, params string[] column)
        {
            if (columnWidth.Length != column.Length)
                return;

            if (isHead)
            {
                for (int i = 0; i < (columnWidth.Sum(x => x + 3)) + 1; i++)
                    Console.Write("_");
                Console.WriteLine();
            }
            for (int i = 0; i < columnWidth.Length; i++)
            {
                Console.Write(" | ");
                if (isHead)
                    WriteText(String.Format($"{{0,{columnWidth[i]}}}", column[i]), ConsoleColor.Magenta);
                else
                    WriteText(String.Format($"{{0,{columnWidth[i]}}}", column[i]), ConsoleColor.White);
            }
            Console.WriteLine(" | ");
        }

        private void CheckErr(OpenCL.Net.ErrorCode err, string name)
        {
            if (err != OpenCL.Net.ErrorCode.Success)
            {
                Console.WriteLine("ERROR: " + name + " (" + err.ToString() + ")");
            }
        }

        private void ContextNotify(string errInfo, byte[] data, IntPtr cb, IntPtr userData)
        {
            Console.WriteLine("OpenCL Notification: " + errInfo);
        }

        public void Launch()
        {
            OpenCL.Net.ErrorCode error;

            OpenCL.Net.Platform[] platforms = OpenCL.Net.Cl.GetPlatformIDs(out error);
            List<OpenCL.Net.Device> devicesList = new List<OpenCL.Net.Device>();

            CheckErr(error, "Cl.GetPlatformIDs");

            int[] columns = new int[] { 50, 25, 30 };
            WriteTableRow(true, columns, "Platform", "Version", "Vendor");
            foreach (OpenCL.Net.Platform platform in platforms)
            {
                string platformName = OpenCL.Net.Cl.GetPlatformInfo(platform, OpenCL.Net.PlatformInfo.Name, out error).ToString();

                WriteTableRow(false,
                    columns,
                    platformName,
                    OpenCL.Net.Cl.GetPlatformInfo(platform, OpenCL.Net.PlatformInfo.Version, out error).ToString(),
                    OpenCL.Net.Cl.GetPlatformInfo(platform, OpenCL.Net.PlatformInfo.Vendor, out error).ToString());
                CheckErr(error, "Cl.GetPlatformInfo");
                //We will be looking only for GPU devices
                foreach (OpenCL.Net.Device dev in OpenCL.Net.Cl.GetDeviceIDs(platform, OpenCL.Net.DeviceType.All, out error))
                {
                    CheckErr(error, "Cl.GetDeviceIDs");
                    devicesList.Add(dev);
                }
            }

            if (devicesList.Count <= 0)
            {
                Console.WriteLine("No devices found.");
                return;
            }

            OpenCL.Net.Device device = devicesList[0];

            if (OpenCL.Net.Cl.GetDeviceInfo(device, OpenCL.Net.DeviceInfo.ImageSupport,
                      out error).CastTo<OpenCL.Net.Bool>() == OpenCL.Net.Bool.False)
            {
                Console.WriteLine("No image support.");
                return;
            }

            OpenCL.Net.Context context = OpenCL.Net.Cl.CreateContext(null, 1, new[] { device }, ContextNotify, IntPtr.Zero, out error);    //Second parameter is amount of devices
            CheckErr(error, "Cl.CreateContext");

            //Load and compile kernel source code.
            string programPath = Environment.CurrentDirectory + "/../../kernel.cl";
            //The path to the source file may vary

            if (!System.IO.File.Exists(programPath))
            {
                Console.WriteLine("Program doesn't exist at path " + programPath);
                return;
            }

            string programSource = System.IO.File.ReadAllText(programPath);

            OpenCL.Net.Program program = OpenCL.Net.Cl.CreateProgramWithSource(context, 1, new[] { programSource }, null, out error);
            CheckErr(error, "Cl.CreateProgramWithSource");

            //Compile kernel source
            error = OpenCL.Net.Cl.BuildProgram(program, 1, new[] { device }, string.Empty, null, IntPtr.Zero);
            CheckErr(error, "Cl.BuildProgram");

            //Check for any compilation errors
            if (OpenCL.Net.Cl.GetProgramBuildInfo(program, device, OpenCL.Net.ProgramBuildInfo.Status, out error).CastTo<OpenCL.Net.BuildStatus>()
                != OpenCL.Net.BuildStatus.Success)
            {
                CheckErr(error, "Cl.GetProgramBuildInfo");
                Console.WriteLine("Cl.GetProgramBuildInfo != Success");
                Console.WriteLine(OpenCL.Net.Cl.GetProgramBuildInfo(program, device, OpenCL.Net.ProgramBuildInfo.Log, out error));
                return;
            }

            //Create the required kernel (entry function)
            OpenCL.Net.Kernel kernel = OpenCL.Net.Cl.CreateKernel(program, "vector_add", out error);
            CheckErr(error, "Cl.CreateKernel");

            //Create a command queue, where all of the commands for execution will be added
            OpenCL.Net.CommandQueue cmdQueue = OpenCL.Net.Cl.CreateCommandQueue(context, device, (OpenCL.Net.CommandQueueProperties)0, out error);
            CheckErr(error, "Cl.CreateCommandQueue");

            OpenCL.Net.Event clevent;
            IntPtr[] workGroupSizePtr = new IntPtr[] { (IntPtr)8, (IntPtr)1, (IntPtr)1 };

            int[] a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            int[] b = new int[] { 1, 4, 6, 8, 10, 12, 14, 16 };

            OpenCL.Net.IMem bufferA = OpenCL.Net.Cl.CreateBuffer(context, OpenCL.Net.MemFlags.ReadOnly, new IntPtr(sizeof(int) * 8), out error);
            CheckErr(error, "CL.CreateBuffer");
            OpenCL.Net.IMem bufferB = OpenCL.Net.Cl.CreateBuffer(context, OpenCL.Net.MemFlags.ReadOnly, new IntPtr(sizeof(int) * 8), out error);
            CheckErr(error, "CL.CreateBuffer");
            OpenCL.Net.IMem bufferC = OpenCL.Net.Cl.CreateBuffer(context, OpenCL.Net.MemFlags.WriteOnly, new IntPtr(sizeof(int) * 8), out error);
            CheckErr(error, "CL.CreateBuffer");

            // Copy our host buffer of values to the device buffer
            error = OpenCL.Net.Cl.EnqueueWriteBuffer(cmdQueue, bufferA, OpenCL.Net.Bool.True, IntPtr.Zero, new IntPtr(sizeof(int) * 8), a, 0, null, out clevent);
            CheckErr(error, "CL.EnqueueWriteBuffer");
            error = OpenCL.Net.Cl.EnqueueWriteBuffer(cmdQueue, bufferB, OpenCL.Net.Bool.True, IntPtr.Zero, new IntPtr(sizeof(int) * 8), b, 0, null, out clevent);
            CheckErr(error, "CL.EnqueueWriteBuffer");

            // Set kernel arguments
            OpenCL.Net.Cl.SetKernelArg(kernel, 0, bufferA);
            OpenCL.Net.Cl.SetKernelArg(kernel, 1, bufferB);
            OpenCL.Net.Cl.SetKernelArg(kernel, 2, bufferC);

            //Execute our kernel (OpenCL code)
            error = OpenCL.Net.Cl.EnqueueNDRangeKernel(cmdQueue, kernel, 1, null, workGroupSizePtr, null, 0, null, out clevent);
            CheckErr(error, "CL.EnqueueNDRangeKernel");
            OpenCL.Net.Cl.Finish(cmdQueue);

            // Read back the results
            int[] results = new int[8];
            error = OpenCL.Net.Cl.EnqueueReadBuffer(cmdQueue, bufferC, OpenCL.Net.Bool.True, IntPtr.Zero, new IntPtr(8 * sizeof(int)), results, 0, null, out clevent);
            CheckErr(error, "CL.EnqueueReadBuffer");

            // Validate our results
            Console.Write($"\n-> Test\nOutput: {string.Join(" ", results)}\n");

            Console.ReadLine();

            // Clean up
            error = OpenCL.Net.Cl.Flush(cmdQueue);
            error = OpenCL.Net.Cl.ReleaseMemObject(bufferA);
            error = OpenCL.Net.Cl.ReleaseMemObject(bufferB);
            error = OpenCL.Net.Cl.ReleaseMemObject(bufferC);
            error = OpenCL.Net.Cl.ReleaseCommandQueue(cmdQueue);
            error = OpenCL.Net.Cl.ReleaseKernel(kernel);
            error = OpenCL.Net.Cl.ReleaseProgram(program);
            error = OpenCL.Net.Cl.ReleaseContext(context);
        }

        private static void Main(string[] args)
        {
            Program prog = new Program();
            prog.Launch();
        }
    }
}