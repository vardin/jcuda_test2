package jcuda_test;


import java.io.IOException;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;


import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

import jcuda.samples.utils.JCudaSamplesUtils;

public class jcuda_test {
	
	
	public static void main(String[] args) {
		
		
		// Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);

        // Create the PTX file by calling the NVCC
        String ptxFileName = JCudaSamplesUtils.preparePtxFile(
            "src/main/resources/kernels/JCudaVectorMatrixMultiplication.cu");

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "add" function.
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "multiplication");

        
        int m =23;
        int n = 23;
        int numElements = m*n;

        // Allocate and fill the host input data
        int hostInputA[] = new int[numElements];
        int hostInputB[] = new int[numElements];
        int hostOutput[] = new int[numElements];
        for(int i = 0; i < numElements; i++)
        {
            hostInputA[i] = i;
            hostInputB[i] = i;
        }

        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputA = new CUdeviceptr();
        cuMemAlloc(deviceInputA, numElements * Sizeof.INT);
        CUdeviceptr deviceInputB = new CUdeviceptr();
        cuMemAlloc(deviceInputB, numElements * Sizeof.INT);
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, numElements * Sizeof.INT);

        
        long start = System.currentTimeMillis(); 
                 
        cuMemcpyHtoD(deviceInputA, Pointer.to(hostInputA),
            numElements * Sizeof.INT);
                  
        cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputB),
            numElements * Sizeof.INT);

        // Allocate device output memory
      
        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
          

        Pointer kernelParameters = Pointer.to(Pointer.to(deviceInputA),
					Pointer.to(deviceInputB), Pointer.to(deviceOutput),Pointer.to(new int[] { m }));

        // Call the kernel function.
     
        
        int block_size = 16;
 //     int gridSizeX = (int)Math.ceil((double)numElements / blockSizeX);
      
    	int gridSizeX = (m+block_size-1)/block_size;	 //the number of block
        cuLaunchKernel(function,
            gridSizeX,  gridSizeX, 1,      // Grid dimension    //number of block
            block_size, block_size, 1,      // Block dimension   //number of thread
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
       
      
        // Allocate host output memory and copy the device output
        // to the host.
  
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
            numElements * Sizeof.INT);
        long end = System.currentTimeMillis();
        System.out.println( "실행 시간 : " + ( end - start )+"ms" );
        // Verify the result000000000000000000
       
        System.out.println(hostOutput[1]);
        
        // Clean up.
        cuMemFree(deviceInputA);
        cuMemFree(deviceInputB);
        cuMemFree(deviceOutput);
    }
			
	//utf-8
}
