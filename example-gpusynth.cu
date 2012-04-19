// includes, system
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <shrUtils.h>
#include <cutil_inline.h>

#include "heat_kernel.cuh"

using namespace std;



__global__ void heat_kernel_hwblk_main(
    int nblk_input,
    int n_input,
    int *input,
    int nblk_output,
    int n_output,
    int *output,

    int nsteps)
{
    extern __shared__ int sharedmem[];

    
    heat_kernel(
        nblk_input,
        n_input,
        input,
        nblk_output,
        n_output,
        output,

        blockIdx.x,
        sharedmem,
        nsteps
        );
}





int main(int argc, char** argv) {
    char* s_fname;
    char* r_fname;
    char* r_gold_fname;
    uint hTimer;
    cutCreateTimer(&hTimer);
    const int iCycles = 3000;

    // use command-line specified CUDA device,
    // otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    bool check_spec = true;

    // file names, either specified as cmd line args or use default
    if ((cutGetCmdLineArgumentstr(argc,
            (const char**)argv, "signal", &s_fname) != CUTTrue) ||
        (cutGetCmdLineArgumentstr(argc,
            (const char**)argv, "result", &r_fname) != CUTTrue) ||
        (cutGetCmdLineArgumentstr(argc,
            (const char**)argv, "gold", &r_gold_fname) != CUTTrue)) 
    {
        check_spec = false;
    }

    // read in signal
    unsigned int slength = 0;
    int* signal = NULL;

    if (check_spec) {
        if (s_fname == 0) {
            fprintf(stderr, "Cannot find the file containing the signal.\n");
            cudaThreadExit();
            exit(1);
        }
        if (cutReadFilei(s_fname, &signal, &slength) == CUTTrue) {
            printf("Read signal from %s, length = %d\n", s_fname, slength);
        } else {
            cudaThreadExit();
            exit(1);
        }
    }

    int *d_input = NULL;
    cutilSafeCall( cudaMalloc( (void**) &d_input, 33554432));
    int *d_output = NULL;
    cutilSafeCall( cudaMalloc( (void**) &d_output, 33554432));
    int *d_tmp_1 = NULL;
    cutilSafeCall( cudaMalloc( (void**) &d_tmp_1, 33554432));

    // copy input data to device
    if (check_spec) {
        cutilCondition(slength >= 8388608);
        cutilSafeCall( cudaMemcpy( d_input, signal, 33554432, 
            cudaMemcpyHostToDevice) ); 
    }

    // clear result memory
    cutilSafeCall( cudaMemset( d_output, 0, 33554432) ); 

    {
        cutilSafeCall( cudaThreadSynchronize() );
        shrLog("*** Running test on %d bytes of input ...", 33554432);
        cutResetTimer(hTimer);
        cutStartTimer(hTimer);
    }




    /// === kernels

    for (int a = 0; a < iCycles; a++) {
            // hardware tiled version
            heat_kernel_hwblk_main
                <<< 32768,
                    64,
                    1024>>>
                (256, 8388608, d_input, 256, 8388608, d_tmp_1, 1);
    }




    {
        cutilSafeCall( cudaThreadSynchronize() );
        cutStopTimer(hTimer);
        double timerValue = 1.0e-3 *
            cutGetTimerValue(hTimer) / iCycles;
        double microseconds = 1.0e6 * timerValue;
        shrLog(" time = %f microsecs.\n", microseconds);
        shrLog("    bandwidth: %.3f GB/s.\n",
            0.00093132 * 33554432 / microseconds);
    }





    if (check_spec) {
        // get the result back from the server
        // allocate mem for the result
        int* odata = (int*) malloc(33554432);
        cutilSafeCall( cudaMemcpy( odata, d_output,
            33554432, cudaMemcpyDeviceToHost));


        // post processing
        {
            // write file for regression test
            if (r_fname == 0) {
                fprintf(stderr, "Cannot write the output file storing"
                    " the result of the wavelet decomposition.\n");
                cudaThreadExit();
                exit(1);
            }
            if (cutWriteFilei(r_fname, odata, 8388608, false) == CUTTrue) 
                printf("Writing result to %s\n", r_fname);
            else {
                cudaThreadExit();
                exit(1);
            }
        }
        if( !cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
        {
            // load the reference solution
            unsigned int len_reference = 0;
            int* reference = NULL;
            if (r_gold_fname == 0) {
                fprintf(stderr, "Cannot read the file containing the reference result of the wavelet decomposition.\n");
                cudaThreadExit();
                exit(1);
            }
            if (cutReadFilei( r_gold_fname, &reference, &len_reference) == CUTTrue) 
                printf("Reading reference result from %s\n", r_gold_fname);
            else {
                cudaThreadExit();
                exit(1);
            }
            cutilCondition( 8388608 == len_reference);

            int nmismatches = 0;
            for (int a = 0; a < 8388608; a++) {
                if (odata[a] != reference[a]) {
                    if (nmismatches < 10) {
                        printf("mismatch at index %d; reference = %d, "
                            "computed = %d\n", a, reference[a], odata[a]);
                    }
                    nmismatches++;
                }
            }
            printf("*** Total mismatches: %d\n", nmismatches);
            if (nmismatches > 0) {
                int off = 8388608 / 2;
                for (int a = 0; a < 10 && a + off < 8388608; a++) {
                    printf("at index %d . reference = %d, computed = %d\n",
                        a + off, reference[a + off], odata[a + off]);
                }
            }

            cutFree(reference);
        }


        cutFree(signal);
        free(odata);
        cutFree(s_fname); 
        cutFree(r_fname);  
        cutFree(r_gold_fname);   

    // end "if check spec"
    }

    // free allocated host and device memory
        cutilSafeCall(cudaFree(d_input));
        cutilSafeCall(cudaFree(d_output));
        cutilSafeCall(cudaFree(d_tmp_1));
    
    cudaThreadExit();
}
