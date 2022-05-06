/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

#define COPY_SIZE (100 * (1024 * 1024))
#define MAX_GPUS 8

struct gpu_args {
	int gpu_src;
	int gpu_dst;
	int buffer_size;
	int iterations;
	int duration;
};

/*
 * A CPU thread, generate NVlink traffic between 2 GPUs
 * on a unique CUDA stream.
 */
void *nvl_load(void *n_args)
{
	struct gpu_args *nvlp = (struct gpu_args *)n_args;
	int *src_buf, *dst_buf;
	int gpu_src = nvlp->gpu_src;
	int gpu_dst = nvlp->gpu_dst;
	int cp_size = nvlp->buffer_size;
	int duration = nvlp->duration;
	int peer_access = 0;
	time_t start_time, cur_time;

	printf("GPUs: %d and %d\n", gpu_src, gpu_dst);
	cudaDeviceCanAccessPeer(&peer_access, gpu_src, gpu_dst);
        if (!peer_access) {
                printf("No device peer access, exiting...\n");
                pthread_exit((void **)0);
        }
        cudaSetDevice(gpu_src);
        cudaMalloc((void **)&src_buf, cp_size);
	cudaMemset(src_buf, 5, cp_size);
        cudaSetDevice(gpu_dst);
        cudaMalloc((void **)&dst_buf, cp_size);

        cudaSetDevice(gpu_src);
        cudaDeviceEnablePeerAccess(gpu_dst, 0);
        cudaSetDevice(gpu_dst);
        cudaDeviceEnablePeerAccess(gpu_src, 0);

        cudaStream_t streamA;
        cudaSetDevice(gpu_src);
        cudaStreamCreateWithFlags(&streamA, cudaStreamNonBlocking);

	start_time = time(NULL);
	do {
		cudaMemcpyAsync(dst_buf, src_buf, cp_size, cudaMemcpyDeviceToDevice, streamA);
		cudaStreamSynchronize(streamA);
		cur_time = time(NULL);
	} while ((difftime(cur_time, start_time)) < duration);

	cudaStreamSynchronize(streamA);
	printf("D2D copies complete\n");

	cudaSetDevice(gpu_src);
	cudaFree(src_buf);
	cudaStreamDestroy(streamA);
	cudaSetDevice(gpu_dst);
	cudaFree(dst_buf);

	pthread_exit((void *)NULL);
}

int main(int argc, char *argv[])
{
	int device_cnt = 0;
	int cp_size = COPY_SIZE;
	int i, ret, j, retval;
	int nthreads = 0;
	int last_gpu, duration;
	struct gpu_args nvl_args;
	pthread_t tids[MAX_GPUS];
	pthread_attr_t attr;
	
	cudaGetDeviceCount(&device_cnt);
	cudaCheckError();
	if (device_cnt == 0) {
		printf("No CUDA Devices Found...\n");
		exit(0);
	}
	printf("Found %d CUDA devices\n\n", device_cnt);
	if (device_cnt < 2) {
		printf("Require at least 2 GPUs to run NVlink load\n");
		printf("%d GPUs found\n",device_cnt);
		exit(0);
	}
	if ((device_cnt % 2) != 0) {
		printf("Odd number of GPUs: %d\n",device_cnt);
		device_cnt--;
		if (device_cnt < 2) {
			printf("Require at least 2 GPUs...\n");
			exit(0);
		}
	}
	last_gpu = device_cnt - 1;
	duration = 60;
	if (argc > 1) {
		if ((strcmp(argv[1], "-d")) == 0) 
			duration = atoi(argv[2]);
		else {
			printf("Usage %s -d <run duration in seconds>\n",argv[0]);
			printf("default duration is 60 seconds\n");
			exit(0);
		}
	}
	printf("Run duration set to %d seconds, %d GPUs\n",duration,device_cnt);
/*
 * Each iteration in the for loop launches 2 threads,
 * using each GPU for both a source and destination for
 * the D2D copies.
 */
	for (i = 0; i < (device_cnt / 2); i++, last_gpu--) {
		nvl_args.gpu_src = i;
		nvl_args.gpu_dst = last_gpu;
		nvl_args.buffer_size = cp_size;
		nvl_args.iterations = 100;
		nvl_args.duration = duration;
		pthread_attr_init(&attr);

		/* printf("Launching with Source: %d, Destination: %d\n",nvl_args.gpu_src,nvl_args.gpu_dst); */

		ret = pthread_create(&tids[nthreads], &attr, nvl_load, &nvl_args); 
		if (ret != 0) {
			printf("pthread_create failed\n");
			exit(0);
		}
		nthreads++;
		sleep(1);
		/*
		 * Flip the source and destination GPU for the D2D
		 */
		nvl_args.gpu_src = last_gpu;
		nvl_args.gpu_dst = i;
		ret = pthread_create(&tids[nthreads], &attr, nvl_load, &nvl_args); 
		if (ret != 0) {
			printf("pthread_create failed\n");
			exit(0);
		}
		nthreads++;
		sleep(1);
	}
	printf("%d Threads Created\n",nthreads);

	for (j=0; j < nthreads; j++)
		retval = pthread_join(tids[j], (void **)NULL);
		if (retval != 0) {
			printf("pthread_join error return: %d\n",retval);
		}

	printf("All threads completed/exited\n");
}
