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
#include <sys/time.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>
#include <mpi.h>


#define MAX_GPUS 1
#define MAX_GPU_INDEX 15
#define SECSINADAY (24 * (60 * 60))


struct gpt_args {
    int gpu;
    int cores;
    int low;
    int drop;
    double up_secs;
    double down_secs;
};

void burn(int gpu, int cores, int low, int drop, double upsecs, double downsecs);

void sig_usr1()
{
    /* printf("Timer fired, thread exiting...\n"); */
    pthread_exit((void *)NULL);
}

void *launch_kernel(struct gpt_args *gargs)
{
    /* int gpu = *((int *) input_gpu); */
    pthread_t ptid;
    ptid = pthread_self();
    int mpi_rank = -1;
    (void) MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    int rtn = MPI_Barrier(MPI_COMM_WORLD);

    /*
    printf("Launching GPU Kernel, Thread ID %ld GPU %d mpi_rank %d\n", 
            ptid, gargs->gpu, mpi_rank);
    */
    burn(gargs->gpu, gargs->cores, gargs->low, gargs->drop, gargs->up_secs, gargs->down_secs);
}

int main(int argc, char *argv[])
{


    int provided = -1;
    (void) MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided);
    int initiallzed = -1;
    (void) MPI_Initialized(&initiallzed);
    /*
    printf("MPI initilized with provided %d initialzed %d\n", provided, initiallzed); 
    */


    struct timespec ts_start, ts_end;
    struct sigaction act;
    struct sigevent   ev;


    double up = 1.0, down = 1.0;
    int load_time = 60;
    int gpu_cnt = 1;
    int cores = 0;
    int low = 0;
    int drop = 0;
    int opt, n, g, i, ret;
    struct gpt_args gargs;

    int mpi_rank = -1;
    (void) MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    int mpi_size = -1;
    (void) MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    char* mpi_local_rank = getenv ("OMPI_COMM_WORLD_LOCAL_RANK");
    int gpu = (mpi_local_rank) ? atoi(mpi_local_rank) : 0;
    /*
    printf("OMPI_COMM_WORLD_LOCAL_RANK %s GPU %d mpi_rank %d mpi_size %d\n", 
            mpi_local_rank, gpu, mpi_rank, mpi_size);
    */

    pthread_t tids[MAX_GPUS];
    pthread_attr_t attr;

    if (argc == 2 && argv[1][1] == 'h') {
         printf("Usage: %s [-u <power load up duration> (in float seconds) default 1.0]"
             "\n\t[-d <power load down duration> (in float seconds) default 1.0] "
             "\n\t[-t <load test duration> (in int seconds) default 60]"
             "\n\t[-c <spin N CPU cores per GPU> (in int) default 0]"
             "\n\t[-i <GPU number> (in int) default 0]"
             "\n\t[-L <reduce to minimum power on down phase> (boolean) default is medium power]"
             "\n\t[-D <random power dropouts: 0 - 1, 1 - 4 sec, every 6 sec> (boolean) default is not]\n",  argv[0]);
         exit(0);
    }

    if (argc > 1) {
        while ((opt = getopt(argc, argv, ":u:d:t:i:c:L:D")) != -1) {
            switch(opt) {
		case 'u':
		    up = atof(optarg);
		    if (up <= 0.0) {
	 	        printf("Invalid up duration value: %f\n",up);
		        exit(0);
		    }
		    break;
		case 'd':
		    down = atof(optarg);
		    if (down <= 0.0) {
	 	        printf("Invalid down duration value: %f\n",down);
		        exit(0);
		    }
		    break; 
                case 't':
                    load_time = atoi(optarg);
                    break;
                case 'i':
                    if (mpi_local_rank) {
                        printf("GPU number set by OMPI_COMM_WORLD_LOCAL_RANK as %d (-i ignored)", gpu);
                    } else {
                        g = atoi(optarg);
                        if (( g < 0) || ( g > MAX_GPU_INDEX)) {
                            printf("Invalid GPU Index Value: %d\n",g);
                            printf("Valid range is 0 - 15\n");
                            printf("Exiting\n");
                            exit(0);
                        } else {
                            gpu = g;
                        }
                    }
                    break;
		case 'c':
		    cores = atoi(optarg);
		    if (cores < 0) {
	 	        printf("Invalid cores per GPU : %d\n", cores);
		        exit(0);
		    }
		    break;
		case 'L':
		    low = 1;
		    break;
		case 'D':
		    drop = 1;
		    break;
            }
        }
    } else /* no GPU args so use defaut */

    if ((up + down) > load_time) {
	printf("up/down cycle time exceeds total load time\n");
	exit(0);
    }
    gargs.gpu = gpu;
    gargs.cores = cores;
    gargs.low = low;
    gargs.drop = drop;
    gargs.up_secs = up;
    gargs.down_secs = down;
    /* printf("up: %d seconds, down: %d seconds, load_time: %d seconds\n", gargs.up_secs, gargs.down_secs, load_time); */

    pthread_attr_init(&attr);
    ev.sigev_notify = SIGEV_SIGNAL;
    ev.sigev_signo  = SIGUSR1;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    act.sa_sigaction = sig_usr1;
    (void) sigaction(SIGUSR1, &act, NULL);

    gpu_cnt = 1;
    for (i=0; i < gpu_cnt; i++) {
        ret = pthread_create(&tids[i], &attr, launch_kernel, (struct gpt_args *) &gargs);
        if (ret != 0) {
            perror("pthread_create");
            exit(0);
        }
    }
    /* printf("threads created...\n"); */

    int rtn = MPI_Barrier(MPI_COMM_WORLD);
    
    sleep(load_time);
    printf("Specified run load time (-t %d) reached\n", load_time);
    for (i=0; i < gpu_cnt; i++) {
            ret = pthread_kill(tids[i], SIGUSR1);
            if (ret != 0)
                perror("pthread_kill");
    }
    for (i=0; i < gpu_cnt; i++) {
        pthread_join(tids[i], (void **)NULL);
    }

    printf("GPT Exiting...\n");
    MPI_Finalize();
}


