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

#define MAX_GPUS 8
#define MAX_GPU_INDEX 7

void burn(int gpu);

void sig_usr1()
{
    printf("SIGUSR1 signal, thread exiting...\n");
    pthread_exit((void *)NULL);
}

void *launch_kernel(void *input_gpu)
{
    int gpu = *((int *) input_gpu);
    pthread_t ptid;
    ptid = pthread_self();
    int mpi_rank = -1;
    (void) MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    printf("Launching GPU Kernel, Thread ID %ld GPU %d mpi_rank %d\n", 
            ptid, gpu, mpi_rank);
    int rtn = MPI_Barrier(MPI_COMM_WORLD);
    burn(gpu);
}


int main(int argc, char *argv[])
{
//    MPI_Init(NULL, NULL);

    int provided = -1;
    (void) MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    int initiallzed = -1;
    (void) MPI_Initialized(&initiallzed);
    printf("MPI initilized with provided %d initialzed %d\n", provided, initiallzed);

    struct timespec ts_start, ts_end;
    struct sigaction act;
    struct sigevent   ev;


    int load_time = 60;
    int gpus[MAX_GPUS] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    int gpu_cnt = 0;
    int opt, n, g, i, ret;

    int mpi_rank = -1;
    (void) MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    int mpi_size = -1;
    (void) MPI_Comm_rank(MPI_COMM_WORLD, &mpi_size);
    char* mpi_local_rank = getenv ("OMPI_COMM_WORLD_LOCAL_RANK");
    int gpu = (mpi_local_rank) ? atoi(mpi_local_rank) : 0;
    printf("OMPI_COMM_WORLD_LOCAL_RANK %s GPU %d mpi_rank %d mpi_size %d\n", 
            mpi_local_rank, gpu, mpi_rank, mpi_size);

    pthread_t tids[MAX_GPUS];
    pthread_attr_t attr;

    if (argc > 1) {
        while ((opt = getopt(argc, argv, ":t:i:")) != -1) {
            switch(opt) {
                case 't':
                    load_time = atoi(optarg);
                    break;
                case 'i':
                    n = strlen(optarg);
                    while (n > 0) {
                        if (strncmp(optarg, ",", 1) != 0) {
                            g = atoi(optarg);
                            if (( g < 0) || ( g > MAX_GPU_INDEX)) {
                                printf("Invalid GPU Index Value: %d\n",g);
                                printf("Valid range is 0 - 7\n");
                                printf("Exiting\n");
                                exit(0);
                            } else {
                                gpus[gpu_cnt] = g;
                                gpu_cnt++;
                            }
                        }
                        n--;
                        optarg++;
                    }
                    break;
                default:
                    printf("Usage: %s [ -t load_time_seconds ] [ -i comma-seprated GPU list ]\n",argv[0]);
                    exit(0);
            }
        }
    } else /* no GPU args so use defaut */
        gpu_cnt = MAX_GPUS;

/*
    printf("load_time: %d, Loading %d GPUs, Index ", load_time, gpu_cnt);
    for (i = 0; i < gpu_cnt; i++)
        printf("%d ",gpus[i]);
        printf("%d ",gpu);
    printf("\n\n");
*/

    printf("load_time: %d sec., GPU %d\n", load_time, gpu);

    pthread_attr_init(&attr);
    ev.sigev_notify = SIGEV_SIGNAL;
    ev.sigev_signo  = SIGUSR1;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    act.sa_sigaction = sig_usr1;
    (void) sigaction(SIGUSR1, &act, NULL);

    gpu_cnt = 1;
    for (i=0; i < gpu_cnt; i++) {
        ret = pthread_create(&tids[i], &attr, launch_kernel, (void *) &gpu);
        if (ret != 0) {
            perror("pthread_create");
            exit(0);
        }
    }
    printf("threads created...\n");
    sleep(load_time);
    printf("awake...\n");
    for (i=0; i < gpu_cnt; i++) {
            ret=pthread_kill(tids[i], SIGUSR1);
            if (ret != 0)
                perror("pthread_kill");
    }
    for (i=0; i < gpu_cnt; i++) {
        pthread_join(tids[i], (void **)NULL);
    }

    printf("MAIN Exiting...\n");
    MPI_Finalize();
}


