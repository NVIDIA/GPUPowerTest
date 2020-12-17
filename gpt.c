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
    int i = 0;
    int gpu = *((int *) input_gpu);
    pthread_t ptid;
    ptid = pthread_self();
    printf("Launching GPU Kernel, Thread ID %ld GPU %d\n", 
            ptid, gpu);
    int rtn = MPI_Barrier(MPI_COMM_WORLD);
    burn(gpu);
}

int main(int argc, char *argv[])
{
    struct timespec ts_start, ts_end;
    struct sigaction act;
    struct sigevent   ev;

    MPI_Init(NULL, NULL);

    int load_time = 60;
    int gpus[MAX_GPUS] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    int gpu_cnt = 0;
    int opt, n, g, i, ret;

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

    printf("load_time: %d, Loading %d GPUs, Index ", load_time, gpu_cnt);
    for (i = 0; i < gpu_cnt; i++)
        printf("%d ",gpus[i]);
    printf("\n\n");

    pthread_attr_init(&attr);
    ev.sigev_notify = SIGEV_SIGNAL;
    ev.sigev_signo  = SIGUSR1;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    act.sa_sigaction = sig_usr1;
    (void) sigaction(SIGUSR1, &act, NULL);

    for (i=0; i < gpu_cnt; i++) {
        ret = pthread_create(&tids[i], &attr, launch_kernel, (void *) &gpus[i]);
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


