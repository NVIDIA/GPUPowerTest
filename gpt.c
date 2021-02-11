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
#define SECSINADAY (24 * (60 * 60))

struct gpt_args {
    int gpu;
    double up_secs;
    double down_secs;
};

void burn(int gpu, double upsecs, double downsecs);

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
    burn(gargs->gpu, gargs->up_secs, gargs->down_secs);
}


int main(int argc, char *argv[])
{
    int provided = -1;
    (void) MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    int initiallzed = -1;
    (void) MPI_Initialized(&initiallzed);
    /* printf("MPI initilized with provided %d initialzed %d\n", provided, initiallzed); */


    struct timespec ts_start, ts_end;
    struct sigaction act;
    struct sigevent   ev;


    double up = 0.0, down = 0.0;
    int  load_time = 60.0;
    int gpus[MAX_GPUS] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    int gpu_cnt = 0;
    int opt, n, g, i, ret;
    struct gpt_args gargs;

    int mpi_rank = -1;
    (void) MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    int mpi_size = -1;
    (void) MPI_Comm_rank(MPI_COMM_WORLD, &mpi_size);
    char* mpi_local_rank = getenv ("OMPI_COMM_WORLD_LOCAL_RANK");
    int gpu = (mpi_local_rank) ? atoi(mpi_local_rank) : 0;
    /*
    printf("OMPI_COMM_WORLD_LOCAL_RANK %s GPU %d mpi_rank %d mpi_size %d\n", 
            mpi_local_rank, gpu, mpi_rank, mpi_size);
    */

    pthread_t tids[MAX_GPUS];
    pthread_attr_t attr;

    if (argc > 1) {
        while ((opt = getopt(argc, argv, ":u:d:t:i:")) != -1) {
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
		    if (down <= 0) {
	 	        printf("Invalid down duration value: %f\n",down);
		        exit(0);
		    }
		    break; 
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
                    printf("Usage: %s -u <power load up duration seconds> -d <power load down duration seconds> -t <load test duration seconds>  [ -i comma-seprated GPU list ]\n",argv[0]);
                    exit(0);
            }
        }
    } else /* no GPU args so use defaut */
        gpu_cnt = MAX_GPUS;

    if ((up + down) > load_time) {
	printf("up/down cycle time exceeds total load time\n");
	exit(0);
    }
    gargs.gpu = gpu;
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
    printf("Specified run load time (-t %d) reached\n",load_time);
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


