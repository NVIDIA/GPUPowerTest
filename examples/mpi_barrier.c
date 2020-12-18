#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);
  int rtn = MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}


