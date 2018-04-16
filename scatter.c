#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>

double *create_rand_nums(int num_elements) {
  double *rand_nums = (double *)malloc(sizeof(double) * num_elements);
  assert(rand_nums != NULL);
  int i;
  for (i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand() / (double)RAND_MAX);
  }
  return rand_nums;
}

double compute_avg(double *array, int num_elements) {
  double sum = 0.f;
  int i;
  for (i = 0; i < num_elements; i++) {
    sum += array[i];
  }
  return sum / num_elements;
}

int main(int argc, char** argv) {
  
  if (argc != 2) {
    fprintf(stderr, "Usage: avg num_elements_per_proc\n");
    exit(1);
  }

  int num_elements_per_proc = atoi(argv[1]);
  srand(time(NULL));

  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  double *rand_nums = NULL;
  if (world_rank == 0) {
    rand_nums = create_rand_nums(num_elements_per_proc * world_size);
  }

  // For each process, create a buffer that will hold a subset of the entire
  // array
  double *sub_rand_nums = (double *)malloc(sizeof(double) * num_elements_per_proc);
  assert(sub_rand_nums != NULL);

  // Scatter the random numbers from the root process to all processes in
  // the MPI world
  MPI_Scatter(rand_nums, num_elements_per_proc, MPI_DOUBLE, sub_rand_nums,
              num_elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Compute the average of your subset
  double sub_avg = compute_avg(sub_rand_nums, num_elements_per_proc);

  // Gather all partial averages down to the root process
  double *sub_avgs = NULL;
  if (world_rank == 0) {
    sub_avgs = (double *)malloc(sizeof(double) * world_size);
    assert(sub_avgs != NULL);
  }
  MPI_Gather(&sub_avg, 1, MPI_DOUBLE, sub_avgs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Now that we have all of the partial averages on the root, compute the
  // total average of all numbers. Since we are assuming each process computed
  // an average across an equal amount of elements, this computation will
  // produce the correct answer.
  if (world_rank == 0) {
    double avg = compute_avg(sub_avgs, world_size);
    printf("Avg of all elements is %f\n", avg);
    // Compute the average across the original data for comparison
    double original_data_avg =
      compute_avg(rand_nums, num_elements_per_proc * world_size);
    printf("Avg computed across original data is %f\n", original_data_avg);
  }

  // Clean up
  if (world_rank == 0) {
    free(rand_nums);
    free(sub_avgs);
  }
  free(sub_rand_nums);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}

