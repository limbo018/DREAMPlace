// Fast Block Distributed CUDA Implementation of the Hungarian Algorithm
//
// Annex to the paper:
// Paulo A. C. Lopes, Satyendra Singh Yadav, Aleksandar Ilic, Sarat Kumar Patra , 
// "Fast Block Distributed CUDA Implementation of the Hungarian Algorithm",
// Journal Parallel Distributed Computing
//
// Hungarian algorithm:
// (This algorithm was modified to result in an efficient GPU implementation, see paper)
//
// Initialize the slack matrix with the cost matrix, and then work with the slack matrix.
//
// STEP 1: Subtract the row minimum from each row. Subtract the column minimum from each column.
//
// STEP 2: Find a zero of the slack matrix. If there are no starred zeros in its column or row star the zero.
// Repeat for each zero.
//
// STEP 3: Cover each column with a starred zero. If all the columns are
// covered then the matching is maximum.
//
// STEP 4: Find a non-covered zero and prime it. If there is no starred zero in the row containing this primed zero,
// Go to Step 5. Otherwise, cover this row and uncover the column containing the starred zero.
// Continue in this manner until there are no uncovered zeros left.
// Save the smallest uncovered value and Go to Step 6.
//
// STEP 5: Construct a series of alternating primed and starred zeros as follows:
// Let Z0 represent the uncovered primed zero found in Step 4.
// Let Z1 denote the starred zero in the column of Z0(if any).
// Let Z2 denote the primed zero in the row of Z1(there will always be one).
// Continue until the series terminates at a primed zero that has no starred zero in its column.
// Un-star each starred zero of the series, star each primed zero of the series, 
// erase all primes and uncover every row in the matrix. Return to Step 3.
//
// STEP 6: Add the minimum uncovered value to every element of each covered row, 
// and subtract it from every element of each uncovered column.
// Return to Step 4 without altering any stars, primes, or covered rows.

#include <cuda.h>
#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <device_functions.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <random>
#include <assert.h>
#include <chrono>

// Uncomment to use chars as the data type, otherwise use int
// #define CHAR_DATA_TYPE

// Uncomment to use a 4x4 predefined matrix for testing
// #define USE_TEST_MATRIX

// Comment to use managed variables instead of dynamic parallelism; usefull for debugging
 #define DYNAMIC

#define klog2(n) ((n<8)?2:((n<16)?3:((n<32)?4:((n<64)?5:((n<128)?6:((n<256)?7:((n<512)?8:((n<1024)?9:((n<2048)?10:((n<4096)?11:((n<8192)?12:((n<16384)?13:0))))))))))))

#ifndef DYNAMIC
#define MANAGED __managed__ 
#define dh_checkCuda checkCuda
#define dh_get_globaltime get_globaltime
#define dh_get_timer_period get_timer_period
#else
#define dh_checkCuda d_checkCuda
#define dh_get_globaltime d_get_globaltime
#define dh_get_timer_period d_get_timer_period
#define MANAGED
#endif

#define kmin(x,y) ((x<y)?x:y)
#define kmax(x,y) ((x>y)?x:y)

#ifndef USE_TEST_MATRIX
#ifdef _n_
// These values are meant to be changed by scripts
const int n = _n_;							// size of the cost/pay matrix
const int range = _range_;					// defines the range of the random matrix.
const int user_n = n;					
const int n_tests = 100;
#else
// User inputs: These values should be changed by the user
const int user_n = 4096;				// This is the size of the cost matrix as supplied by the user
//const int n = 1<<(klog2(user_n)+1);		// The size of the cost/pay matrix used in the algorithm that is increased to a power of two
const int n = user_n;		// The size of the cost/pay matrix used in the algorithm that is increased to a power of two
const int range = n;					// defines the range of the random matrix.
const int n_tests = 10;					// defines the number of tests performed
#endif

// End of user inputs

const int log2_n = klog2(n);			// log2(n)
const int n_threads = kmin(n,64);		// Number of threads used in small kernels grid size (typically grid size equal to n)
										// Used in steps 3ini, 3, 4ini, 4a, 4b, 5a and 5b (64)
const int n_threads_reduction = kmin(n, 256);		// Number of threads used in the redution kernels in step 1 and 6 (256)
const int n_blocks_reduction = kmin(n, 256);		// Number of blocks used in the redution kernels in step 1 and 6 (256)
const int n_threads_full = kmin(n, 512);			// Number of threads used the largest grids sizes (typically grid size equal to n*n)
										// Used in steps 2 and 6 (512)
const int seed = 45345;					// Initialization for the random number generator

#else
const int n = 4;
const int log2_n = 2;
const int n_threads = 2;
const int n_threads_reduction = 2;
const int n_blocks_reduction = 2;
const int n_threads_full = 2;
#endif

const int n_blocks = n / n_threads;									// Number of blocks used in small kernels grid size (typically grid size equal to n)
const int n_blocks_full = n * n / n_threads_full;					// Number of blocks used the largest gris sizes (typically grid size equal to n*n)
const int row_mask = (1 << log2_n) - 1;								// Used to extract the row from tha matrix position index (matrices are column wise)
const int nrows = n, ncols = n;										// The matrix is square so the number of rows and columns is equal to n
const int max_threads_per_block = 1024;								// The maximum number of threads per block
const int columns_per_block_step_4 = 512;							// Number of columns per block in step 4
const int n_blocks_step_4 = kmax(n / columns_per_block_step_4, 1);	// Number of blocks in step 4 and 2
const int data_block_size = columns_per_block_step_4 * n;			// The size of a data block. Note that this can be bigger than the matrix size.
const int log2_data_block_size = log2_n + klog2(columns_per_block_step_4);	// log2 of the size of a data block. Note that klog2 cannot handle very large sizes

// For the selection of the data type used
#ifndef CHAR_DATA_TYPE
typedef int data;
#define MAX_DATA INT_MAX
#define MIN_DATA INT_MIN
#else
typedef unsigned char data;
#define MAX_DATA 255
#define MIN_DATA 0
#endif

// Host Variables

// Some host variables start with h_ to distinguish them from the corresponding device variables
// Device variables have no prefix.

#ifndef USE_TEST_MATRIX
data h_cost[ncols][nrows];
#else
data h_cost[n][n] = { { 1, 2, 3, 4 }, { 2, 4, 6, 8 }, { 3, 6, 9, 12 }, { 4, 8, 12, 16 } };
#endif
int h_column_of_star_at_row[nrows];
int h_zeros_vector_size;
int h_n_matches;
bool h_found;
bool h_goto_5;

// Device Variables

__device__ data slack[nrows*ncols];						// The slack matrix
__device__ data min_in_rows[nrows];						// Minimum in rows
__device__ data min_in_cols[ncols];						// Minimum in columns
__device__ int zeros[nrows*ncols];						// A vector with the position of the zeros in the slack matrix
__device__ int zeros_size_b[n_blocks_step_4];			// The number of zeros in block i

__device__ int row_of_star_at_column[ncols];			// A vector that given the column j gives the row of the star at that column (or -1, no star)
__device__ int column_of_star_at_row[nrows];			// A vector that given the row i gives the column of the star at that row (or -1, no star)
__device__ int cover_row[nrows];						// A vector that given the row i indicates if it is covered (1- covered, 0- uncovered)
__device__ int cover_column[ncols];						// A vector that given the column j indicates if it is covered (1- covered, 0- uncovered)
__device__ int column_of_prime_at_row[nrows];			// A vector that given the row i gives the column of the prime at that row  (or -1, no prime)
__device__ int row_of_green_at_column[ncols];			// A vector that given the row j gives the column of the green at that row (or -1, no green)

__device__ data max_in_mat_row[nrows];					// Used in step 1 to stores the maximum in rows
__device__ data min_in_mat_col[ncols];					// Used in step 1 to stores the minimums in columns
__device__ data d_min_in_mat_vect[n_blocks_reduction];	// Used in step 6 to stores the intermediate results from the first reduction kernel
__device__ data d_min_in_mat;							// Used in step 6 to store the minimum

MANAGED __device__ int zeros_size;					// The number fo zeros
MANAGED __device__ int n_matches;					// Used in step 3 to count the number of matches found
MANAGED __device__ bool goto_5;						// After step 4, goto step 5?
MANAGED __device__ bool repeat_kernel;				// Needs to repeat the step 2 and step 4 kernel?
#if defined(DEBUG) || defined(_DEBUG)
MANAGED __device__ int n_covered_rows;				// Used in debug mode to check for the number of covered rows
MANAGED __device__ int n_covered_columns;			// Used in debug mode to check for the number of covered columns
#endif

__shared__ extern data sdata[];							// For access to shared memory

// -------------------------------------------------------------------------------------
// Device code
// -------------------------------------------------------------------------------------

#if defined(DEBUG) || defined(_DEBUG)
__global__ void convergence_check() {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (cover_column[i]) atomicAdd((int*)&n_covered_columns, 1);
	if (cover_row[i]) atomicAdd((int*)&n_covered_rows, 1);
}

#endif

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline __device__ cudaError_t d_checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		printf("CUDA Runtime Error: %s\n",
			cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
};

__global__ void init()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	// initializations
	//for step 2
	if (i < nrows){
		cover_row[i] = 0;
		column_of_star_at_row[i] = -1;
	}
	if (i < ncols){
		cover_column[i] = 0;
		row_of_star_at_column[i] = -1;
	}
}

// STEP 1.
// a) Subtracting the row by the minimum in each row
const int n_rows_per_block = n / n_blocks_reduction;

__device__ void min_in_rows_warp_reduce(volatile data* sdata, int tid) {
	if (n_threads_reduction >= 64 && n_rows_per_block < 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
	if (n_threads_reduction >= 32 && n_rows_per_block < 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
	if (n_threads_reduction >= 16 && n_rows_per_block < 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
	if (n_threads_reduction >= 8 && n_rows_per_block < 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
	if (n_threads_reduction >= 4 && n_rows_per_block < 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
	if (n_threads_reduction >= 2 && n_rows_per_block < 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

__global__ void calc_min_in_rows()
{
	__shared__ data sdata[n_threads_reduction];		// One temporary result for each thread.

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	// One gets the line and column from the blockID and threadID.
	unsigned int l = bid * n_rows_per_block + tid % n_rows_per_block;
	unsigned int c = tid / n_rows_per_block;
	unsigned int i = c * nrows + l;
	const unsigned int gridSize = n_threads_reduction * n_blocks_reduction;
	data thread_min = MAX_DATA;

	while (i < n * n) {
		thread_min = min(thread_min, slack[i]);
		i += gridSize;  // go to the next piece of the matrix...
						// gridSize = 2^k * n, so that each thread always processes the same line or column
	}
	sdata[tid] = thread_min;

	__syncthreads();
	if (n_threads_reduction >= 1024 && n_rows_per_block < 1024) { if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
	if (n_threads_reduction >= 512 && n_rows_per_block < 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (n_threads_reduction >= 256 && n_rows_per_block < 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (n_threads_reduction >= 128 && n_rows_per_block < 128) { if (tid <  64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) min_in_rows_warp_reduce(sdata, tid);
	if (tid < n_rows_per_block) min_in_rows[bid*n_rows_per_block + tid] = sdata[tid];
}

// a) Subtracting the column by the minimum in each column
const int n_cols_per_block = n / n_blocks_reduction;

__device__ void min_in_cols_warp_reduce(volatile data* sdata, int tid) {
	if (n_threads_reduction >= 64 && n_cols_per_block < 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
	if (n_threads_reduction >= 32 && n_cols_per_block < 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
	if (n_threads_reduction >= 16 && n_cols_per_block < 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
	if (n_threads_reduction >= 8 && n_cols_per_block < 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
	if (n_threads_reduction >= 4 && n_cols_per_block < 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
	if (n_threads_reduction >= 2 && n_cols_per_block < 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

__global__ void calc_min_in_cols()
{
	__shared__ data sdata[n_threads_reduction];		// One temporary result for each thread

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	// One gets the line and column from the blockID and threadID.
	unsigned int c = bid * n_cols_per_block + tid % n_cols_per_block;
	unsigned int l = tid / n_cols_per_block;
	const unsigned int gridSize = n_threads_reduction * n_blocks_reduction;
	data thread_min = MAX_DATA;

	while (l < n) {
		unsigned int i = c * nrows + l;
		thread_min = min(thread_min, slack[i]);
		l += gridSize / n;  // go to the next piece of the matrix...
							// gridSize = 2^k * n, so that each thread always processes the same line or column
	}
	sdata[tid] = thread_min;

	__syncthreads();
	if (n_threads_reduction >= 1024 && n_cols_per_block < 1024) { if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
	if (n_threads_reduction >= 512 && n_cols_per_block < 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (n_threads_reduction >= 256 && n_cols_per_block < 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (n_threads_reduction >= 128 && n_cols_per_block < 128) { if (tid <  64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) min_in_cols_warp_reduce(sdata, tid);
	if (tid < n_cols_per_block) min_in_cols[bid*n_cols_per_block + tid] = sdata[tid];
}

__global__ void step_1_row_sub()
{

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int l = i & row_mask;

	slack[i] = slack[i] - min_in_rows[l];  // subtract the minimum in row from that row

}

__global__ void step_1_col_sub()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int c = i >> log2_n;
	slack[i] = slack[i] - min_in_cols[c]; // subtract the minimum in row from that row

	if (i == 0) zeros_size = 0;
	if (i < n_blocks_step_4) zeros_size_b[i] = 0;
}

// Compress matrix
__global__ void compress_matrix(){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (slack[i] == 0) {
		atomicAdd(&zeros_size, 1);
		int b = i >> log2_data_block_size;
		int i0 = i & ~(data_block_size - 1);		// == b << log2_data_block_size
		int j = atomicAdd(zeros_size_b + b, 1);
		zeros[i0 + j] = i;
	}
}

// STEP 2
// Find a zero of slack. If there are no starred zeros in its
// column or row star the zero. Repeat for each zero.

// The zeros are split through blocks of data so we run step 2 with several thread blocks and rerun the kernel if repeat was set to true.
__global__ void step_2()
{
	int i = threadIdx.x;
	int b = blockIdx.x;
	__shared__ bool repeat;
	__shared__ bool s_repeat_kernel;

	if (i == 0) s_repeat_kernel = false;

	do {
		__syncthreads();
		if (i == 0) repeat = false;
		__syncthreads();

		for (int j = i; j < zeros_size_b[b]; j += blockDim.x)
		{
			int z = zeros[(b << log2_data_block_size) + j];
			int l = z & row_mask;
			int c = z >> log2_n;

			if (cover_row[l] == 0 && cover_column[c] == 0) {
				// thread trys to get the line
				if (!atomicExch((int *)&(cover_row[l]), 1)){
					// only one thread gets the line
					if (!atomicExch((int *)&(cover_column[c]), 1)){
						// only one thread gets the column
						row_of_star_at_column[c] = l;
						column_of_star_at_row[l] = c;
					}
					else {
						cover_row[l] = 0;
						repeat = true;
						s_repeat_kernel = true;
					}
				}
			}
		}
		__syncthreads();
	} while (repeat);

	if (s_repeat_kernel) repeat_kernel = true;
}

// STEP 3
// uncover all the rows and columns before going to step 3
__global__ void step_3ini()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
		cover_row[i] = 0;
		cover_column[i] = 0;
		if (i == 0) n_matches = 0;
}

// Cover each column with a starred zero. If all the columns are
// covered then the matching is maximum
__global__ void step_3()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (row_of_star_at_column[i]>=0)
	{
		cover_column[i] = 1;
		atomicAdd((int*)&n_matches, 1);
	}
}

// STEP 4
// Find a noncovered zero and prime it. If there is no starred
// zero in the row containing this primed zero, go to Step 5.
// Otherwise, cover this row and uncover the column containing
// the starred zero. Continue in this manner until there are no
// uncovered zeros left. Save the smallest uncovered value and
// Go to Step 6.

__global__ void step_4_init()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	column_of_prime_at_row[i] = -1;
	row_of_green_at_column[i] = -1;
}

__global__ void step_4() {
	__shared__	bool s_found;
	__shared__	bool s_goto_5;
	__shared__	bool s_repeat_kernel;
	volatile int *v_cover_row = cover_row;
	volatile int *v_cover_column = cover_column;

	int i = threadIdx.x;
	int b = blockIdx.x;
	// int limit; my__syncthreads_init(limit);

	if (i == 0) {
		s_repeat_kernel = false;
		s_goto_5 = false;
	}

	do {
		__syncthreads();
		if (i == 0) s_found = false;
		__syncthreads();

		for (int j = i; j < zeros_size_b[b]; j += blockDim.x)
		{
			int z = zeros[(b << log2_data_block_size) + j];
			int l = z & row_mask;
			int c = z >> log2_n;
			int c1 = column_of_star_at_row[l];

			for (int n = 0; n < 10; n++) {

				if (!v_cover_column[c] && !v_cover_row[l]) {
					s_found = true; s_repeat_kernel = true;
					column_of_prime_at_row[l] = c;

					if (c1 >= 0) {
						v_cover_row[l] = 1;
						__threadfence();
						v_cover_column[c1] = 0;
					}
					else {
						s_goto_5 = true;
					}
				}
			} // for(int n

		} // for(int j
		__syncthreads();
	} while (s_found && !s_goto_5);

	if (i == 0 && s_repeat_kernel) repeat_kernel = true;
	if (i == 0 && s_goto_5) goto_5 = true;
}

/* STEP 5:
Construct a series of alternating primed and starred zeros as
follows:
Let Z0 represent the uncovered primed zero found in Step 4.
Let Z1 denote the starred zero in the column of Z0(if any).
Let Z2 denote the primed zero in the row of Z1(there will always
be one). Continue until the series terminates at a primed zero
that has no starred zero in its column. Unstar each starred
zero of the series, star each primed zero of the series, erase
all primes and uncover every line in the matrix. Return to Step 3.*/

// Eliminates joining paths
__global__ void step_5a()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	int r_Z0, c_Z0;

	c_Z0 = column_of_prime_at_row[i];
	if (c_Z0 >= 0 && column_of_star_at_row[i] < 0) {
		row_of_green_at_column[c_Z0] = i;

		while ((r_Z0 = row_of_star_at_column[c_Z0]) >= 0) {
			c_Z0 = column_of_prime_at_row[r_Z0];
			row_of_green_at_column[c_Z0] = r_Z0;
		}
	}
}

// Applies the alternating paths
__global__ void step_5b()
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	int r_Z0, c_Z0, c_Z2;

	r_Z0 = row_of_green_at_column[j];

	if (r_Z0 >= 0 && row_of_star_at_column[j] < 0) {

		c_Z2 = column_of_star_at_row[r_Z0];

		column_of_star_at_row[r_Z0] = j;
		row_of_star_at_column[j] = r_Z0;

		while (c_Z2 >= 0) {
			r_Z0 = row_of_green_at_column[c_Z2];	// row of Z2
			c_Z0 = c_Z2;							// col of Z2
			c_Z2 = column_of_star_at_row[r_Z0];		// col of Z4

													// star Z2
			column_of_star_at_row[r_Z0] = c_Z0;
			row_of_star_at_column[c_Z0] = r_Z0;
		}
	}
}

// STEP 6
// Add the minimum uncovered value to every element of each covered
// row, and subtract it from every element of each uncovered column.
// Return to Step 4 without altering any stars, primes, or covered lines.

template <unsigned int blockSize>
__device__ void min_warp_reduce(volatile data* sdata, int tid) {
	if (blockSize >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
	if (blockSize >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
	if (blockSize >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
	if (blockSize >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
	if (blockSize >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
	if (blockSize >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

template <unsigned int blockSize>  // blockSize is the size of a block of threads
__device__ void min_reduce1(volatile data *g_idata, volatile data *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = MAX_DATA;

	while (i < n) {
		int i1 = i;
		int i2 = i + blockSize;
		int l1 = i1 & row_mask;
		int c1 = i1 >> log2_n; 
		data g1;
		if (cover_row[l1] == 1 || cover_column[c1] == 1) g1 = MAX_DATA;
		else g1 = g_idata[i1];
		int l2 = i2 & row_mask;
		int c2 = i2 >> log2_n;
		data g2;
		if (cover_row[l2] == 1 || cover_column[c2] == 1) g2 = MAX_DATA;
		else g2 = g_idata[i2];
		sdata[tid] = min(sdata[tid], min(g1, g2));
		i += gridSize;
	}

	__syncthreads();
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) min_warp_reduce<blockSize>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__device__ void min_reduce2(volatile data *g_idata, volatile data *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;

	sdata[tid] = min(g_idata[i], g_idata[i + blockSize]);

	__syncthreads();
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) min_warp_reduce<blockSize>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void step_6_add_sub()
{
	// STEP 6:
	//	/*STEP 6: Add the minimum uncovered value to every element of each covered
	//	row, and subtract it from every element of each uncovered column.
	//	Return to Step 4 without altering any stars, primes, or covered lines. */
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int l = i & row_mask;
	int c = i >> log2_n;
	if (cover_row[l] == 1 && cover_column[c] == 1)
		slack[i] += d_min_in_mat;
	if (cover_row[l] == 0 && cover_column[c] == 0)
		slack[i] -= d_min_in_mat;

	if (i == 0) zeros_size = 0;
	if (i < n_blocks_step_4) zeros_size_b[i] = 0;
}

__global__ void min_reduce_kernel1() {
	min_reduce1<n_threads_reduction>(slack, d_min_in_mat_vect, nrows*ncols);
}

__global__ void min_reduce_kernel2() {
	min_reduce2<n_threads_reduction / 2>(d_min_in_mat_vect, &d_min_in_mat, n_blocks_reduction);
}

__device__ inline long long int d_get_globaltime(void) {
	long long int ret;

	asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(ret));

	return ret;
}

// Returns the period in miliseconds
__device__ inline double d_get_timer_period(void) {
	return 1.0e-6;
}

// -------------------------------------------------------------------------------------
// Host code
// -------------------------------------------------------------------------------------

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		printf("CUDA Runtime Error: %s\n",
			cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
};

typedef std::chrono::high_resolution_clock::rep hr_clock_rep;

inline hr_clock_rep get_globaltime(void) {
	using namespace std::chrono;
	return high_resolution_clock::now().time_since_epoch().count();
}

// Returns the period in miliseconds
inline double get_timer_period(void) {
	using namespace std::chrono;
	return 1000.0 * high_resolution_clock::period::num / high_resolution_clock::period::den;
}

#define declare_kernel(k)						\
	hr_clock_rep k##_time = 0;						\
	int k##_runs = 0

#define call_kernel(k, n_blocks, n_threads)		call_kernel_s(k, n_blocks, n_threads, 0ll)

#define call_kernel_s(k, n_blocks, n_threads, shared)	\
{														\
	timer_start = dh_get_globaltime();					\
	k << < n_blocks, n_threads,  shared>> > ();			\
	dh_checkCuda(cudaDeviceSynchronize());					\
	timer_stop = dh_get_globaltime();					\
	k##_time += timer_stop - timer_start;				\
	k##_runs++;											\
}
// printf("Finished kernel " #k "(%d,%d,%lld)\n", n_blocks, n_threads, shared);			\
// fflush(0);											\

#define kernel_stats(k)						\
	printf(#k "\t %g \t %d\n", dh_get_timer_period() * k##_time, k##_runs)

// Hungarian_Algorithm
#ifndef DYNAMIC
void Hungarian_Algorithm()
#else
__global__ void Hungarian_Algorithm()
#endif 
{
	hr_clock_rep timer_start, timer_stop;
	hr_clock_rep total_time_start, total_time_stop;
#if defined(DEBUG) || defined(_DEBUG)
	int last_n_covered_rows = 0, last_n_matches = 0;
#endif

	declare_kernel(init); 
	declare_kernel(calc_min_in_rows); declare_kernel(step_1_row_sub);
	declare_kernel(calc_min_in_cols); declare_kernel(step_1_col_sub);
	declare_kernel(compress_matrix);
	declare_kernel(step_2); 
	declare_kernel(step_3ini); declare_kernel(step_3);
	declare_kernel(step_4_init); declare_kernel(step_4);
	declare_kernel(min_reduce_kernel1); declare_kernel(min_reduce_kernel2); declare_kernel(step_6_add_sub);
	declare_kernel(step_5a); declare_kernel(step_5b); declare_kernel(step_5c);

	total_time_start = dh_get_globaltime();

	// Initialization
	call_kernel(init, n_blocks, n_threads);

	// Step 1 kernels
	call_kernel(calc_min_in_rows, n_blocks_reduction, n_threads_reduction);
	call_kernel(step_1_row_sub, n_blocks_full, n_threads_full);
	call_kernel(calc_min_in_cols, n_blocks_reduction, n_threads_reduction);
	call_kernel(step_1_col_sub, n_blocks_full, n_threads_full);

	// compress_matrix
	call_kernel(compress_matrix, n_blocks_full, n_threads_full);

	// Step 2 kernels
	do {
		repeat_kernel = false; dh_checkCuda(cudaDeviceSynchronize());
		call_kernel(step_2, n_blocks_step_4, (n_blocks_step_4 > 1 || zeros_size > max_threads_per_block) ? max_threads_per_block : zeros_size);
		// If we have more than one block it means that we have 512 lines per block so 1024 threads should be adequate.
	} while (repeat_kernel);

	while (1) {  // repeat steps 3 to 6

		// Step 3 kernels
		call_kernel(step_3ini, n_blocks, n_threads);
		call_kernel(step_3, n_blocks, n_threads);

		if (n_matches >= ncols) break;			// It's done

		//step 4_kernels
		call_kernel(step_4_init, n_blocks, n_threads);

		while (1) // repeat step 4 and 6
		{
#if defined(DEBUG) || defined(_DEBUG)
			// At each iteraton either the number of matched or covered rows has to increase.
			// If we went to step 5 the number of matches increases.
			// If we went to step 6 the number of covered rows increases.
			n_covered_rows = 0; n_covered_columns = 0;
			dh_checkCuda(cudaDeviceSynchronize());
			convergence_check << < n_blocks, n_threads >> > ();
			dh_checkCuda(cudaDeviceSynchronize());
			assert(n_matches>last_n_matches || n_covered_rows>last_n_covered_rows);
			assert(n_matches == n_covered_columns + n_covered_rows);
			last_n_matches = n_matches;
			last_n_covered_rows = n_covered_rows;
#endif
			do {  // step 4 loop
				goto_5 = false; repeat_kernel = false; 
				dh_checkCuda(cudaDeviceSynchronize());
				
				call_kernel(step_4, n_blocks_step_4, (n_blocks_step_4 > 1 || zeros_size > max_threads_per_block) ? max_threads_per_block : zeros_size);
				// If we have more than one block it means that we have 512 lines per block so 1024 threads should be adequate.

			} while (repeat_kernel && !goto_5);

			if (goto_5) break;

			//step 6_kernel
			call_kernel_s(min_reduce_kernel1, n_blocks_reduction, n_threads_reduction, n_threads_reduction*sizeof(int));
			call_kernel_s(min_reduce_kernel2, 1, n_blocks_reduction / 2, (n_blocks_reduction / 2) * sizeof(int));
			call_kernel(step_6_add_sub, n_blocks_full, n_threads_full);

			//compress_matrix
			call_kernel(compress_matrix, n_blocks_full, n_threads_full);

		} // repeat step 4 and 6

		call_kernel(step_5a, n_blocks, n_threads);
		call_kernel(step_5b, n_blocks, n_threads);

	}  // repeat steps 3 to 6

	total_time_stop = dh_get_globaltime();

	printf("kernel \t time (ms) \t runs\n");

	kernel_stats(init);
	kernel_stats(calc_min_in_rows); kernel_stats(step_1_row_sub);
	kernel_stats(calc_min_in_cols); kernel_stats(step_1_col_sub);
	kernel_stats(compress_matrix);
	kernel_stats(step_2);
	kernel_stats(step_3ini); kernel_stats(step_3);
	kernel_stats(step_4_init); kernel_stats(step_4);
	kernel_stats(min_reduce_kernel1); kernel_stats(min_reduce_kernel2); kernel_stats(step_6_add_sub);
	kernel_stats(step_5a); kernel_stats(step_5b); kernel_stats(step_5c);

	printf("Total time(ms) \t %g\n", dh_get_timer_period() * (total_time_stop - total_time_start));
}

// Used to make sure some constants are properly set
void check(bool val, const char *str){
	if (!val) {
		printf("Check failed: %s!\n", str);
		getchar();
		exit(-1);
	}
}

int main()
{
	// Constant checks:
	check(n == (1 << log2_n), "Incorrect log2_n!");
	check(n_threads*n_blocks == n, "n_threads*n_blocks != n\n");
	// step 1
	check(n_blocks_reduction <= n, "Step 1: Should have several lines per block!");
	check(n % n_blocks_reduction == 0, "Step 1: Number of lines per block should be integer!");
	check((n_blocks_reduction*n_threads_reduction) % n == 0, "Step 1: The grid size must be a multiple of the line size!");
	check(n_threads_reduction*n_blocks_reduction <= n*n, "Step 1: The grid size is bigger than the matrix size!");
	// step 6
	check(n_threads_full*n_blocks_full <= n*n, "Step 6: The grid size is bigger than the matrix size!");
	check(columns_per_block_step_4*n == (1 << log2_data_block_size), "Columns per block of step 4 is not a power of two!");

	printf("Running. See out.txt for output.\n");

	// Open text file
	FILE *file = freopen("out.txt", "w", stdout);
	if (file == NULL)
	{
		perror("Error opening the output file!\n");
		getchar();
		exit(1);
	};

	// Prints the current time
	time_t current_time;
	time(&current_time);
	printf("%s\n", ctime(&current_time));
	fflush(file);

#ifndef USE_TEST_MATRIX
	std::default_random_engine generator(seed);
	std::uniform_int_distribution<int> distribution(0, range-1);
	
	for (int test = 0; test < n_tests; test++) {
		printf("\n\n\n\ntest %d\n", test);
		fflush(file);

		for (int c = 0; c < ncols; c++)
			for (int r = 0; r < nrows; r++) {
				if (c < user_n && r < user_n)
					h_cost[c][r] = distribution(generator);
				else {
					if (c == r) h_cost[c][r] = 0;
					else h_cost[c][r] = MAX_DATA;
				}
			}
#endif

		// Copy vectors from host memory to device memory
		cudaMemcpyToSymbol(slack, h_cost, sizeof(data)*nrows*ncols); // symbol refers to the device memory hence "To" means from Host to Device

		// Invoke kernels

		time_t start_time = clock();

		cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 *1024 * 1024);

#ifndef DYNAMIC
		Hungarian_Algorithm();
#else
		Hungarian_Algorithm << <1, 1 >> > ();
#endif 
		checkCuda(cudaDeviceSynchronize());

		time_t stop_time = clock();
		fflush(file);

		// Copy assignments from Device to Host and calculate the total Cost
		cudaMemcpyFromSymbol(h_column_of_star_at_row, column_of_star_at_row, nrows * sizeof(int));

		int total_cost = 0;
		for (int r = 0; r < nrows; r++) {
			int c = h_column_of_star_at_row[r];
			if (c >= 0) total_cost += h_cost[c][r];
		}

		printf("Total cost is \t %d \n", total_cost);
		printf("Low resolution time is \t %f \n", 1000.0*(double)(stop_time - start_time) / CLOCKS_PER_SEC);

#ifndef USE_TEST_MATRIX
	} // for (int test
#endif

	fclose(file);
}
