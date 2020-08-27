/**
 * @file   reduce_min.cuh
 * @author Jiaqi Gu, Yibo Lin
 * @date   Feb 2019
 */

#ifndef _DREAMPLACE_GLOBAL_SWAP_REDUCE_MIN_CUH
#define _DREAMPLACE_GLOBAL_SWAP_REDUCE_MIN_CUH

#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T, typename V>
__device__ void warpReduce(/*volatile*/ T *sdata, int tid, V comp)
{
    sdata[tid] = sdata[tid+32*!comp(sdata[tid], sdata[tid+32])]; 
    sdata[tid] = sdata[tid+16*!comp(sdata[tid], sdata[tid+16])]; 
    sdata[tid] = sdata[tid+8*!comp(sdata[tid], sdata[tid+8])]; 
    sdata[tid] = sdata[tid+4*!comp(sdata[tid], sdata[tid+4])]; 
    sdata[tid] = sdata[tid+2*!comp(sdata[tid], sdata[tid+2])]; 
    sdata[tid] = sdata[tid+1*!comp(sdata[tid], sdata[tid+1])]; 
}

/**
* 优化：解决了 reduce3 中存在的多余同步操作（每个warp默认自动同步）。
* globalInputData  输入数据，位于全局内存
* globalOutputData 输出数据，位于全局内存
* n length of array 
* ref reference value 
* comp compare function which returns the target element 
*/
template <typename T, typename V, unsigned int BlockSize=256>
__global__ void reduce4(T *globalInputData, T *globalOutputData, int n, T ref, V comp)
{
	__shared__ T sdata[BlockSize];

	// 坐标索引
	int tid = threadIdx.x;
	int index = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	int indexWithOffset = index + blockDim.x;

	if (index >= n) sdata[tid] = ref;
	else if (indexWithOffset >= n) sdata[tid] = globalInputData[index];
	else 
    {
        //printf("tid = %d, index = %d, indexWithOffset = %d, index+blockDim.x*!comp(globalInputData[index], globalInputData[indexWithOffset]) = %d\n", 
        //        tid, index, indexWithOffset, index+blockDim.x*!comp(globalInputData[index], globalInputData[indexWithOffset])
        //        );
        sdata[tid] = (comp(globalInputData[index], globalInputData[indexWithOffset]))? globalInputData[index] : globalInputData[indexWithOffset]; 
    }

	__syncthreads();

	// 在共享内存中对每一个块进行规约计算
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s) 
        {
            sdata[tid] = sdata[tid+s*!comp(sdata[tid], sdata[tid+s])]; 
        }

		__syncthreads();
	}
	//if (tid < 32) 
    //{
    //    warpReduce(sdata, tid, comp);
    //}

	// 把计算结果从共享内存写回全局内存
	if (tid == 0) 
    {
        globalOutputData[blockIdx.x] = sdata[0];
    }
}

template <typename T, typename V, unsigned int BlockSize=256>
void reduce(T* fMatrix_Device, int iMatrixSize, const T& ref, const V& comp)
{
    for (int i = 1, iNum = iMatrixSize; i < iMatrixSize; i = 2 * i * BlockSize)
    {
        int iBlockNum = (iNum + (2 * BlockSize) - 1) / (2 * BlockSize);
        reduce4 <T, V, BlockSize> <<<iBlockNum, BlockSize>>>(fMatrix_Device, fMatrix_Device, iNum, ref, comp);
        iNum = iBlockNum;
    }
}

template <typename T, typename V, unsigned int BlockSize=256>
void reduce(T* fMatrix_Device, int iMatrixSize, const T& ref, const V& comp, cudaStream_t& stream)
{
    for (int i = 1, iNum = iMatrixSize; i < iMatrixSize; i = 2 * i * BlockSize)
    {
        int iBlockNum = (iNum + (2 * BlockSize) - 1) / (2 * BlockSize);
        reduce4 <T, V, BlockSize> <<<iBlockNum, BlockSize, 0, stream>>>(fMatrix_Device, fMatrix_Device, iNum, ref, comp);
        iNum = iBlockNum;
    }
}

/**
* improvement: resolved redundant synchronization in reduce3, i.e., synchronize each warp 
* globalInputData  input data, located in global memory
* globalOutputData output data, located in global memory
* nc number of initial columns before reduction
* n number of columns 
* ref reference value 
* comp compare function which returns the target element 
*/
template <typename T, typename V, unsigned int BlockSize=256>
__global__ void reduce4_2d(T *globalInputData, T *globalOutputData, int nc, int n, T ref, V comp)
{
	__shared__ T sdata[BlockSize];

    // compute indices 
	int tid = threadIdx.x;
    int yOffset = blockIdx.y * nc;
	int index = yOffset + blockIdx.x*(blockDim.x * 2) + threadIdx.x;
    
	int indexWithOffset = index + blockDim.x;

	if (index >= yOffset + n) sdata[tid] = ref;
	else if (indexWithOffset >= yOffset + n) sdata[tid] = globalInputData[index];
	else 
    {
        //printf("tid = %d, index = %d, indexWithOffset = %d, index+blockDim.x*!comp(globalInputData[index], globalInputData[indexWithOffset]) = %d\n", 
        //        tid, index, indexWithOffset, index+blockDim.x*!comp(globalInputData[index], globalInputData[indexWithOffset])
        //        );
        sdata[tid] = (comp(globalInputData[index], globalInputData[indexWithOffset]))? globalInputData[index] : globalInputData[indexWithOffset]; 
    }

	__syncthreads();

    // reduction for data in shared memory 
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s) 
        {
            sdata[tid] = sdata[tid+s*!comp(sdata[tid], sdata[tid+s])]; 
        }

		__syncthreads();
	}
	//if (tid < 32) 
    //{
    //    warpReduce(sdata, tid, comp);
    //}

    // write data back from shared memory to global memory 
	if (tid == 0) 
    {
        globalOutputData[yOffset + blockIdx.x] = sdata[0];
        //printf("globalOutputData[%d] = %g\n", blockIdx.y*nc + blockIdx.x, sdata[0].cost);
    }
}

template <typename T, typename V, unsigned int BlockSize=256>
__host__ __device__ 
void reduce_2d(T* fMatrix_Device, int m, int n, const T& ref, const V& comp)
{
    for (int i = 1, iNum = n; i < n; i = 2 * i * BlockSize)
    {
        int iBlockNum = (iNum + (2 * BlockSize) - 1) / (2 * BlockSize);
        dim3 grid(iBlockNum, m, 1);
        reduce4_2d <T, V, BlockSize> <<<grid, BlockSize>>>(fMatrix_Device, fMatrix_Device, n, iNum, ref, comp);
        iNum = iBlockNum;
    }
}

DREAMPLACE_END_NAMESPACE

#endif
