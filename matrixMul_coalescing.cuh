/*
 * 5KK73
 * Eindhoven University of Technology
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

// Ref: 
// 1. Memory coalessing : https://cvw.cac.cornell.edu/gpu/coalesced
// 2. Bank conflict : https://www.youtube.com/watch?v=CZgM3DEBplE

#ifndef _MATRIXMUL_COALESCING_H_
#define _MATRIXMUL_COALESCING_H_

#include <stdio.h>
#include "matrixMul.h"

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void
matrixMul_coalescing( float* C, float* A, float* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[(wA * ty) + (a + tx)];
        BS(tx, ty) = B[(wB * ty) + (b + tx)];  // !!! GMEM -> SMEM : tx, ty indexing is switched. 
	                                       // So, BS is transposed.
	    
	                                       // 1 transaction size for GMEM is 128B.
	                                       // But, we are reading a row of A (4B*16=64B) with 1 transaction.
	                                       // So, we are losing 64B per transaction.
	                                       // If we can change memory access pattern of A, we can optimize more.
	                                       // (Theoritically, we can reduce the number of GMEM accesses up to 1/2.)

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
	  Csub += AS(ty, k) * BS(tx, k);  // !!! x, y indexing of BS is switched
	                                  // bank conflict!                              
	    
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[(wB * ty) + (c + tx)] = Csub;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
