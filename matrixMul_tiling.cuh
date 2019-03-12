/*
 * 5KK73
 * Eindhoven University of Technology
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_TILING_H_
#define _MATRIXMUL_TILING_H_

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
// BLOCK_SIZE = 16
// Grid Dimension = (24, 16) 
// Block Dimension = (16, 16) 
// A = 256 x 512 
// B = 512 x 384 
// C = 256 x 384 
////////////////////////////////////////////////////////////////////////////////
__global__ void matrixMul_tiling( float* C, float* A, float* B, int wA, int wB){
    // Block index
    int bx = blockIdx.x;          // 0 to 23 = WC(384) / 24 BLKs
    int by = blockIdx.y;          // 0 to 15 = HC(256) / 16 BLKs
    // Thread index
    int tx = threadIdx.x;         // 0 to 15 per BLK 
    int ty = threadIdx.y;         // 0 to 15 per BLK 

    // Declaration of the shared memory array As used to store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];  // 16 * 16 * 4B = 1024B
    // Declaration of the shared memory array Bs used to store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];  // 1024B
   
    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * (BLOCK_SIZE * by);		// 512 * 16 * (0:15) = (0:8192:122880) 
    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;		
    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;
    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;
    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;
    // Csub is used to store the element of the block sub-matrix that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load the matrices from device memory to shared memory; 
	// each thread loads one element of each matrix
        AS(ty, tx) = A[(wA * ty) + (a + tx)];  // (wA * ty)-th row, (a + tx)-th col
        BS(tx, ty) = B[(wB * tx) + (b + ty)];  // (wB * tx)-th row, (b + ty)-th col

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together; each thread computes one element of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding computation is done 
        // before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;  // Initial index of sub-matrix of C for each block
    C[(wB * ty) + (c + tx)] = Csub;   // (wB * ty)-th row, (c + tx)-th col
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_

// the amount of computation = 2 x M x N x K FLOP = 2 * 384(WC) * 256(HC) * 512(WA) = 100663296 FLOP
// the amount of global memory access = 2 x M x N x K / B word = 2 * 384 * 256 * 512 / 16 = 6291456 * 4B = 25165824 B
// computation : memory op. ratio = 100663296 / 25165824 B = 4
