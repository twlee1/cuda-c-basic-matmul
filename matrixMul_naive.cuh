/*
 * 5KK73
 * Eindhoven University of Technology
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */


////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
// Grid Dimension = (24, 16)
// Block Dimension = (16, 16)
// A = 256 x 512
// B = 512 x 384
// C = 256 x 384
////////////////////////////////////////////////////////////////////////////////
__global__ void matrixMul_naive( float* C, float* A, float* B, int wA, int wB){
  // Block index
  int bx = blockIdx.x;		// 0 to 23 = WC(384) / 24 BLKs
  int by = blockIdx.y;		// 0 to 15 = HC(256) / 16 BLKs

  // Thread index
  int tx = threadIdx.x;		// 0 to 15 per BLK
  int ty = threadIdx.y;		// 0 to 15 per BLK

  // Accumulate row i of A and column j of B, indices for C
  int i = by * blockDim.y + ty;		// (0:15) * 16 + (0:15) = 0:256
  int j = bx * blockDim.x + tx;		// (0:15) * 24 + (0:15) = 0:384

  float accu = 0.0;

  //  Each thread accum. (all i-th row elem. of A) .* (all k-th col elem. of B)
  // .* means element-wise multiplication
  for(int k=0; k<wA; k++){
    accu = accu + A[ i * wA + k ] * B[ k * wB + j ];
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  C[ i * wB + j ] = accu;
}

