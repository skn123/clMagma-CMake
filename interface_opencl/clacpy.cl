/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @generated from zlacpy.cl normal z -> c, Fri Jan 10 15:51:19 2014
 */

/*
   Matrix is divided into 64 x n block rows.
   Each block has 64 threads.
   Each thread copies one row, iterating across all columns.
   The bottom block of rows may be partially outside the matrix;
   if so, rows outside the matrix (row >= m) are disabled.

   @author Mark Gates
 */

#define PRECISION_c
#if defined(PRECISION_c) || defined(PRECISION_z)
typedef float2 magmaFloatComplex;
#endif

__kernel void clacpy_kernel(
    int m, int n,
    __global magmaFloatComplex *A, int offset_A, int lda,
    __global magmaFloatComplex *B, int offset_B, int ldb)
{
    int row = get_group_id(0)*64 + get_local_id(0);
    if(row < m){
        A += (offset_A + row);
        B += (offset_B + row);
        __global magmaFloatComplex *Aend = A + lda*n;
        while(A < Aend){
            *B = *A;
            A += lda;
            B += ldb;
        }
    }
}
