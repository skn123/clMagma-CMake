/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @precisions normal z -> s d c
 */

/*
   Matrix is divided into 64 x n block rows.
   Each block has 64 threads.
   Each thread copies one row, iterating across all columns.
   The bottom block of rows may be partially outside the matrix;
   if so, rows outside the matrix (row >= m) are disabled.

   @author Mark Gates
 */

#define PRECISION_z
#if defined(PRECISION_c) || defined(PRECISION_z)
typedef double2 magmaDoubleComplex;
#endif

__kernel void zlacpy_kernel(
    int m, int n,
    __global magmaDoubleComplex *A, int offset_A, int lda,
    __global magmaDoubleComplex *B, int offset_B, int ldb)
{
    int row = get_group_id(0)*64 + get_local_id(0);
    if(row < m){
        A += (offset_A + row);
        B += (offset_B + row);
        __global magmaDoubleComplex *Aend = A + lda*n;
        while(A < Aend){
            *B = *A;
            A += lda;
            B += ldb;
        }
    }
}
