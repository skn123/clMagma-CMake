/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @precisions normal z -> s d c

*/
//#include "common_magma.h"
//#include "commonblas.h"

#define PRECISION_z

#define ZSIZE_1SHARED 32

#if defined(PRECISION_c) || defined(PRECISION_z)
typedef double2 magmaDoubleComplex;
#endif

__kernel void ztranspose3_32(
    __global magmaDoubleComplex *B, int offsetB, int ldb,
    __global magmaDoubleComplex *A, int offsetA, int lda,
    int m, int m32, int n, int n32)
{
     __local magmaDoubleComplex a[32][ZSIZE_1SHARED+1];
    
    int inx = get_local_id(0);
    int iny = get_local_id(1);
    int ibx = get_group_id(0)*32;
    int iby = get_group_id(1)*32;
    
    A += offsetA;
    B += offsetB;
    
    A += ibx + inx + (iby + iny)*lda;
    B += iby + inx + (ibx + iny)*ldb;
    
    int t2 = iby + iny;
    if (ibx + inx < m) {
        if (t2 < n) {
            a[iny+0][inx] = A[0*lda];
            if (t2 + 8 < n) {
                a[iny+8][inx] = A[8*lda];
                if (t2 + 16 < n) {
                    a[iny+16][inx] = A[16*lda];
                    if (t2 + 24 < n)
                        a[iny+24][inx] = A[24*lda];
                }
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

#if defined(PRECISION_s) || defined(PRECISION_d) || defined(PRECISION_c) || defined(PRECISION_z)
    if (iby + inx < n) {
        if (ibx + iny < m) {
            B[0*ldb] = a[inx][iny+0];
            if (ibx + iny + 8 < m) {
                B[8*ldb] = a[inx][iny+8];
                if (ibx + iny + 16 < m) {
                    B[16*ldb] = a[inx][iny+16];
                    if (ibx + iny + 24 < m)
                        B[24*ldb] = a[inx][iny+24];
                }
            }
        }
    }
#else
    if (iby + inx < n) {
        if (ibx + iny < m) {
            B[0*ldb] = a[inx][iny+0];
            if (ibx + iny + 8 < m) {
                B[8*ldb] = a[inx][iny+8];
            }
        }
        if (iby + inx + 16 < n) {
            if (ibx + iny < m) {
                B[0*ldb+16] = a[inx+16][iny+0];
                if (ibx + iny + 8 < m) {
                    B[8*ldb+16] = a[inx+16][iny+8];
                }
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    A += ZSIZE_1SHARED;
    B += 16*ldb;
    
    a[iny+ 0][inx] = A[ 0*lda];
    a[iny+ 8][inx] = A[ 8*lda];
    a[iny+16][inx] = A[16*lda];
    a[iny+24][inx] = A[24*lda];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (iby + inx < n) {
        if (ibx + iny + 16 < m) {
            B[0*ldb] = a[inx][iny+0];
            if (ibx + iny + 24 < m) {
                B[8*ldb] = a[inx][iny+8];
            }
        }
        if (iby + inx + 16 < n) {
            if (ibx + iny + 16 < m) {
                B[0*ldb+16] = a[inx+16][iny+0];
                if (ibx + iny + 24 < m) {
                    B[8*ldb+16] = a[inx+16][iny+8];
                }
            }
        }
    }
#endif

}



__kernel void ztranspose2_32(
    __global magmaDoubleComplex *B, int offsetB, int ldb,
    __global magmaDoubleComplex *A, int offsetA, int lda,
    int m, int m32, int n, int n32)
{
    __local magmaDoubleComplex a[32][ZSIZE_1SHARED+1];
    
    int inx = get_local_id(0);
    int iny = get_local_id(1);
    int ibx = get_group_id(0)*32;
    int iby = get_group_id(1)*32;
    
    int dx, dy;
    if (ibx + 32 < m)
       dx = 0;
    else
       dx = m32;

    if (iby + 32 < n)
       dy = 0;
    else
       dy = n32;

    A += offsetA;
    B += offsetB;

    A += ibx + inx - dx + (iby + iny - dy)*lda;
    B += iby + inx - dy + (ibx + iny - dx)*ldb;
    
    a[iny+ 0][inx] = A[ 0*lda];
    a[iny+ 8][inx] = A[ 8*lda];
    a[iny+16][inx] = A[16*lda];
    a[iny+24][inx] = A[24*lda];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
#if defined(PRECISION_s) || defined(PRECISION_d) || defined(PRECISION_c) || defined(PRECISION_z)
    B[ 0*ldb] = a[inx][iny+ 0];
    B[ 8*ldb] = a[inx][iny+ 8];
    B[16*ldb] = a[inx][iny+16];
    B[24*ldb] = a[inx][iny+24];
#else
    B[0*ldb]    = a[inx   ][iny+0];
    B[8*ldb]    = a[inx   ][iny+8];
    B[0*ldb+16] = a[inx+16][iny+0];
    B[8*ldb+16] = a[inx+16][iny+8];

    barrier(CLK_LOCAL_MEM_FENCE);
    A += ZSIZE_1SHARED;
    B += 16*ldb;

    a[iny+ 0][inx] = A[ 0*lda];
    a[iny+ 8][inx] = A[ 8*lda];
    a[iny+16][inx] = A[16*lda];
    a[iny+24][inx] = A[24*lda];

    barrier(CLK_LOCAL_MEM_FENCE);

    B[0*ldb]    = a[inx   ][iny+0];
    B[8*ldb]    = a[inx   ][iny+8];
    B[0*ldb+16] = a[inx+16][iny+0];
    B[8*ldb+16] = a[inx+16][iny+8];
#endif
}

//
//        m, n - dimensions in the source (input) matrix
//             This version transposes for general m, n .
//             Note that ldi >= m and ldo >= n.
//
/*
extern "C" void
magmablas_stranspose2(float *odata, int ldo,
                      float *idata, int ldi,
                      int m, int n )
{
    // Quick return
    if ( (m == 0) || (n == 0) )
        return;

    dim3 threads( SSIZE_1SHARED, 8, 1 );
    dim3 grid( (m+31)/32, (n+31)/32, 1 );
    stranspose3_32<<< grid, threads, 0, magma_stream >>>( odata, ldo, idata, ldi,
                                         // m, m%32, n, n%32);
                                         m, (32-m%32)%32, n, (32-n%32)%32);
}

extern "C" void
magmablas_stranspose2s(float *odata, int ldo,
                       float *idata, int ldi,
                       int m, int n, cudaStream_t *stream )
{
    // Quick return
    if ( (m == 0) || (n == 0) )
        return;

    dim3 threads( SSIZE_1SHARED, 8, 1 );
    dim3 grid( (m+31)/32, (n+31)/32, 1 );
    stranspose3_32<<< grid, threads, 0, *stream >>>( odata, ldo, idata, ldi,
                                         // m, m%32, n, n%32);
                                         m, (32-m%32)%32, n, (32-n%32)%32);
}
*/
