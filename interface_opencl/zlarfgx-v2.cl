/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @precisions normal z -> s d c
*/

#include "kernels_header.h"

//#define BLOCK_SIZE 768
#define BLOCK_SIZE 256
#define PRECISION_z

//==============================================================================

__kernel
void magma_zlarfgx_gpu_kernel( int n, __global magmaDoubleComplex* dx0, int dx0_offset, __global magmaDoubleComplex* dx, int dx_offset,  
                               __global magmaDoubleComplex *dtau, int dtau_offset, __global double *dxnorm, int dxnorm_offset, 
                               __global magmaDoubleComplex *dA, int dA_offset, int it)
{
    dx0 += dx0_offset;
    dx += dx_offset;
    dtau += dtau_offset;
    dxnorm += dxnorm_offset;
    dA += dA_offset;

    const int i = get_local_id(0);
    const int j = i + BLOCK_SIZE *get_group_id(0);
    __local magmaDoubleComplex scale;
    __local double xnorm;    
  
    magmaDoubleComplex dxi;

    if ( j < n-1)
        dxi = dx[j];
  
    if ( i == 0 ) {
        xnorm = *dxnorm;
        if ( xnorm == 0 ) {
            *dtau = MAGMA_Z_ZERO;
        }
        else {

#if (defined(PRECISION_s) || defined(PRECISION_d))
            double alpha = *dx0;

            // no need to compute the norm as it is passed as input
            double beta  = xnorm; // sqrt( alpha*alpha + xnorm*xnorm );
            beta  = -copysign( beta, alpha );
 
            // todo: deal with badly scaled vectors (see lapack's larfg)
            if (j==0){
               *dtau = (beta - alpha) / beta;
               //*dx0  = 1.;
               *dA   = beta;  
            }

            scale = 1. / (alpha - beta);
#else
            magmaDoubleComplex alpha = *dx0;
            double alphar =  MAGMA_Z_REAL(alpha), alphai = MAGMA_Z_IMAG(alpha);

            // no need to compute the norm as it is passed as input
            double beta  = xnorm; // sqrt( alphar*alphar + alphai*alphai + xnorm*xnorm );
            beta  = -copysign( beta, alphar );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            if (j==0){
               *dtau = MAGMA_Z_MAKE((beta - alphar)/beta, -alphai/beta);
               //*dx0  = MAGMA_Z_MAKE(  1., 0.);
               *dA   = MAGMA_Z_MAKE(beta, 0.);
            }            

            alpha = MAGMA_Z_MAKE( MAGMA_Z_REAL(alpha) - beta, MAGMA_Z_IMAG(alpha));
            scale = MAGMA_Z_DIV( MAGMA_Z_ONE, alpha);
#endif
        }
    }

    // scale x
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( xnorm != 0 && j < n-1)
        dx[j] = MAGMA_Z_MUL(dxi, scale);

    if (j<it){
        *( dA-it+j) = *(dx0-it+j);
        *(dx0-it+j) = MAGMA_Z_MAKE(0., 0.);
    } 
}
