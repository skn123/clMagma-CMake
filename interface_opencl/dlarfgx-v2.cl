/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from zlarfgx-v2.cl normal z -> d, Fri Jan 10 15:51:19 2014
*/

#include "kernels_header.h"

//#define BLOCK_SIZE 768
#define BLOCK_SIZE 256
#define PRECISION_d

//==============================================================================

__kernel
void magma_dlarfgx_gpu_kernel( int n, __global double* dx0, int dx0_offset, __global double* dx, int dx_offset,  
                               __global double *dtau, int dtau_offset, __global double *dxnorm, int dxnorm_offset, 
                               __global double *dA, int dA_offset, int it)
{
    dx0 += dx0_offset;
    dx += dx_offset;
    dtau += dtau_offset;
    dxnorm += dxnorm_offset;
    dA += dA_offset;

    const int i = get_local_id(0);
    const int j = i + BLOCK_SIZE *get_group_id(0);
    __local double scale;
    __local double xnorm;    
  
    double dxi;

    if ( j < n-1)
        dxi = dx[j];
  
    if ( i == 0 ) {
        xnorm = *dxnorm;
        if ( xnorm == 0 ) {
            *dtau = MAGMA_D_ZERO;
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
            double alpha = *dx0;
            double alphar =  MAGMA_D_REAL(alpha), alphai = MAGMA_D_IMAG(alpha);

            // no need to compute the norm as it is passed as input
            double beta  = xnorm; // sqrt( alphar*alphar + alphai*alphai + xnorm*xnorm );
            beta  = -copysign( beta, alphar );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            if (j==0){
               *dtau = MAGMA_D_MAKE((beta - alphar)/beta, -alphai/beta);
               //*dx0  = MAGMA_D_MAKE(  1., 0.);
               *dA   = MAGMA_D_MAKE(beta, 0.);
            }            

            alpha = MAGMA_D_MAKE( MAGMA_D_REAL(alpha) - beta, MAGMA_D_IMAG(alpha));
            scale = MAGMA_D_DIV( MAGMA_D_ONE, alpha);
#endif
        }
    }

    // scale x
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( xnorm != 0 && j < n-1)
        dx[j] = MAGMA_D_MUL(dxi, scale);

    if (j<it){
        *( dA-it+j) = *(dx0-it+j);
        *(dx0-it+j) = MAGMA_D_MAKE(0., 0.);
    } 
}
