/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from zlarfx.cl normal z -> c, Fri Jan 10 15:51:19 2014

*/

#include "kernels_header.h"

//#define BLOCK_SIZE 768
#define BLOCK_SIZE 256

#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  8

void zsum_reduce( int n, int i, __local magmaFloatComplex* x )
{
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  barrier(CLK_LOCAL_MEM_FENCE); }
}


//magma_ctrmv_tkernel
__kernel void magma_ctrmv_tkernel(__global magmaFloatComplex *T, int T_offset, int ldt, __global magmaFloatComplex *t, int t_offset, 
                                  __global magmaFloatComplex *y, int y_offset)
{
    T += T_offset;
    t += t_offset;
    y += y_offset;

    const int i = get_local_id(0);
    T += get_group_id(0)*ldt;
    
    __local magmaFloatComplex sum[ 128 ];
    
    sum[i] = MAGMA_C_CNJG(T[i])*t[i];
    zsum_reduce(get_local_size(0), i, sum);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (i==0)
        y[get_group_id(0)] = sum[0];
}

//magma_ctrmv_kernel2
__kernel 
void magma_ctrmv_kernel2(__global magmaFloatComplex *T, int T_offset, int ldt, __global magmaFloatComplex *t, int t_offset, 
                         __global magmaFloatComplex *y, int y_offset, __global magmaFloatComplex *tau, int tau_offset)
{
    T += T_offset;
    t += t_offset;
    y += y_offset;
    tau += tau_offset;

    const int i = get_local_id(0);
    T += get_group_id(0);

    __local magmaFloatComplex sum[ 128 ];

    sum[i] = T[i*ldt]*t[i];
    zsum_reduce(get_local_size(0), i, sum);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i==0){
        y[get_group_id(0)] = sum[0];
        if (get_group_id(0)==0)
            y[get_num_groups(0)] = tau[0];
    }
}

