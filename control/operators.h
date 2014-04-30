/**
 *
 *  @file operators.h
 *
 *  clMAGMA (version 1.1.0) --
 *  Univ. of Tennessee, Knoxville
 *  Univ. of California, Berkeley
 *  Univ. of Colorado, Denver
 *  @date January 2014
 *
 **/
#ifndef MAGMA_OPERATORS_H
#define MAGMA_OPERATORS_H

// todo define these correctly for CUDA
#define __host__
#define __device__
#define __inline__ inline

/*************************************************************
 *              magmaDoubleComplex
 */

__host__ __device__ static __inline__ magmaDoubleComplex
operator-(const magmaDoubleComplex &a)
{
    return MAGMA_Z_MAKE(-a.x, -a.y);
}

__host__ __device__ static __inline__ magmaDoubleComplex
operator+(const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return MAGMA_Z_MAKE(a.x + b.x, a.y + b.y);
}

__host__ __device__ static __inline__ void
operator+=(magmaDoubleComplex &a, const magmaDoubleComplex b)
{
    a.x += b.x;
    a.y += b.y;
}

__host__ __device__ static __inline__ magmaDoubleComplex
operator-(const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return MAGMA_Z_MAKE(a.x - b.x, a.y - b.y);
}

__host__ __device__ static __inline__ void
operator-=(magmaDoubleComplex &a, const magmaDoubleComplex b)
{
    a.x -= b.x;
    a.y -= b.y;
}

__host__ __device__ static __inline__ magmaDoubleComplex
operator*(const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return MAGMA_Z_MAKE(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

__host__ __device__ static __inline__ magmaDoubleComplex
operator*(const magmaDoubleComplex a, const double s)
{
    return MAGMA_Z_MAKE(a.x * s, a.y * s);
}

__host__ __device__ static __inline__ magmaDoubleComplex
operator*(const double s, const magmaDoubleComplex a)
{
    return MAGMA_Z_MAKE(a.x * s, a.y * s);
}

__host__ __device__ static __inline__ void
operator*=(magmaDoubleComplex &a, const magmaDoubleComplex b)
{
    double tmp = a.y * b.x + a.x * b.y;
    a.x = a.x * b.x - a.y * b.y;
    a.y = tmp;
}

__host__ __device__ static __inline__ void
operator*=(magmaDoubleComplex &a, const double s)
{
    a.x *= s;
    a.y *= s;
}

/*************************************************************
 *              magmaFloatComplex
 */

__host__ __device__ static __inline__ magmaFloatComplex
operator-(const magmaFloatComplex &a)
{
    return MAGMA_C_MAKE(-a.x, -a.y);
}

__host__ __device__ static __inline__ magmaFloatComplex
operator+(const magmaFloatComplex a, const magmaFloatComplex b)
{
    return MAGMA_C_MAKE(a.x + b.x, a.y + b.y);
}

__host__ __device__ static __inline__ void
operator+=(magmaFloatComplex &a, const magmaFloatComplex b)
{
    a.x += b.x;
    a.y += b.y;
}

__host__ __device__ static __inline__ magmaFloatComplex
operator-(const magmaFloatComplex a, const magmaFloatComplex b)
{
    return MAGMA_C_MAKE(a.x - b.x, a.y - b.y);
}

__host__ __device__ static __inline__ void
operator-=(magmaFloatComplex &a, const magmaFloatComplex b)
{
    a.x -= b.x;
    a.y -= b.y;
}

__host__ __device__ static __inline__ magmaFloatComplex
operator*(const magmaFloatComplex a, const magmaFloatComplex b)
{
    return MAGMA_C_MAKE(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

__host__ __device__ static __inline__ magmaFloatComplex
operator*(const magmaFloatComplex a, const float s)
{
    return MAGMA_C_MAKE(a.x * s, a.y * s);
}

__host__ __device__ static __inline__ magmaFloatComplex
operator*(const float s, const magmaFloatComplex a)
{
    return MAGMA_C_MAKE(a.x * s, a.y * s);
}

__host__ __device__ static __inline__ void
operator*=(magmaFloatComplex &a, const magmaFloatComplex b)
{
    float tmp = a.y * b.x + a.x * b.y;
    a.x = a.x * b.x - a.y * b.y;
    a.y = tmp;
}

__host__ __device__ static __inline__ void
operator*=(magmaFloatComplex &a, const float s)
{
    a.x *= s;
    a.y *= s;
}

#endif  // MAGMA_OPERATORS_H
