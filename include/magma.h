/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @author Mark Gates
 */

#ifndef MAGMA_H
#define MAGMA_H

/* ------------------------------------------------------------
 * MAGMA Blas Functions
 * --------------------------------------------------------- */
#include "magmablas.h"
#include "auxiliary.h"

/* ------------------------------------------------------------
 * MAGMA functions
 * --------------------------------------------------------- */
#include "magma_z.h"
#include "magma_c.h"
#include "magma_d.h"
#include "magma_s.h"

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// initialization
magma_err_t
magma_init( void );

magma_err_t
magma_finalize( void );


// ========================================
// memory allocation
magma_err_t
magma_malloc( magma_ptr *ptrPtr, size_t bytes );

magma_err_t
magma_free( magma_ptr ptr );

magma_err_t
magma_malloc_cpu( void **ptrPtr, size_t bytes );

magma_err_t
magma_free_cpu( void *ptr );


// type-safe convenience functions to avoid using (void**) cast and sizeof(...)
// here n is the number of elements (floats, doubles, etc.) not the number of bytes.
static inline magma_err_t magma_imalloc( magmaInt_ptr           *ptrPtr, size_t n ) { return magma_malloc( (magma_ptr*) ptrPtr, n*sizeof(magma_int_t)        ); }
static inline magma_err_t magma_smalloc( magmaFloat_ptr         *ptrPtr, size_t n ) { return magma_malloc( (magma_ptr*) ptrPtr, n*sizeof(float)              ); }
static inline magma_err_t magma_dmalloc( magmaDouble_ptr        *ptrPtr, size_t n ) { return magma_malloc( (magma_ptr*) ptrPtr, n*sizeof(double)             ); }
static inline magma_err_t magma_cmalloc( magmaFloatComplex_ptr  *ptrPtr, size_t n ) { return magma_malloc( (magma_ptr*) ptrPtr, n*sizeof(magmaFloatComplex)  ); }
static inline magma_err_t magma_zmalloc( magmaDoubleComplex_ptr *ptrPtr, size_t n ) { return magma_malloc( (magma_ptr*) ptrPtr, n*sizeof(magmaDoubleComplex) ); }

static inline magma_err_t magma_imalloc_cpu( magma_int_t        **ptrPtr, size_t n ) { return magma_malloc_cpu( (void**) ptrPtr, n*sizeof(magma_int_t)        ); }
static inline magma_err_t magma_smalloc_cpu( float              **ptrPtr, size_t n ) { return magma_malloc_cpu( (void**) ptrPtr, n*sizeof(float)              ); }
static inline magma_err_t magma_dmalloc_cpu( double             **ptrPtr, size_t n ) { return magma_malloc_cpu( (void**) ptrPtr, n*sizeof(double)             ); }
static inline magma_err_t magma_cmalloc_cpu( magmaFloatComplex  **ptrPtr, size_t n ) { return magma_malloc_cpu( (void**) ptrPtr, n*sizeof(magmaFloatComplex)  ); }
static inline magma_err_t magma_zmalloc_cpu( magmaDoubleComplex **ptrPtr, size_t n ) { return magma_malloc_cpu( (void**) ptrPtr, n*sizeof(magmaDoubleComplex) ); }

//static inline magma_err_t magma_imalloc_pinned( magma_int_t        **ptrPtr, size_t n ) { return magma_malloc_pinned( (void**) ptrPtr, n*sizeof(magma_int_t)        ); }
//static inline magma_err_t magma_smalloc_pinned( float              **ptrPtr, size_t n ) { return magma_malloc_pinned( (void**) ptrPtr, n*sizeof(float)              ); }
//static inline magma_err_t magma_dmalloc_pinned( double             **ptrPtr, size_t n ) { return magma_malloc_pinned( (void**) ptrPtr, n*sizeof(double)             ); }
//static inline magma_err_t magma_cmalloc_pinned( magmaFloatComplex  **ptrPtr, size_t n ) { return magma_malloc_pinned( (void**) ptrPtr, n*sizeof(magmaFloatComplex)  ); }
//static inline magma_err_t magma_zmalloc_pinned( magmaDoubleComplex **ptrPtr, size_t n ) { return magma_malloc_pinned( (void**) ptrPtr, n*sizeof(magmaDoubleComplex) ); }


// ========================================
// device & queue support
magma_err_t
magma_get_devices(
    magma_device_t* devices,
    magma_int_t     size,
    magma_int_t*    numPtr );

magma_int_t 
magma_num_gpus( void );

magma_int_t
magma_queue_meminfo(magma_queue_t queue );

magma_err_t
magma_queue_create( magma_device_t device, magma_queue_t* queuePtr );

magma_err_t
magma_queue_destroy( magma_queue_t  queue );

magma_err_t
magma_queue_sync( magma_queue_t queue );


// ========================================
// event support
magma_err_t
magma_setevent( magma_event_t* eventPtr );

magma_err_t
magma_event_create( magma_event_t* eventPtr );

magma_err_t
magma_event_destroy( magma_event_t event );

magma_err_t
magma_event_record( magma_event_t event, magma_queue_t queue );

magma_err_t
magma_event_query( magma_event_t event );

magma_err_t
magma_event_sync( magma_event_t event );


// ========================================
// error handler
void magma_xerbla( const char *name, magma_int_t info );

const char* magma_strerror( magma_err_t error );

#ifdef __cplusplus
}
#endif

#endif // MAGMA_H
