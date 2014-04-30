#ifndef TRACE_H
#define TRACE_H

// has MagmaMaxGPUs, strlcpy, max
#include "common_magma.h"

// ----------------------------------------
const int MAX_CORES       = 1;                 // CPU cores
const int MAX_GPU_QUEUES  = MagmaMaxGPUs * 4;  // #devices * #queues per device
const int MAX_EVENTS      = 20000;
const int MAX_LABEL_LEN   = 16;

// ----------------------------------------
#ifdef TRACING

void trace_init               ( int ncore, int ngpus, int nqueue, magma_queue_t *queues );
void trace_cpu_start          ( int core, const char* tag, const char* label );
void trace_cpu_end            ( int core );
void trace_gpu_start          ( int dev, int queue_num, const char* tag, const char* label );
magma_event_t* trace_gpu_event( int dev, int queue_num, const char* tag, const char* label );
void trace_gpu_end            ( int core, int stream_num );
void trace_finalize           ( const char* filename, const char* cssfile );

extern "C"
size_t magma_strlcpy          ( char *dst, const char *src, size_t siz );

#else

#define trace_init(      x1, x2, x3, x4 ) ((void)(0))
#define trace_cpu_start( x1, x2, x3     ) ((void)(0))
#define trace_cpu_end(   x1             ) ((void)(0))
#define trace_gpu_event( x1, x2, x3, x4 ) (NULL)
#define trace_gpu_start( x1, x2, x3, x4 ) ((void)(0))
#define trace_gpu_end( x1, x2 ) ((void)(0))
#define trace_finalize(  x1, x2         ) ((void)(0))

#endif

#endif        //  #ifndef TRACE_H
