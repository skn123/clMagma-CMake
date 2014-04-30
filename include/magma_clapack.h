/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @generated from magma_zlapack.h normal z -> c, Fri Jan 10 15:51:16 2014
 */

#ifndef MAGMA_CLAPACK_H
#define MAGMA_CLAPACK_H

#include "magma_types.h"

#define PRECISION_c

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- LAPACK Externs used in MAGMA
*/

#define blasf77_caxpy      FORTRAN_NAME( caxpy,  CAXPY  )
#define blasf77_ccopy      FORTRAN_NAME( ccopy,  CCOPY  )

/* complex versions use C wrapper to return value; no name mangling. */
#if  defined(PRECISION_z) || defined(PRECISION_c)    
#define blasf77_cdotc      cdotc
#else
#define blasf77_cdotc      FORTRAN_NAME( cdotc,  CDOTC  )
#endif

#define blasf77_cgemm      FORTRAN_NAME( cgemm,  CGEMM  )
#define blasf77_cgemv      FORTRAN_NAME( cgemv,  CGEMV  )
#define blasf77_chemm      FORTRAN_NAME( chemm,  CHEMM  )
#define blasf77_chemv      FORTRAN_NAME( chemv,  CHEMV  )
#define blasf77_cher2k     FORTRAN_NAME( cher2k, CHER2K )
#define blasf77_cherk      FORTRAN_NAME( cherk,  CHERK  )
#define blasf77_cscal      FORTRAN_NAME( cscal,  CSCAL  )
#define blasf77_csscal     FORTRAN_NAME( csscal, CSSCAL ) 
#define blasf77_csymm      FORTRAN_NAME( csymm,  CSYMM  )
#define blasf77_csyr2k     FORTRAN_NAME( csyr2k, CSYR2K )
#define blasf77_csyrk      FORTRAN_NAME( csyrk,  CSYRK  )
#define blasf77_cswap      FORTRAN_NAME( cswap,  CSWAP  )
#define blasf77_ctrmm      FORTRAN_NAME( ctrmm,  CTRMM  )
#define blasf77_ctrmv      FORTRAN_NAME( ctrmv,  CTRMV  )
#define blasf77_ctrsm      FORTRAN_NAME( ctrsm,  CTRSM  )
#define blasf77_ctrsv      FORTRAN_NAME( ctrsv,  CTRSV  )
#define blasf77_cgeru      FORTRAN_NAME( cgeru,  CGERU  )

#define lapackf77_cbdsqr   FORTRAN_NAME( cbdsqr, CBDSQR )
#define lapackf77_cgebak   FORTRAN_NAME( cgebak, CGEBAK )
#define lapackf77_cgebal   FORTRAN_NAME( cgebal, CGEBAL )
#define lapackf77_cgebd2   FORTRAN_NAME( cgebd2, CGEBD2 )
#define lapackf77_cgebrd   FORTRAN_NAME( cgebrd, CGEBRD )
#define lapackf77_cgeev    FORTRAN_NAME( cgeev,  CGEEV  )
#define lapackf77_cgehd2   FORTRAN_NAME( cgehd2, CGEHD2 )
#define lapackf77_cgehrd   FORTRAN_NAME( cgehrd, CGEHRD )
#define lapackf77_cgelqf   FORTRAN_NAME( cgelqf, CGELQF )
#define lapackf77_cgels    FORTRAN_NAME( cgels,  CGELS  )
#define lapackf77_cgeqlf   FORTRAN_NAME( cgeqlf, CGEQLF )
#define lapackf77_cgeqrf   FORTRAN_NAME( cgeqrf, CGEQRF )
#define lapackf77_cgesv    FORTRAN_NAME( cgesv,  CGESV  )
#define lapackf77_cgesvd   FORTRAN_NAME( cgesvd, CGESVD )
#define lapackf77_cgetrf   FORTRAN_NAME( cgetrf, CGETRF )
#define lapackf77_cgetri   FORTRAN_NAME( cgetri, CGETRI )
#define lapackf77_cgetrs   FORTRAN_NAME( cgetrs, CGETRS )
#define lapackf77_cheev    FORTRAN_NAME( cheev,  CHEEV  )
#define lapackf77_cheevd   FORTRAN_NAME( cheevd, CHEEVD )
#define lapackf77_chegs2   FORTRAN_NAME( chegs2, CHEGS2 )
#define lapackf77_chegvd   FORTRAN_NAME( chegvd, CHEGVD )
#define lapackf77_chetd2   FORTRAN_NAME( chetd2, CHETD2 )
#define lapackf77_chetrd   FORTRAN_NAME( chetrd, CHETRD )
#define lapackf77_chseqr   FORTRAN_NAME( chseqr, CHSEQR )
#define lapackf77_clacpy   FORTRAN_NAME( clacpy, CLACPY )
#define lapackf77_clacgv   FORTRAN_NAME( clacgv, CLACGV )
#define lapackf77_clange   FORTRAN_NAME( clange, CLANGE )
#define lapackf77_clanhe   FORTRAN_NAME( clanhe, CLANHE )
#define lapackf77_clanht   FORTRAN_NAME( clanht, CLANHT )
#define lapackf77_clansy   FORTRAN_NAME( clansy, CLANSY )
#define lapackf77_clarfb   FORTRAN_NAME( clarfb, CLARFB )
#define lapackf77_clarfg   FORTRAN_NAME( clarfg, CLARFG )
#define lapackf77_clarft   FORTRAN_NAME( clarft, CLARFT )
#define lapackf77_clarnv   FORTRAN_NAME( clarnv, CLARNV )
#define lapackf77_clartg   FORTRAN_NAME( clartg, CLARTG )
#define lapackf77_clascl   FORTRAN_NAME( clascl, CLASCL )
#define lapackf77_claset   FORTRAN_NAME( claset, CLASET )
#define lapackf77_claswp   FORTRAN_NAME( claswp, CLASWP )
#define lapackf77_clatrd   FORTRAN_NAME( clatrd, CLATRD )
#define lapackf77_clabrd   FORTRAN_NAME( clabrd, CLABRD )
#define lapackf77_clauum   FORTRAN_NAME( clauum, CLAUUM )
#define lapackf77_clavhe   FORTRAN_NAME( clavhe, CLAVHE )
#define lapackf77_cposv    FORTRAN_NAME( cposv,  CPOSV  )
#define lapackf77_cpotrf   FORTRAN_NAME( cpotrf, CPOTRF )
#define lapackf77_cpotrs   FORTRAN_NAME( cpotrs, CPOTRS )
#define lapackf77_cpotri   FORTRAN_NAME( cpotri, CPOTRI )
#define lapackf77_ctrevc   FORTRAN_NAME( ctrevc, CTREVC )
#define lapackf77_sstebz   FORTRAN_NAME( sstebz, SSTEBZ )
#define lapackf77_slamc3   FORTRAN_NAME( slamc3, SLAMC3 )
#define lapackf77_slaed4   FORTRAN_NAME( slaed4, SLAED4 )
#define lapackf77_slamrg   FORTRAN_NAME( slamrg, SLAMRG )
#define lapackf77_ctrtri   FORTRAN_NAME( ctrtri, CTRTRI )
#define lapackf77_csteqr   FORTRAN_NAME( csteqr, CSTEQR )
#define lapackf77_cstedc   FORTRAN_NAME( cstedc, CSTEDC )
#define lapackf77_cstein   FORTRAN_NAME( cstein, CSTEIN )
#define lapackf77_cstemr   FORTRAN_NAME( cstemr, CSTEMR )
#define lapackf77_csymv    FORTRAN_NAME( csymv,  CSYMV  )
#define lapackf77_cung2r   FORTRAN_NAME( cung2r, CUNG2R )
#define lapackf77_cungbr   FORTRAN_NAME( cungbr, CUNGBR )
#define lapackf77_cunghr   FORTRAN_NAME( cunghr, CUNGHR )
#define lapackf77_cunglq   FORTRAN_NAME( cunglq, CUNGLQ )
#define lapackf77_cungql   FORTRAN_NAME( cungql, CUNGQL )
#define lapackf77_cungqr   FORTRAN_NAME( cungqr, CUNGQR )
#define lapackf77_cungtr   FORTRAN_NAME( cungtr, CUNGTR )
#define lapackf77_cunm2r   FORTRAN_NAME( cunm2r, CUNM2R )
#define lapackf77_cunmbr   FORTRAN_NAME( cunmbr, CUNMBR )
#define lapackf77_cunmlq   FORTRAN_NAME( cunmlq, CUNMLQ )
#define lapackf77_cunmql   FORTRAN_NAME( cunmql, CUNMQL )
#define lapackf77_cunmqr   FORTRAN_NAME( cunmqr, CUNMQR )
#define lapackf77_cunmtr   FORTRAN_NAME( cunmtr, CUNMTR )

/* testing functions */
#define lapackf77_cbdt01   FORTRAN_NAME( cbdt01, CBDT01 )
#define lapackf77_cget22   FORTRAN_NAME( cget22, CGET22 )
#define lapackf77_chet21   FORTRAN_NAME( chet21, CHET21 )
#define lapackf77_chst01   FORTRAN_NAME( chst01, CHST01 )
#define lapackf77_cqrt02   FORTRAN_NAME( cqrt02, CQRT02 )
#define lapackf77_cunt01   FORTRAN_NAME( cunt01, CUNT01 )
#define lapackf77_clarfy   FORTRAN_NAME( clarfy, CLARFY )
#define lapackf77_cstt21   FORTRAN_NAME( cstt21, CSTT21 )


#if defined(PRECISION_z) || defined(PRECISION_c)
#define DWORKFORZ        float *rwork,
#define DWORKFORZ_AND_LD float *rwork, magma_int_t *ldrwork,
#define WSPLIT           magmaFloatComplex *w
#else
#define DWORKFORZ 
#define DWORKFORZ_AND_LD
#define WSPLIT           float *wr, float *wi
#endif

  /*
   * BLAS functions (Alphabetical order)
   */
void     blasf77_caxpy(const int *, magmaFloatComplex *, magmaFloatComplex *, 
                       const int *, magmaFloatComplex *, const int *);
void     blasf77_ccopy(const int *, magmaFloatComplex *, const int *,
                       magmaFloatComplex *, const int *);
#if defined(PRECISION_z) || defined(PRECISION_c)
void     blasf77_cdotc(magmaFloatComplex *, int *, magmaFloatComplex *, int *, 
                       magmaFloatComplex *, int *);
#endif
void     blasf77_cgemm(const char *, const char *, const int *, const int *, const int *,
                       magmaFloatComplex *, magmaFloatComplex *, const int *, 
                       magmaFloatComplex *, const int *, magmaFloatComplex *,
                       magmaFloatComplex *, const int *);
void     blasf77_cgemv(const char *, const int  *, const int *, magmaFloatComplex *, 
                       magmaFloatComplex *, const int *, magmaFloatComplex *, const int *, 
                       magmaFloatComplex *, magmaFloatComplex *, const int *);
void     blasf77_cgeru(int *, int *, magmaFloatComplex *, magmaFloatComplex *, int *, 
                       magmaFloatComplex *, int *, magmaFloatComplex *, int *);
void     blasf77_chemm(const char *, const char *, const int *, const int *, 
                       magmaFloatComplex *, magmaFloatComplex *, const int *, 
                       magmaFloatComplex *, const int *, magmaFloatComplex *,
                       magmaFloatComplex *, const int *);
void     blasf77_chemv(const char *, const int  *, magmaFloatComplex *, magmaFloatComplex *,
                       const int *, magmaFloatComplex *, const int *, magmaFloatComplex *,
                       magmaFloatComplex *, const int *);
void    blasf77_cher2k(const char *, const char *, const int *, const int *, 
                       magmaFloatComplex *, magmaFloatComplex *, const int *, 
                       magmaFloatComplex *, const int *, float *, 
                       magmaFloatComplex *, const int *);
void    blasf77_cherk( const char *, const char *, const int *, const int *, float *, 
                       magmaFloatComplex *, const int *, float *, magmaFloatComplex *, 
                       const int *);
void    blasf77_cscal( const int *, magmaFloatComplex *, magmaFloatComplex *, const int *);
#if defined(PRECISION_z) || defined(PRECISION_c)
void    blasf77_csscal( const int *, float *, magmaFloatComplex *, const int *);
#endif
void    blasf77_csymm( const char *, const char *, const int *, const int *, 
                       magmaFloatComplex *, magmaFloatComplex *, const int *, 
                       magmaFloatComplex *, const int *, magmaFloatComplex *,
                       magmaFloatComplex *, const int *);
void    blasf77_csyr2k(const char *, const char *, const int *, const int *, 
                       magmaFloatComplex *, magmaFloatComplex *, const int *, 
                       magmaFloatComplex *, const int *, magmaFloatComplex *, 
                       magmaFloatComplex *, const int *);
void    blasf77_csyrk( const char *, const char *, const int *, const int *, 
                       magmaFloatComplex *, magmaFloatComplex *, const int *, 
                       magmaFloatComplex *, magmaFloatComplex *, const int *);
void    blasf77_cswap( int *, magmaFloatComplex *, int *, magmaFloatComplex *, int *);
void    blasf77_ctrmm( const char *, const char *, const char *, const char *, 
                       const int *, const int *, magmaFloatComplex *,
                       magmaFloatComplex *, const int *, magmaFloatComplex *,const int *);
void    blasf77_ctrmv( const char *, const char *, const char *, const int *, 
                       magmaFloatComplex*,  const int *, magmaFloatComplex *, const int*);
void    blasf77_ctrsm( const char *, const char *, const char *, const char *, 
                       const int *, const int *, magmaFloatComplex *, 
                       magmaFloatComplex *, const int *, magmaFloatComplex *,const int*);
void    blasf77_ctrsv( const char *, const char *, const char *, const int *, 
                       magmaFloatComplex *, const int *, magmaFloatComplex *, const int*);

  /*
   * Lapack functions (Alphabetical order)
   */
void    lapackf77_cbdsqr(const char *uplo, magma_int_t *n, magma_int_t *nvct, 
                         magma_int_t *nru,  magma_int_t *ncc, float *D, float *E, 
                         magmaFloatComplex *VT, magma_int_t *ldvt, 
                         magmaFloatComplex *U, magma_int_t *ldu, 
                         magmaFloatComplex *C, magma_int_t *ldc, 
                         float *work, magma_int_t *info);
void    lapackf77_cgebak(const char *job, const char *side, magma_int_t *n, 
                         magma_int_t *ilo, magma_int_t *ihi, 
                         float *scale, magma_int_t *m,
                         magmaFloatComplex *v, magma_int_t *ldv, magma_int_t *info);
void    lapackf77_cgebal(const char *job, magma_int_t *n, magmaFloatComplex *A, magma_int_t *lda, 
                         magma_int_t *ilo, magma_int_t *ihi, float *scale, magma_int_t *info);
void    lapackf77_cgebd2(magma_int_t *m, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, float *d, float *e,
                         magmaFloatComplex *tauq, magmaFloatComplex *taup,
                         magmaFloatComplex *work, magma_int_t *info);
void    lapackf77_cgebrd(magma_int_t *m, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, float *d, float *e,
                         magmaFloatComplex *tauq, magmaFloatComplex *taup, 
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void     lapackf77_cgeev(const char *jobl, const char *jobr, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, WSPLIT, 
                         magmaFloatComplex *vl, magma_int_t *ldvl, 
                         magmaFloatComplex *vr, magma_int_t *ldvr, 
                         magmaFloatComplex *work, magma_int_t *lwork, 
                         DWORKFORZ magma_int_t *info);
void    lapackf77_cgehd2(magma_int_t *n, magma_int_t *ilo, magma_int_t *ihi, 
                         magmaFloatComplex *a, magma_int_t *lda, magmaFloatComplex *tau, 
                         magmaFloatComplex *work, magma_int_t *info);
void    lapackf77_cgehrd(magma_int_t *n, magma_int_t *ilo, magma_int_t *ihi, 
                         magmaFloatComplex *a, magma_int_t *lda, magmaFloatComplex *tau,
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_cgelqf(magma_int_t *m, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, magmaFloatComplex *tau,
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void     lapackf77_cgels(const char *trans, 
                         magma_int_t *m, magma_int_t *n, magma_int_t *nrhs, 
                         magmaFloatComplex *a, magma_int_t *lda, 
                         magmaFloatComplex *b, magma_int_t *ldb,
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_cgeqlf(magma_int_t *m, magma_int_t *n,
                         magmaFloatComplex *a, magma_int_t *lda, magmaFloatComplex *tau, 
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_cgeqrf(magma_int_t *m, magma_int_t *n,
                         magmaFloatComplex *a, magma_int_t *lda, magmaFloatComplex *tau,
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void     lapackf77_cgesv(magma_int_t *n, magma_int_t *nrhs,
                         magmaFloatComplex *A, magma_int_t *lda,
                         magma_int_t *ipiv,
                         magmaFloatComplex *B, magma_int_t *ldb,
                         magma_int_t *info );
void    lapackf77_cgetrf(magma_int_t *m, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, 
                         magma_int_t *ipiv, magma_int_t *info);
void    lapackf77_cgetri(magma_int_t *n,
                         magmaFloatComplex *a, magma_int_t *lda, magma_int_t *ipiv,
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_cgetrs(const char* trans,
                         magma_int_t *n, magma_int_t *nrhs,
                         magmaFloatComplex *a, magma_int_t *lda, magma_int_t *ipiv,
                         magmaFloatComplex *b, magma_int_t *ldb, magma_int_t *info);
void    lapackf77_cgesvd(const char *jobu, const char *jobvt, 
                         magma_int_t *m, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, 
                         float *s, magmaFloatComplex *u, magma_int_t *ldu, 
                         magmaFloatComplex *vt, magma_int_t *ldvt, 
                         magmaFloatComplex *work, magma_int_t *lwork, 
                         DWORKFORZ magma_int_t *info );
void    lapackf77_cheev(const char *jobz, const char *uplo, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, float *w, 
                         magmaFloatComplex *work, magma_int_t *lwork,
                         DWORKFORZ_AND_LD magma_int_t *info);
void    lapackf77_cheevd(const char *jobz, const char *uplo, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, float *w, 
                         magmaFloatComplex *work, magma_int_t *lwork,
                         DWORKFORZ_AND_LD magma_int_t *iwork, 
                         magma_int_t *liwork, magma_int_t *info);
void    lapackf77_chegs2(int *itype, const char *uplo, int *n, 
                         magmaFloatComplex *a, int *lda, 
                         magmaFloatComplex *b, int *ldb, int *info);
void    lapackf77_chegvd(magma_int_t *itype, const char *jobz, const char *uplo, 
                         magma_int_t *n, magmaFloatComplex *a, magma_int_t *lda,
                         magmaFloatComplex *b, magma_int_t *ldb, float *w,
                         magmaFloatComplex *work, magma_int_t *lwork, 
                         DWORKFORZ_AND_LD magma_int_t *iwork, magma_int_t *liwork,
                         magma_int_t *info);
void    lapackf77_chetd2(const char *uplo, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, 
                         float *d, float *e, magmaFloatComplex *tau, magma_int_t *info);
void    lapackf77_chetrd(const char *uplo, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, 
                         float *d, float *e, magmaFloatComplex *tau, 
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_chseqr(const char *job, const char *compz, magma_int_t *n, 
                         magma_int_t *ilo, magma_int_t *ihi, 
                         magmaFloatComplex *H, magma_int_t *ldh, WSPLIT, 
                         magmaFloatComplex *Z, magma_int_t *ldz, 
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_clacpy(const char *uplo, magma_int_t *m, magma_int_t *n, 
                         const magmaFloatComplex *a, magma_int_t *lda, 
                         magmaFloatComplex *b, magma_int_t *ldb);
void    lapackf77_clacgv(magma_int_t *n, magmaFloatComplex *x, magma_int_t *incx);
float  lapackf77_clange(const char *norm, magma_int_t *m, magma_int_t *n, 
                         const magmaFloatComplex *a, magma_int_t *lda, float *work);
float  lapackf77_clanhe(const char *norm, const char *uplo, magma_int_t *n, 
                         const magmaFloatComplex *a, magma_int_t *lda, float * work);
float  lapackf77_clanht(char* norm, magma_int_t* n, 
                         const float* d, const magmaFloatComplex* e);
float  lapackf77_clansy(const char *norm, const char *uplo, magma_int_t *n, 
                         const magmaFloatComplex *a, magma_int_t *lda, float * work);
void    lapackf77_clarfb(const char *side, const char *trans, const char *direct, 
                         const char *storev, magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                         const magmaFloatComplex *v, magma_int_t *ldv, 
                         const magmaFloatComplex *t, magma_int_t *ldt, 
                         magmaFloatComplex *c, magma_int_t *ldc, 
                         magmaFloatComplex *work, magma_int_t *ldwork);
void    lapackf77_clarfg(magma_int_t *n, magmaFloatComplex *alpha, 
                         magmaFloatComplex *x, magma_int_t *incx, magmaFloatComplex *tau);
void    lapackf77_clarft(const char *direct, const char *storev, magma_int_t *n, magma_int_t *k, 
                         magmaFloatComplex *v, magma_int_t *ldv, const magmaFloatComplex *tau, 
                         magmaFloatComplex *t, magma_int_t *ldt);
void    lapackf77_clarnv(magma_int_t *idist, magma_int_t *iseed, magma_int_t *n, 
                         magmaFloatComplex *x);
void    lapackf77_clartg(magmaFloatComplex *F, magmaFloatComplex *G, float *cs, 
                         magmaFloatComplex *SN, magmaFloatComplex *R);
void    lapackf77_clascl(const char *type, magma_int_t *kl, magma_int_t *ku, 
                         float *cfrom, float *cto, 
                         magma_int_t *m, magma_int_t *n, 
                         magmaFloatComplex *A, magma_int_t *lda, magma_int_t *info);
void    lapackf77_claset(const char *uplo, magma_int_t *m, magma_int_t *n, 
                         magmaFloatComplex *alpha, magmaFloatComplex *beta,
                         magmaFloatComplex *A, magma_int_t *lda);
void    lapackf77_claswp(magma_int_t *n, magmaFloatComplex *a, magma_int_t *lda, 
                         magma_int_t *k1, magma_int_t *k2, magma_int_t *ipiv,
                         magma_int_t *incx);
void    lapackf77_clatrd(const char *uplo, magma_int_t *n, magma_int_t *nb, 
                         magmaFloatComplex *a, magma_int_t *lda, float *e,
                         magmaFloatComplex *tau, magmaFloatComplex *work, magma_int_t *ldwork);
void    lapackf77_clabrd(magma_int_t *m, magma_int_t *n, magma_int_t *nb, 
                         magmaFloatComplex *a, magma_int_t *lda, float *d__, float *e, 
                         magmaFloatComplex *tauq, magmaFloatComplex *taup,
                         magmaFloatComplex *x, magma_int_t *ldx,
                         magmaFloatComplex *y, magma_int_t *ldy);
void    lapackf77_cpotrf(const char *uplo, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, magma_int_t *info);
void    lapackf77_cpotrs(const char *uplo, magma_int_t *n, magma_int_t *nrhs,
                         magmaFloatComplex *a, magma_int_t *lda,
                         magmaFloatComplex *b, magma_int_t *ldb, magma_int_t *info);
void    lapackf77_cpotri(const char *uplo, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, magma_int_t *info);
void    lapackf77_clauum(const char *uplo, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, magma_int_t *info);
void    lapackf77_clavhe(const char *uplo, const char *trans, const char *diag,
                         magma_int_t *n, magma_int_t *nrhs,
                         magmaFloatComplex *A, magma_int_t *lda,
                         magma_int_t *ipiv,
                         magmaFloatComplex *B, magma_int_t *ldb,
                         magma_int_t *info );
void     lapackf77_cposv(const char *uplo,
                         const magma_int_t *n, const magma_int_t *nrhs,
                         magmaFloatComplex *A, const magma_int_t *lda,
                         magmaFloatComplex *B,  const magma_int_t *ldb,
                         magma_int_t *info );
void    lapackf77_ctrevc(const char *side, const char *howmny, magma_int_t *select, magma_int_t *n, 
                         magmaFloatComplex *T,  magma_int_t *ldt,  magmaFloatComplex *VL, magma_int_t *ldvl,
                         magmaFloatComplex *VR, magma_int_t *ldvr, magma_int_t *MM, magma_int_t *M, 
                         magmaFloatComplex *work, DWORKFORZ magma_int_t *info);
void    lapackf77_sstebz(char *range, char *order, magma_int_t *n, float *vl, float *vu,
                         magma_int_t *il, magma_int_t *iu, float *abstol,
                         float *d__, float *e, magma_int_t *m, magma_int_t *nsplit,
                         float *w, magma_int_t *iblock, magma_int_t *isplit, float *work,
                         magma_int_t *iwork, magma_int_t *info);
float  lapackf77_slamc3(float* a, float* b);
void    lapackf77_slamrg(magma_int_t* n1, magma_int_t* n2, float* a, 
                         magma_int_t* dtrd1, magma_int_t* dtrd2, magma_int_t* index);
void    lapackf77_slaed4(magma_int_t* n, magma_int_t* i, float* d, float* z,
                         float* delta, float* rho, float* dlam, magma_int_t* info);
void    lapackf77_csteqr(const char *compz, magma_int_t *n, float *D, float *E, 
                         magmaFloatComplex *Z, magma_int_t *ldz, 
                         float *work, magma_int_t *info);
void    lapackf77_cstedc(const char *compz, magma_int_t *n, float *D, float *E, 
                         magmaFloatComplex *Z, magma_int_t *ldz, 
                         magmaFloatComplex *work, magma_int_t *ldwork, 
                         DWORKFORZ_AND_LD magma_int_t *iwork, magma_int_t *liwork,
                         magma_int_t *info);
void    lapackf77_cstein(magma_int_t *n, float *d__, float *e, 
                         magma_int_t *m, float *w, magma_int_t *iblock, magma_int_t *isplit, 
                         magmaFloatComplex *z__, magma_int_t *ldz, float *work, magma_int_t *iwork, 
                         magma_int_t *ifail, magma_int_t *info);
void    lapackf77_cstemr(char *jobz, char *range, magma_int_t *n, float *d__, float *e, 
                         float *vl, float *vu, magma_int_t *il, magma_int_t *iu, magma_int_t *m,
                         float *w, magmaFloatComplex *z__, magma_int_t *ldz, magma_int_t *nzc, 
                         magma_int_t *isuppz, magma_int_t *tryrac, float *work, magma_int_t *lwork, 
                         magma_int_t *iwork, magma_int_t *liwork, magma_int_t *info);
void    lapackf77_ctrtri(const char *uplo, const char *diag, magma_int_t *n,
                         magmaFloatComplex *a, magma_int_t *lda, magma_int_t *info);
#if defined(PRECISION_z) || defined(PRECISION_c)
void    lapackf77_csymv(const char *uplo, const magma_int_t *N, const magmaFloatComplex *alpha, 
                        const magmaFloatComplex *A, const magma_int_t *lda, 
                        const magmaFloatComplex *X, const magma_int_t *incX,
                        const magmaFloatComplex *beta, 
                        magmaFloatComplex *Y, const magma_int_t *incY);
#endif
void    lapackf77_cung2r(magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                         magmaFloatComplex *a, magma_int_t *lda,
                         const magmaFloatComplex *tau, magmaFloatComplex *work,
                         magma_int_t *info);
void    lapackf77_cungbr(const char *vect, magma_int_t *m, magma_int_t *n, magma_int_t *k,
                         magmaFloatComplex *a, magma_int_t *lda, const magmaFloatComplex *tau,
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_cunghr(magma_int_t *n, magma_int_t *ilo, magma_int_t *ihi, 
                         magmaFloatComplex *a, magma_int_t *lda, const magmaFloatComplex *tau,
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_cunglq(magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                         magmaFloatComplex *a, magma_int_t *lda, const magmaFloatComplex *tau, 
                         magmaFloatComplex *work, magma_int_t *ldwork, magma_int_t *info);
void    lapackf77_cungql(magma_int_t *, magma_int_t *, magma_int_t *,
                         magmaFloatComplex *, magma_int_t *, magmaFloatComplex *, 
                         magmaFloatComplex *, magma_int_t *, magma_int_t *);
void    lapackf77_cungqr(magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                         magmaFloatComplex *a, magma_int_t *lda, const magmaFloatComplex *tau, 
                         magmaFloatComplex *work, magma_int_t *ldwork, magma_int_t *info);
void    lapackf77_cungtr(const char *uplo, magma_int_t *n, 
                         magmaFloatComplex *a, magma_int_t *lda, const magmaFloatComplex *tau, 
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_cunm2r(const char *side, const char *trans, 
                         magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                         const magmaFloatComplex *a, magma_int_t *lda, 
                         const magmaFloatComplex *tau, magmaFloatComplex *c, magma_int_t *ldc,
                         magmaFloatComplex *work, magma_int_t *info);
void    lapackf77_cunmbr(const char *vect, const char *side, const char *trans,
                         magma_int_t *M, magma_int_t *N, magma_int_t *K, 
                         magmaFloatComplex *A, magma_int_t *lda, magmaFloatComplex *Tau,
                         magmaFloatComplex *C, magma_int_t *ldc, 
                         magmaFloatComplex *work, magma_int_t *ldwork, magma_int_t *info);
void    lapackf77_cunmlq(const char *side, const char *trans, 
                         magma_int_t *m, magma_int_t *n, magma_int_t *k,
                         const magmaFloatComplex *a, magma_int_t *lda, 
                         const magmaFloatComplex *tau, magmaFloatComplex *c, magma_int_t *ldc, 
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_cunmql(const char *side, const char *trans, 
                         magma_int_t *m, magma_int_t *n, magma_int_t *k,
                         const magmaFloatComplex *a, magma_int_t *lda, 
                         const magmaFloatComplex *tau, magmaFloatComplex *c, magma_int_t *ldc,
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_cunmqr(const char *side, const char *trans, 
                         magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                         const magmaFloatComplex *a, magma_int_t *lda, 
                         const magmaFloatComplex *tau, magmaFloatComplex *c, magma_int_t *ldc, 
                         magmaFloatComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_cunmtr(const char *side, const char *uplo, const char *trans,
                         magma_int_t *M, magma_int_t *N,
                         magmaFloatComplex *A, magma_int_t *lda, magmaFloatComplex *Tau,
                         magmaFloatComplex *C, magma_int_t *ldc, 
                         magmaFloatComplex *work, magma_int_t *ldwork, magma_int_t *info);


  /*
   * Testing functions
   */

#if defined(PRECISION_z) || defined(PRECISION_c)

void    lapackf77_cbdt01(int *m, int *n, int *kd, magmaFloatComplex *A, int *lda, 
                         magmaFloatComplex *Q, int *ldq, float *D, float *E, 
                         magmaFloatComplex *PT, int *ldpt, magmaFloatComplex *work, 
                         float *rwork, float *resid);
void    lapackf77_cget22(const char *transa, const char *transe, const char *transw, int *n,
                         magmaFloatComplex *a, int *lda, magmaFloatComplex *e, int *lde,
                         magmaFloatComplex *w, magmaFloatComplex *work,
                         float *rwork, float *result);
void    lapackf77_chet21(int *itype, const char *uplo, int *n, int *kband, 
                         magmaFloatComplex *A, int *lda, float *D, float *E, 
                         magmaFloatComplex *U, int *ldu, magmaFloatComplex *V, int *ldv, 
                         magmaFloatComplex *TAU, magmaFloatComplex *work,
                         float *rwork, float *result);
void    lapackf77_chst01(int *n, int *ilo, int *ihi, magmaFloatComplex *A, int *lda, 
                         magmaFloatComplex *H, int *ldh, magmaFloatComplex *Q, int *ldq,
                         magmaFloatComplex *work, int *lwork, float *rwork, float *result);
void    lapackf77_cstt21(int *n, int *kband, float *AD, float *AE, float *SD,
                         float *SE, magmaFloatComplex *U, int *ldu, 
                         magmaFloatComplex *work, float *rwork, float *result);
void    lapackf77_cunt01(const char *rowcol, int *m, int *n, magmaFloatComplex *U, int *ldu,
                         magmaFloatComplex *work, int *lwork, float *rwork, float *resid);

#else

void    lapackf77_cbdt01(int *m, int *n, int *kd, magmaFloatComplex *A, int *lda, 
                         magmaFloatComplex *Q, int *ldq, float *D, float *E, 
                         magmaFloatComplex *PT, int *ldpt, 
                         magmaFloatComplex *work, float *resid);
void    lapackf77_cget22(const char *transa, const char *transe, const char *transw, int *n,
                         magmaFloatComplex *a, int *lda, magmaFloatComplex *e, int *lde,
                         magmaFloatComplex *wr, magmaFloatComplex *wi, 
                         float *work, float *result);
void    lapackf77_chet21(int *itype, const char *uplo, int *n, int *kband, 
                         magmaFloatComplex *A, int *lda, float *D, float *E,
                         magmaFloatComplex *U, int *ldu, magmaFloatComplex *V, int *ldv, 
                         magmaFloatComplex *TAU, magmaFloatComplex *work, float *result);
void    lapackf77_chst01(int *n, int *ilo, int *ihi, magmaFloatComplex *A, int *lda, 
                         magmaFloatComplex *H, int *ldh, magmaFloatComplex *Q, int *ldq, 
                         magmaFloatComplex *work, int *lwork, float *result);
void    lapackf77_cstt21(int *n, int *kband, float *AD, float *AE, float *SD, 
                         float *SE, magmaFloatComplex *U, int *ldu, 
                         magmaFloatComplex *work, float *result);
void    lapackf77_cunt01(const char *rowcol, int *m, int *n, magmaFloatComplex *U, int *ldu,
                         magmaFloatComplex *work, int *lwork, float *resid);
#endif

void    lapackf77_clarfy(const char *uplo, int *N, magmaFloatComplex *V, int *incv, 
                         magmaFloatComplex *tau, magmaFloatComplex *C, int *ldc, 
                         magmaFloatComplex *work);
void    lapackf77_cqrt02(int *m, int *n, int *k, magmaFloatComplex *A, magmaFloatComplex *AF,
                         magmaFloatComplex *Q, magmaFloatComplex *R, int *lda, 
                         magmaFloatComplex *TAU, magmaFloatComplex *work, int *lwork,
                         float *rwork, float *result);

#ifdef __cplusplus
}
#endif

#undef DWORKFORZ 
#undef DWORKFORZ_AND_LD
#undef WSPLIT
#undef PRECISION_c
#endif /* MAGMA ZLAPACK */
