/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @precisions normal z -> s d c
 */

#ifndef MAGMA_ZLAPACK_H
#define MAGMA_ZLAPACK_H

#include "magma_types.h"

#define PRECISION_z

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- LAPACK Externs used in MAGMA
*/

#define blasf77_zaxpy      FORTRAN_NAME( zaxpy,  ZAXPY  )
#define blasf77_zcopy      FORTRAN_NAME( zcopy,  ZCOPY  )

/* complex versions use C wrapper to return value; no name mangling. */
#if  defined(PRECISION_z) || defined(PRECISION_c)    
#define blasf77_zdotc      zdotc
#else
#define blasf77_zdotc      FORTRAN_NAME( zdotc,  ZDOTC  )
#endif

#define blasf77_zgemm      FORTRAN_NAME( zgemm,  ZGEMM  )
#define blasf77_zgemv      FORTRAN_NAME( zgemv,  ZGEMV  )
#define blasf77_zhemm      FORTRAN_NAME( zhemm,  ZHEMM  )
#define blasf77_zhemv      FORTRAN_NAME( zhemv,  ZHEMV  )
#define blasf77_zher2k     FORTRAN_NAME( zher2k, ZHER2K )
#define blasf77_zherk      FORTRAN_NAME( zherk,  ZHERK  )
#define blasf77_zscal      FORTRAN_NAME( zscal,  ZSCAL  )
#define blasf77_zdscal     FORTRAN_NAME( zdscal, ZDSCAL ) 
#define blasf77_zsymm      FORTRAN_NAME( zsymm,  ZSYMM  )
#define blasf77_zsyr2k     FORTRAN_NAME( zsyr2k, ZSYR2K )
#define blasf77_zsyrk      FORTRAN_NAME( zsyrk,  ZSYRK  )
#define blasf77_zswap      FORTRAN_NAME( zswap,  ZSWAP  )
#define blasf77_ztrmm      FORTRAN_NAME( ztrmm,  ZTRMM  )
#define blasf77_ztrmv      FORTRAN_NAME( ztrmv,  ZTRMV  )
#define blasf77_ztrsm      FORTRAN_NAME( ztrsm,  ZTRSM  )
#define blasf77_ztrsv      FORTRAN_NAME( ztrsv,  ZTRSV  )
#define blasf77_zgeru      FORTRAN_NAME( zgeru,  ZGERU  )

#define lapackf77_zbdsqr   FORTRAN_NAME( zbdsqr, ZBDSQR )
#define lapackf77_zgebak   FORTRAN_NAME( zgebak, ZGEBAK )
#define lapackf77_zgebal   FORTRAN_NAME( zgebal, ZGEBAL )
#define lapackf77_zgebd2   FORTRAN_NAME( zgebd2, ZGEBD2 )
#define lapackf77_zgebrd   FORTRAN_NAME( zgebrd, ZGEBRD )
#define lapackf77_zgeev    FORTRAN_NAME( zgeev,  ZGEEV  )
#define lapackf77_zgehd2   FORTRAN_NAME( zgehd2, ZGEHD2 )
#define lapackf77_zgehrd   FORTRAN_NAME( zgehrd, ZGEHRD )
#define lapackf77_zgelqf   FORTRAN_NAME( zgelqf, ZGELQF )
#define lapackf77_zgels    FORTRAN_NAME( zgels,  ZGELS  )
#define lapackf77_zgeqlf   FORTRAN_NAME( zgeqlf, ZGEQLF )
#define lapackf77_zgeqrf   FORTRAN_NAME( zgeqrf, ZGEQRF )
#define lapackf77_zgesv    FORTRAN_NAME( zgesv,  ZGESV  )
#define lapackf77_zgesvd   FORTRAN_NAME( zgesvd, ZGESVD )
#define lapackf77_zgetrf   FORTRAN_NAME( zgetrf, ZGETRF )
#define lapackf77_zgetri   FORTRAN_NAME( zgetri, ZGETRI )
#define lapackf77_zgetrs   FORTRAN_NAME( zgetrs, ZGETRS )
#define lapackf77_zheev    FORTRAN_NAME( zheev,  ZHEEV  )
#define lapackf77_zheevd   FORTRAN_NAME( zheevd, ZHEEVD )
#define lapackf77_zhegs2   FORTRAN_NAME( zhegs2, ZHEGS2 )
#define lapackf77_zhegvd   FORTRAN_NAME( zhegvd, ZHEGVD )
#define lapackf77_zhetd2   FORTRAN_NAME( zhetd2, ZHETD2 )
#define lapackf77_zhetrd   FORTRAN_NAME( zhetrd, ZHETRD )
#define lapackf77_zhseqr   FORTRAN_NAME( zhseqr, ZHSEQR )
#define lapackf77_zlacpy   FORTRAN_NAME( zlacpy, ZLACPY )
#define lapackf77_zlacgv   FORTRAN_NAME( zlacgv, ZLACGV )
#define lapackf77_zlange   FORTRAN_NAME( zlange, ZLANGE )
#define lapackf77_zlanhe   FORTRAN_NAME( zlanhe, ZLANHE )
#define lapackf77_zlanht   FORTRAN_NAME( zlanht, ZLANHT )
#define lapackf77_zlansy   FORTRAN_NAME( zlansy, ZLANSY )
#define lapackf77_zlarfb   FORTRAN_NAME( zlarfb, ZLARFB )
#define lapackf77_zlarfg   FORTRAN_NAME( zlarfg, ZLARFG )
#define lapackf77_zlarft   FORTRAN_NAME( zlarft, ZLARFT )
#define lapackf77_zlarnv   FORTRAN_NAME( zlarnv, ZLARNV )
#define lapackf77_zlartg   FORTRAN_NAME( zlartg, ZLARTG )
#define lapackf77_zlascl   FORTRAN_NAME( zlascl, ZLASCL )
#define lapackf77_zlaset   FORTRAN_NAME( zlaset, ZLASET )
#define lapackf77_zlaswp   FORTRAN_NAME( zlaswp, ZLASWP )
#define lapackf77_zlatrd   FORTRAN_NAME( zlatrd, ZLATRD )
#define lapackf77_zlabrd   FORTRAN_NAME( zlabrd, ZLABRD )
#define lapackf77_zlauum   FORTRAN_NAME( zlauum, ZLAUUM )
#define lapackf77_zlavhe   FORTRAN_NAME( zlavhe, ZLAVHE )
#define lapackf77_zposv    FORTRAN_NAME( zposv,  ZPOSV  )
#define lapackf77_zpotrf   FORTRAN_NAME( zpotrf, ZPOTRF )
#define lapackf77_zpotrs   FORTRAN_NAME( zpotrs, ZPOTRS )
#define lapackf77_zpotri   FORTRAN_NAME( zpotri, ZPOTRI )
#define lapackf77_ztrevc   FORTRAN_NAME( ztrevc, ZTREVC )
#define lapackf77_dstebz   FORTRAN_NAME( dstebz, DSTEBZ )
#define lapackf77_dlamc3   FORTRAN_NAME( dlamc3, DLAMC3 )
#define lapackf77_dlaed4   FORTRAN_NAME( dlaed4, DLAED4 )
#define lapackf77_dlamrg   FORTRAN_NAME( dlamrg, DLAMRG )
#define lapackf77_ztrtri   FORTRAN_NAME( ztrtri, ZTRTRI )
#define lapackf77_zsteqr   FORTRAN_NAME( zsteqr, ZSTEQR )
#define lapackf77_zstedc   FORTRAN_NAME( zstedc, ZSTEDC )
#define lapackf77_zstein   FORTRAN_NAME( zstein, ZSTEIN )
#define lapackf77_zstemr   FORTRAN_NAME( zstemr, ZSTEMR )
#define lapackf77_zsymv    FORTRAN_NAME( zsymv,  ZSYMV  )
#define lapackf77_zung2r   FORTRAN_NAME( zung2r, ZUNG2R )
#define lapackf77_zungbr   FORTRAN_NAME( zungbr, ZUNGBR )
#define lapackf77_zunghr   FORTRAN_NAME( zunghr, ZUNGHR )
#define lapackf77_zunglq   FORTRAN_NAME( zunglq, ZUNGLQ )
#define lapackf77_zungql   FORTRAN_NAME( zungql, ZUNGQL )
#define lapackf77_zungqr   FORTRAN_NAME( zungqr, ZUNGQR )
#define lapackf77_zungtr   FORTRAN_NAME( zungtr, ZUNGTR )
#define lapackf77_zunm2r   FORTRAN_NAME( zunm2r, ZUNM2R )
#define lapackf77_zunmbr   FORTRAN_NAME( zunmbr, ZUNMBR )
#define lapackf77_zunmlq   FORTRAN_NAME( zunmlq, ZUNMLQ )
#define lapackf77_zunmql   FORTRAN_NAME( zunmql, ZUNMQL )
#define lapackf77_zunmqr   FORTRAN_NAME( zunmqr, ZUNMQR )
#define lapackf77_zunmtr   FORTRAN_NAME( zunmtr, ZUNMTR )

/* testing functions */
#define lapackf77_zbdt01   FORTRAN_NAME( zbdt01, ZBDT01 )
#define lapackf77_zget22   FORTRAN_NAME( zget22, ZGET22 )
#define lapackf77_zhet21   FORTRAN_NAME( zhet21, ZHET21 )
#define lapackf77_zhst01   FORTRAN_NAME( zhst01, ZHST01 )
#define lapackf77_zqrt02   FORTRAN_NAME( zqrt02, ZQRT02 )
#define lapackf77_zunt01   FORTRAN_NAME( zunt01, ZUNT01 )
#define lapackf77_zlarfy   FORTRAN_NAME( zlarfy, ZLARFY )
#define lapackf77_zstt21   FORTRAN_NAME( zstt21, ZSTT21 )


#if defined(PRECISION_z) || defined(PRECISION_c)
#define DWORKFORZ        double *rwork,
#define DWORKFORZ_AND_LD double *rwork, magma_int_t *ldrwork,
#define WSPLIT           magmaDoubleComplex *w
#else
#define DWORKFORZ 
#define DWORKFORZ_AND_LD
#define WSPLIT           double *wr, double *wi
#endif

  /*
   * BLAS functions (Alphabetical order)
   */
void     blasf77_zaxpy(const int *, magmaDoubleComplex *, magmaDoubleComplex *, 
                       const int *, magmaDoubleComplex *, const int *);
void     blasf77_zcopy(const int *, magmaDoubleComplex *, const int *,
                       magmaDoubleComplex *, const int *);
#if defined(PRECISION_z) || defined(PRECISION_c)
void     blasf77_zdotc(magmaDoubleComplex *, int *, magmaDoubleComplex *, int *, 
                       magmaDoubleComplex *, int *);
#endif
void     blasf77_zgemm(const char *, const char *, const int *, const int *, const int *,
                       magmaDoubleComplex *, magmaDoubleComplex *, const int *, 
                       magmaDoubleComplex *, const int *, magmaDoubleComplex *,
                       magmaDoubleComplex *, const int *);
void     blasf77_zgemv(const char *, const int  *, const int *, magmaDoubleComplex *, 
                       magmaDoubleComplex *, const int *, magmaDoubleComplex *, const int *, 
                       magmaDoubleComplex *, magmaDoubleComplex *, const int *);
void     blasf77_zgeru(int *, int *, magmaDoubleComplex *, magmaDoubleComplex *, int *, 
                       magmaDoubleComplex *, int *, magmaDoubleComplex *, int *);
void     blasf77_zhemm(const char *, const char *, const int *, const int *, 
                       magmaDoubleComplex *, magmaDoubleComplex *, const int *, 
                       magmaDoubleComplex *, const int *, magmaDoubleComplex *,
                       magmaDoubleComplex *, const int *);
void     blasf77_zhemv(const char *, const int  *, magmaDoubleComplex *, magmaDoubleComplex *,
                       const int *, magmaDoubleComplex *, const int *, magmaDoubleComplex *,
                       magmaDoubleComplex *, const int *);
void    blasf77_zher2k(const char *, const char *, const int *, const int *, 
                       magmaDoubleComplex *, magmaDoubleComplex *, const int *, 
                       magmaDoubleComplex *, const int *, double *, 
                       magmaDoubleComplex *, const int *);
void    blasf77_zherk( const char *, const char *, const int *, const int *, double *, 
                       magmaDoubleComplex *, const int *, double *, magmaDoubleComplex *, 
                       const int *);
void    blasf77_zscal( const int *, magmaDoubleComplex *, magmaDoubleComplex *, const int *);
#if defined(PRECISION_z) || defined(PRECISION_c)
void    blasf77_zdscal( const int *, double *, magmaDoubleComplex *, const int *);
#endif
void    blasf77_zsymm( const char *, const char *, const int *, const int *, 
                       magmaDoubleComplex *, magmaDoubleComplex *, const int *, 
                       magmaDoubleComplex *, const int *, magmaDoubleComplex *,
                       magmaDoubleComplex *, const int *);
void    blasf77_zsyr2k(const char *, const char *, const int *, const int *, 
                       magmaDoubleComplex *, magmaDoubleComplex *, const int *, 
                       magmaDoubleComplex *, const int *, magmaDoubleComplex *, 
                       magmaDoubleComplex *, const int *);
void    blasf77_zsyrk( const char *, const char *, const int *, const int *, 
                       magmaDoubleComplex *, magmaDoubleComplex *, const int *, 
                       magmaDoubleComplex *, magmaDoubleComplex *, const int *);
void    blasf77_zswap( int *, magmaDoubleComplex *, int *, magmaDoubleComplex *, int *);
void    blasf77_ztrmm( const char *, const char *, const char *, const char *, 
                       const int *, const int *, magmaDoubleComplex *,
                       magmaDoubleComplex *, const int *, magmaDoubleComplex *,const int *);
void    blasf77_ztrmv( const char *, const char *, const char *, const int *, 
                       magmaDoubleComplex*,  const int *, magmaDoubleComplex *, const int*);
void    blasf77_ztrsm( const char *, const char *, const char *, const char *, 
                       const int *, const int *, magmaDoubleComplex *, 
                       magmaDoubleComplex *, const int *, magmaDoubleComplex *,const int*);
void    blasf77_ztrsv( const char *, const char *, const char *, const int *, 
                       magmaDoubleComplex *, const int *, magmaDoubleComplex *, const int*);

  /*
   * Lapack functions (Alphabetical order)
   */
void    lapackf77_zbdsqr(const char *uplo, magma_int_t *n, magma_int_t *nvct, 
                         magma_int_t *nru,  magma_int_t *ncc, double *D, double *E, 
                         magmaDoubleComplex *VT, magma_int_t *ldvt, 
                         magmaDoubleComplex *U, magma_int_t *ldu, 
                         magmaDoubleComplex *C, magma_int_t *ldc, 
                         double *work, magma_int_t *info);
void    lapackf77_zgebak(const char *job, const char *side, magma_int_t *n, 
                         magma_int_t *ilo, magma_int_t *ihi, 
                         double *scale, magma_int_t *m,
                         magmaDoubleComplex *v, magma_int_t *ldv, magma_int_t *info);
void    lapackf77_zgebal(const char *job, magma_int_t *n, magmaDoubleComplex *A, magma_int_t *lda, 
                         magma_int_t *ilo, magma_int_t *ihi, double *scale, magma_int_t *info);
void    lapackf77_zgebd2(magma_int_t *m, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, double *d, double *e,
                         magmaDoubleComplex *tauq, magmaDoubleComplex *taup,
                         magmaDoubleComplex *work, magma_int_t *info);
void    lapackf77_zgebrd(magma_int_t *m, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, double *d, double *e,
                         magmaDoubleComplex *tauq, magmaDoubleComplex *taup, 
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void     lapackf77_zgeev(const char *jobl, const char *jobr, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, WSPLIT, 
                         magmaDoubleComplex *vl, magma_int_t *ldvl, 
                         magmaDoubleComplex *vr, magma_int_t *ldvr, 
                         magmaDoubleComplex *work, magma_int_t *lwork, 
                         DWORKFORZ magma_int_t *info);
void    lapackf77_zgehd2(magma_int_t *n, magma_int_t *ilo, magma_int_t *ihi, 
                         magmaDoubleComplex *a, magma_int_t *lda, magmaDoubleComplex *tau, 
                         magmaDoubleComplex *work, magma_int_t *info);
void    lapackf77_zgehrd(magma_int_t *n, magma_int_t *ilo, magma_int_t *ihi, 
                         magmaDoubleComplex *a, magma_int_t *lda, magmaDoubleComplex *tau,
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zgelqf(magma_int_t *m, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, magmaDoubleComplex *tau,
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void     lapackf77_zgels(const char *trans, 
                         magma_int_t *m, magma_int_t *n, magma_int_t *nrhs, 
                         magmaDoubleComplex *a, magma_int_t *lda, 
                         magmaDoubleComplex *b, magma_int_t *ldb,
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zgeqlf(magma_int_t *m, magma_int_t *n,
                         magmaDoubleComplex *a, magma_int_t *lda, magmaDoubleComplex *tau, 
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zgeqrf(magma_int_t *m, magma_int_t *n,
                         magmaDoubleComplex *a, magma_int_t *lda, magmaDoubleComplex *tau,
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void     lapackf77_zgesv(magma_int_t *n, magma_int_t *nrhs,
                         magmaDoubleComplex *A, magma_int_t *lda,
                         magma_int_t *ipiv,
                         magmaDoubleComplex *B, magma_int_t *ldb,
                         magma_int_t *info );
void    lapackf77_zgetrf(magma_int_t *m, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, 
                         magma_int_t *ipiv, magma_int_t *info);
void    lapackf77_zgetri(magma_int_t *n,
                         magmaDoubleComplex *a, magma_int_t *lda, magma_int_t *ipiv,
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zgetrs(const char* trans,
                         magma_int_t *n, magma_int_t *nrhs,
                         magmaDoubleComplex *a, magma_int_t *lda, magma_int_t *ipiv,
                         magmaDoubleComplex *b, magma_int_t *ldb, magma_int_t *info);
void    lapackf77_zgesvd(const char *jobu, const char *jobvt, 
                         magma_int_t *m, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, 
                         double *s, magmaDoubleComplex *u, magma_int_t *ldu, 
                         magmaDoubleComplex *vt, magma_int_t *ldvt, 
                         magmaDoubleComplex *work, magma_int_t *lwork, 
                         DWORKFORZ magma_int_t *info );
void    lapackf77_zheev(const char *jobz, const char *uplo, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, double *w, 
                         magmaDoubleComplex *work, magma_int_t *lwork,
                         DWORKFORZ_AND_LD magma_int_t *info);
void    lapackf77_zheevd(const char *jobz, const char *uplo, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, double *w, 
                         magmaDoubleComplex *work, magma_int_t *lwork,
                         DWORKFORZ_AND_LD magma_int_t *iwork, 
                         magma_int_t *liwork, magma_int_t *info);
void    lapackf77_zhegs2(int *itype, const char *uplo, int *n, 
                         magmaDoubleComplex *a, int *lda, 
                         magmaDoubleComplex *b, int *ldb, int *info);
void    lapackf77_zhegvd(magma_int_t *itype, const char *jobz, const char *uplo, 
                         magma_int_t *n, magmaDoubleComplex *a, magma_int_t *lda,
                         magmaDoubleComplex *b, magma_int_t *ldb, double *w,
                         magmaDoubleComplex *work, magma_int_t *lwork, 
                         DWORKFORZ_AND_LD magma_int_t *iwork, magma_int_t *liwork,
                         magma_int_t *info);
void    lapackf77_zhetd2(const char *uplo, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, 
                         double *d, double *e, magmaDoubleComplex *tau, magma_int_t *info);
void    lapackf77_zhetrd(const char *uplo, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, 
                         double *d, double *e, magmaDoubleComplex *tau, 
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zhseqr(const char *job, const char *compz, magma_int_t *n, 
                         magma_int_t *ilo, magma_int_t *ihi, 
                         magmaDoubleComplex *H, magma_int_t *ldh, WSPLIT, 
                         magmaDoubleComplex *Z, magma_int_t *ldz, 
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zlacpy(const char *uplo, magma_int_t *m, magma_int_t *n, 
                         const magmaDoubleComplex *a, magma_int_t *lda, 
                         magmaDoubleComplex *b, magma_int_t *ldb);
void    lapackf77_zlacgv(magma_int_t *n, magmaDoubleComplex *x, magma_int_t *incx);
double  lapackf77_zlange(const char *norm, magma_int_t *m, magma_int_t *n, 
                         const magmaDoubleComplex *a, magma_int_t *lda, double *work);
double  lapackf77_zlanhe(const char *norm, const char *uplo, magma_int_t *n, 
                         const magmaDoubleComplex *a, magma_int_t *lda, double * work);
double  lapackf77_zlanht(char* norm, magma_int_t* n, 
                         const double* d, const magmaDoubleComplex* e);
double  lapackf77_zlansy(const char *norm, const char *uplo, magma_int_t *n, 
                         const magmaDoubleComplex *a, magma_int_t *lda, double * work);
void    lapackf77_zlarfb(const char *side, const char *trans, const char *direct, 
                         const char *storev, magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                         const magmaDoubleComplex *v, magma_int_t *ldv, 
                         const magmaDoubleComplex *t, magma_int_t *ldt, 
                         magmaDoubleComplex *c, magma_int_t *ldc, 
                         magmaDoubleComplex *work, magma_int_t *ldwork);
void    lapackf77_zlarfg(magma_int_t *n, magmaDoubleComplex *alpha, 
                         magmaDoubleComplex *x, magma_int_t *incx, magmaDoubleComplex *tau);
void    lapackf77_zlarft(const char *direct, const char *storev, magma_int_t *n, magma_int_t *k, 
                         magmaDoubleComplex *v, magma_int_t *ldv, const magmaDoubleComplex *tau, 
                         magmaDoubleComplex *t, magma_int_t *ldt);
void    lapackf77_zlarnv(magma_int_t *idist, magma_int_t *iseed, magma_int_t *n, 
                         magmaDoubleComplex *x);
void    lapackf77_zlartg(magmaDoubleComplex *F, magmaDoubleComplex *G, double *cs, 
                         magmaDoubleComplex *SN, magmaDoubleComplex *R);
void    lapackf77_zlascl(const char *type, magma_int_t *kl, magma_int_t *ku, 
                         double *cfrom, double *cto, 
                         magma_int_t *m, magma_int_t *n, 
                         magmaDoubleComplex *A, magma_int_t *lda, magma_int_t *info);
void    lapackf77_zlaset(const char *uplo, magma_int_t *m, magma_int_t *n, 
                         magmaDoubleComplex *alpha, magmaDoubleComplex *beta,
                         magmaDoubleComplex *A, magma_int_t *lda);
void    lapackf77_zlaswp(magma_int_t *n, magmaDoubleComplex *a, magma_int_t *lda, 
                         magma_int_t *k1, magma_int_t *k2, magma_int_t *ipiv,
                         magma_int_t *incx);
void    lapackf77_zlatrd(const char *uplo, magma_int_t *n, magma_int_t *nb, 
                         magmaDoubleComplex *a, magma_int_t *lda, double *e,
                         magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t *ldwork);
void    lapackf77_zlabrd(magma_int_t *m, magma_int_t *n, magma_int_t *nb, 
                         magmaDoubleComplex *a, magma_int_t *lda, double *d__, double *e, 
                         magmaDoubleComplex *tauq, magmaDoubleComplex *taup,
                         magmaDoubleComplex *x, magma_int_t *ldx,
                         magmaDoubleComplex *y, magma_int_t *ldy);
void    lapackf77_zpotrf(const char *uplo, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, magma_int_t *info);
void    lapackf77_zpotrs(const char *uplo, magma_int_t *n, magma_int_t *nrhs,
                         magmaDoubleComplex *a, magma_int_t *lda,
                         magmaDoubleComplex *b, magma_int_t *ldb, magma_int_t *info);
void    lapackf77_zpotri(const char *uplo, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, magma_int_t *info);
void    lapackf77_zlauum(const char *uplo, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, magma_int_t *info);
void    lapackf77_zlavhe(const char *uplo, const char *trans, const char *diag,
                         magma_int_t *n, magma_int_t *nrhs,
                         magmaDoubleComplex *A, magma_int_t *lda,
                         magma_int_t *ipiv,
                         magmaDoubleComplex *B, magma_int_t *ldb,
                         magma_int_t *info );
void     lapackf77_zposv(const char *uplo,
                         const magma_int_t *n, const magma_int_t *nrhs,
                         magmaDoubleComplex *A, const magma_int_t *lda,
                         magmaDoubleComplex *B,  const magma_int_t *ldb,
                         magma_int_t *info );
void    lapackf77_ztrevc(const char *side, const char *howmny, magma_int_t *select, magma_int_t *n, 
                         magmaDoubleComplex *T,  magma_int_t *ldt,  magmaDoubleComplex *VL, magma_int_t *ldvl,
                         magmaDoubleComplex *VR, magma_int_t *ldvr, magma_int_t *MM, magma_int_t *M, 
                         magmaDoubleComplex *work, DWORKFORZ magma_int_t *info);
void    lapackf77_dstebz(char *range, char *order, magma_int_t *n, double *vl, double *vu,
                         magma_int_t *il, magma_int_t *iu, double *abstol,
                         double *d__, double *e, magma_int_t *m, magma_int_t *nsplit,
                         double *w, magma_int_t *iblock, magma_int_t *isplit, double *work,
                         magma_int_t *iwork, magma_int_t *info);
double  lapackf77_dlamc3(double* a, double* b);
void    lapackf77_dlamrg(magma_int_t* n1, magma_int_t* n2, double* a, 
                         magma_int_t* dtrd1, magma_int_t* dtrd2, magma_int_t* index);
void    lapackf77_dlaed4(magma_int_t* n, magma_int_t* i, double* d, double* z,
                         double* delta, double* rho, double* dlam, magma_int_t* info);
void    lapackf77_zsteqr(const char *compz, magma_int_t *n, double *D, double *E, 
                         magmaDoubleComplex *Z, magma_int_t *ldz, 
                         double *work, magma_int_t *info);
void    lapackf77_zstedc(const char *compz, magma_int_t *n, double *D, double *E, 
                         magmaDoubleComplex *Z, magma_int_t *ldz, 
                         magmaDoubleComplex *work, magma_int_t *ldwork, 
                         DWORKFORZ_AND_LD magma_int_t *iwork, magma_int_t *liwork,
                         magma_int_t *info);
void    lapackf77_zstein(magma_int_t *n, double *d__, double *e, 
                         magma_int_t *m, double *w, magma_int_t *iblock, magma_int_t *isplit, 
                         magmaDoubleComplex *z__, magma_int_t *ldz, double *work, magma_int_t *iwork, 
                         magma_int_t *ifail, magma_int_t *info);
void    lapackf77_zstemr(char *jobz, char *range, magma_int_t *n, double *d__, double *e, 
                         double *vl, double *vu, magma_int_t *il, magma_int_t *iu, magma_int_t *m,
                         double *w, magmaDoubleComplex *z__, magma_int_t *ldz, magma_int_t *nzc, 
                         magma_int_t *isuppz, magma_int_t *tryrac, double *work, magma_int_t *lwork, 
                         magma_int_t *iwork, magma_int_t *liwork, magma_int_t *info);
void    lapackf77_ztrtri(const char *uplo, const char *diag, magma_int_t *n,
                         magmaDoubleComplex *a, magma_int_t *lda, magma_int_t *info);
#if defined(PRECISION_z) || defined(PRECISION_c)
void    lapackf77_zsymv(const char *uplo, const magma_int_t *N, const magmaDoubleComplex *alpha, 
                        const magmaDoubleComplex *A, const magma_int_t *lda, 
                        const magmaDoubleComplex *X, const magma_int_t *incX,
                        const magmaDoubleComplex *beta, 
                        magmaDoubleComplex *Y, const magma_int_t *incY);
#endif
void    lapackf77_zung2r(magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                         magmaDoubleComplex *a, magma_int_t *lda,
                         const magmaDoubleComplex *tau, magmaDoubleComplex *work,
                         magma_int_t *info);
void    lapackf77_zungbr(const char *vect, magma_int_t *m, magma_int_t *n, magma_int_t *k,
                         magmaDoubleComplex *a, magma_int_t *lda, const magmaDoubleComplex *tau,
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zunghr(magma_int_t *n, magma_int_t *ilo, magma_int_t *ihi, 
                         magmaDoubleComplex *a, magma_int_t *lda, const magmaDoubleComplex *tau,
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zunglq(magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                         magmaDoubleComplex *a, magma_int_t *lda, const magmaDoubleComplex *tau, 
                         magmaDoubleComplex *work, magma_int_t *ldwork, magma_int_t *info);
void    lapackf77_zungql(magma_int_t *, magma_int_t *, magma_int_t *,
                         magmaDoubleComplex *, magma_int_t *, magmaDoubleComplex *, 
                         magmaDoubleComplex *, magma_int_t *, magma_int_t *);
void    lapackf77_zungqr(magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                         magmaDoubleComplex *a, magma_int_t *lda, const magmaDoubleComplex *tau, 
                         magmaDoubleComplex *work, magma_int_t *ldwork, magma_int_t *info);
void    lapackf77_zungtr(const char *uplo, magma_int_t *n, 
                         magmaDoubleComplex *a, magma_int_t *lda, const magmaDoubleComplex *tau, 
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zunm2r(const char *side, const char *trans, 
                         magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                         const magmaDoubleComplex *a, magma_int_t *lda, 
                         const magmaDoubleComplex *tau, magmaDoubleComplex *c, magma_int_t *ldc,
                         magmaDoubleComplex *work, magma_int_t *info);
void    lapackf77_zunmbr(const char *vect, const char *side, const char *trans,
                         magma_int_t *M, magma_int_t *N, magma_int_t *K, 
                         magmaDoubleComplex *A, magma_int_t *lda, magmaDoubleComplex *Tau,
                         magmaDoubleComplex *C, magma_int_t *ldc, 
                         magmaDoubleComplex *work, magma_int_t *ldwork, magma_int_t *info);
void    lapackf77_zunmlq(const char *side, const char *trans, 
                         magma_int_t *m, magma_int_t *n, magma_int_t *k,
                         const magmaDoubleComplex *a, magma_int_t *lda, 
                         const magmaDoubleComplex *tau, magmaDoubleComplex *c, magma_int_t *ldc, 
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zunmql(const char *side, const char *trans, 
                         magma_int_t *m, magma_int_t *n, magma_int_t *k,
                         const magmaDoubleComplex *a, magma_int_t *lda, 
                         const magmaDoubleComplex *tau, magmaDoubleComplex *c, magma_int_t *ldc,
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zunmqr(const char *side, const char *trans, 
                         magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                         const magmaDoubleComplex *a, magma_int_t *lda, 
                         const magmaDoubleComplex *tau, magmaDoubleComplex *c, magma_int_t *ldc, 
                         magmaDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zunmtr(const char *side, const char *uplo, const char *trans,
                         magma_int_t *M, magma_int_t *N,
                         magmaDoubleComplex *A, magma_int_t *lda, magmaDoubleComplex *Tau,
                         magmaDoubleComplex *C, magma_int_t *ldc, 
                         magmaDoubleComplex *work, magma_int_t *ldwork, magma_int_t *info);


  /*
   * Testing functions
   */

#if defined(PRECISION_z) || defined(PRECISION_c)

void    lapackf77_zbdt01(int *m, int *n, int *kd, magmaDoubleComplex *A, int *lda, 
                         magmaDoubleComplex *Q, int *ldq, double *D, double *E, 
                         magmaDoubleComplex *PT, int *ldpt, magmaDoubleComplex *work, 
                         double *rwork, double *resid);
void    lapackf77_zget22(const char *transa, const char *transe, const char *transw, int *n,
                         magmaDoubleComplex *a, int *lda, magmaDoubleComplex *e, int *lde,
                         magmaDoubleComplex *w, magmaDoubleComplex *work,
                         double *rwork, double *result);
void    lapackf77_zhet21(int *itype, const char *uplo, int *n, int *kband, 
                         magmaDoubleComplex *A, int *lda, double *D, double *E, 
                         magmaDoubleComplex *U, int *ldu, magmaDoubleComplex *V, int *ldv, 
                         magmaDoubleComplex *TAU, magmaDoubleComplex *work,
                         double *rwork, double *result);
void    lapackf77_zhst01(int *n, int *ilo, int *ihi, magmaDoubleComplex *A, int *lda, 
                         magmaDoubleComplex *H, int *ldh, magmaDoubleComplex *Q, int *ldq,
                         magmaDoubleComplex *work, int *lwork, double *rwork, double *result);
void    lapackf77_zstt21(int *n, int *kband, double *AD, double *AE, double *SD,
                         double *SE, magmaDoubleComplex *U, int *ldu, 
                         magmaDoubleComplex *work, double *rwork, double *result);
void    lapackf77_zunt01(const char *rowcol, int *m, int *n, magmaDoubleComplex *U, int *ldu,
                         magmaDoubleComplex *work, int *lwork, double *rwork, double *resid);

#else

void    lapackf77_zbdt01(int *m, int *n, int *kd, magmaDoubleComplex *A, int *lda, 
                         magmaDoubleComplex *Q, int *ldq, double *D, double *E, 
                         magmaDoubleComplex *PT, int *ldpt, 
                         magmaDoubleComplex *work, double *resid);
void    lapackf77_zget22(const char *transa, const char *transe, const char *transw, int *n,
                         magmaDoubleComplex *a, int *lda, magmaDoubleComplex *e, int *lde,
                         magmaDoubleComplex *wr, magmaDoubleComplex *wi, 
                         double *work, double *result);
void    lapackf77_zhet21(int *itype, const char *uplo, int *n, int *kband, 
                         magmaDoubleComplex *A, int *lda, double *D, double *E,
                         magmaDoubleComplex *U, int *ldu, magmaDoubleComplex *V, int *ldv, 
                         magmaDoubleComplex *TAU, magmaDoubleComplex *work, double *result);
void    lapackf77_zhst01(int *n, int *ilo, int *ihi, magmaDoubleComplex *A, int *lda, 
                         magmaDoubleComplex *H, int *ldh, magmaDoubleComplex *Q, int *ldq, 
                         magmaDoubleComplex *work, int *lwork, double *result);
void    lapackf77_zstt21(int *n, int *kband, double *AD, double *AE, double *SD, 
                         double *SE, magmaDoubleComplex *U, int *ldu, 
                         magmaDoubleComplex *work, double *result);
void    lapackf77_zunt01(const char *rowcol, int *m, int *n, magmaDoubleComplex *U, int *ldu,
                         magmaDoubleComplex *work, int *lwork, double *resid);
#endif

void    lapackf77_zlarfy(const char *uplo, int *N, magmaDoubleComplex *V, int *incv, 
                         magmaDoubleComplex *tau, magmaDoubleComplex *C, int *ldc, 
                         magmaDoubleComplex *work);
void    lapackf77_zqrt02(int *m, int *n, int *k, magmaDoubleComplex *A, magmaDoubleComplex *AF,
                         magmaDoubleComplex *Q, magmaDoubleComplex *R, int *lda, 
                         magmaDoubleComplex *TAU, magmaDoubleComplex *work, int *lwork,
                         double *rwork, double *result);

#ifdef __cplusplus
}
#endif

#undef DWORKFORZ 
#undef DWORKFORZ_AND_LD
#undef WSPLIT
#undef PRECISION_z
#endif /* MAGMA ZLAPACK */
