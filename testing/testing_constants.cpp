#include <stdio.h>
#include <assert.h>

#include "magma.h"

int main( int argc, char** argv )
{
    printf( "testing lapack -> magma constants\n" );
    assert( magma_trans_const( 'N' ) == MagmaNoTrans   );
    assert( magma_trans_const( 'T' ) == MagmaTrans     );
    assert( magma_trans_const( 'C' ) == MagmaConjTrans );
    assert( magma_trans_const( 'n' ) == MagmaNoTrans   );
    assert( magma_trans_const( 't' ) == MagmaTrans     );
    assert( magma_trans_const( 'c' ) == MagmaConjTrans );
    
    assert( magma_side_const( 'L' ) == MagmaLeft       );
    assert( magma_side_const( 'R' ) == MagmaRight      );
    assert( magma_side_const( 'l' ) == MagmaLeft       );
    assert( magma_side_const( 'r' ) == MagmaRight      );
    
    assert( magma_diag_const( 'N' ) == MagmaNonUnit    );
    assert( magma_diag_const( 'U' ) == MagmaUnit       );
    assert( magma_diag_const( 'n' ) == MagmaNonUnit    );
    assert( magma_diag_const( 'u' ) == MagmaUnit       );
    
    assert( magma_uplo_const( 'U' ) == MagmaUpper      );
    assert( magma_uplo_const( 'L' ) == MagmaLower      );
    assert( magma_uplo_const( 'A' ) == MagmaFull       );
    assert( magma_uplo_const( 'u' ) == MagmaUpper      );
    assert( magma_uplo_const( 'l' ) == MagmaLower      );
    assert( magma_uplo_const( 'a' ) == MagmaFull       );
    
    printf( "testing magma -> lapack constants\n" );
    assert( lapack_const( MagmaRowMajor      )[0] == 'R' );
    assert( lapack_const( MagmaColMajor      )[0] == 'C' );
    assert( lapack_const( MagmaNoTrans       )[0] == 'N' );
    assert( lapack_const( MagmaTrans         )[0] == 'T' );
    assert( lapack_const( MagmaConjTrans     )[0] == 'C' );
    assert( lapack_const( MagmaUpper         )[0] == 'U' );
    assert( lapack_const( MagmaLower         )[0] == 'L' );
    assert( lapack_const( MagmaFull          )[0] == 'A' );
    assert( lapack_const( MagmaNonUnit       )[0] == 'N' );
    assert( lapack_const( MagmaUnit          )[0] == 'U' );
    assert( lapack_const( MagmaLeft          )[0] == 'L' );
    assert( lapack_const( MagmaRight         )[0] == 'R' );
    assert( lapack_const( MagmaOneNorm       )[0] == 'O' );
    assert( lapack_const( MagmaFrobeniusNorm )[0] == 'F' );
    assert( lapack_const( MagmaInfNorm       )[0] == 'I' );
    assert( lapack_const( MagmaMaxNorm       )[0] == 'M' );
    assert( lapack_const( MagmaDistUniform   )[0] == 'U' );
    assert( lapack_const( MagmaDistSymmetric )[0] == 'S' );
    assert( lapack_const( MagmaDistNormal    )[0] == 'N' );
    assert( lapack_const( MagmaHermGeev      )[0] == 'H' );
    assert( lapack_const( MagmaHermPoev      )[0] == 'P' );
    assert( lapack_const( MagmaNonsymPosv    )[0] == 'N' );
    assert( lapack_const( MagmaSymPosv       )[0] == 'S' );
    assert( lapack_const( MagmaNoPacking     )[0] == 'N' );
    assert( lapack_const( MagmaPackSubdiag   )[0] == 'U' );
    assert( lapack_const( MagmaPackSupdiag   )[0] == 'L' );
    assert( lapack_const( MagmaPackColumn    )[0] == 'C' );
    assert( lapack_const( MagmaPackRow       )[0] == 'R' );
    assert( lapack_const( MagmaPackLowerBand )[0] == 'B' );
    assert( lapack_const( MagmaPackUpeprBand )[0] == 'Q' );
    assert( lapack_const( MagmaPackAll       )[0] == 'Z' );
    assert( lapack_const( MagmaNoVec         )[0] == 'N' );
    assert( lapack_const( MagmaVec           )[0] == 'V' );
    assert( lapack_const( MagmaForward       )[0] == 'F' );
    assert( lapack_const( MagmaBackward      )[0] == 'B' );
    assert( lapack_const( MagmaColumnwise    )[0] == 'C' );
    assert( lapack_const( MagmaRowwise       )[0] == 'R' );
    
    printf( "testing magma -> lapacke constants\n" );
    assert( lapacke_const( MagmaRowMajor      ) == 'R' );
    assert( lapacke_const( MagmaColMajor      ) == 'C' );
    assert( lapacke_const( MagmaNoTrans       ) == 'N' );
    assert( lapacke_const( MagmaTrans         ) == 'T' );
    assert( lapacke_const( MagmaConjTrans     ) == 'C' );
    assert( lapacke_const( MagmaUpper         ) == 'U' );
    assert( lapacke_const( MagmaLower         ) == 'L' );
    assert( lapacke_const( MagmaFull          ) == 'A' );
    assert( lapacke_const( MagmaNonUnit       ) == 'N' );
    assert( lapacke_const( MagmaUnit          ) == 'U' );
    assert( lapacke_const( MagmaLeft          ) == 'L' );
    assert( lapacke_const( MagmaRight         ) == 'R' );
    assert( lapacke_const( MagmaOneNorm       ) == 'O' );
    assert( lapacke_const( MagmaFrobeniusNorm ) == 'F' );
    assert( lapacke_const( MagmaInfNorm       ) == 'I' );
    assert( lapacke_const( MagmaMaxNorm       ) == 'M' );
    assert( lapacke_const( MagmaDistUniform   ) == 'U' );
    assert( lapacke_const( MagmaDistSymmetric ) == 'S' );
    assert( lapacke_const( MagmaDistNormal    ) == 'N' );
    assert( lapacke_const( MagmaHermGeev      ) == 'H' );
    assert( lapacke_const( MagmaHermPoev      ) == 'P' );
    assert( lapacke_const( MagmaNonsymPosv    ) == 'N' );
    assert( lapacke_const( MagmaSymPosv       ) == 'S' );
    assert( lapacke_const( MagmaNoPacking     ) == 'N' );
    assert( lapacke_const( MagmaPackSubdiag   ) == 'U' );
    assert( lapacke_const( MagmaPackSupdiag   ) == 'L' );
    assert( lapacke_const( MagmaPackColumn    ) == 'C' );
    assert( lapacke_const( MagmaPackRow       ) == 'R' );
    assert( lapacke_const( MagmaPackLowerBand ) == 'B' );
    assert( lapacke_const( MagmaPackUpeprBand ) == 'Q' );
    assert( lapacke_const( MagmaPackAll       ) == 'Z' );
    assert( lapacke_const( MagmaNoVec         ) == 'N' );
    assert( lapacke_const( MagmaVec           ) == 'V' );
    assert( lapacke_const( MagmaForward       ) == 'F' );
    assert( lapacke_const( MagmaBackward      ) == 'B' );
    assert( lapacke_const( MagmaColumnwise    ) == 'C' );
    assert( lapacke_const( MagmaRowwise       ) == 'R' );
    
    #ifdef HAVE_clAmdBlas
    printf( "testing magma -> clAmdBlas constants\n" );
    assert( amdblas_const( MagmaRowMajor      ) == clAmdBlasRowMajor    );
    assert( amdblas_const( MagmaColMajor      ) == clAmdBlasColumnMajor );
    assert( amdblas_const( MagmaNoTrans       ) == clAmdBlasNoTrans     );
    assert( amdblas_const( MagmaTrans         ) == clAmdBlasTrans       );
    assert( amdblas_const( MagmaConjTrans     ) == clAmdBlasConjTrans   );
    assert( amdblas_const( MagmaUpper         ) == clAmdBlasUpper       );
    assert( amdblas_const( MagmaLower         ) == clAmdBlasLower       );
    assert( amdblas_const( MagmaNonUnit       ) == clAmdBlasNonUnit     );
    assert( amdblas_const( MagmaUnit          ) == clAmdBlasUnit        );
    assert( amdblas_const( MagmaLeft          ) == clAmdBlasLeft        );
    assert( amdblas_const( MagmaRight         ) == clAmdBlasRight       );
    #endif
    
    #ifdef HAVE_CUBLAS
    printf( "testing magma -> cuBLAS constants\n" );
    assert( cublas_const( MagmaNoTrans       ) == CUBLAS_OP_N            );
    assert( cublas_const( MagmaTrans         ) == CUBLAS_OP_T            );
    assert( cublas_const( MagmaConjTrans     ) == CUBLAS_OP_C            );
    assert( cublas_const( MagmaUpper         ) == CUBLAS_FILL_MODE_UPPER );
    assert( cublas_const( MagmaLower         ) == CUBLAS_FILL_MODE_LOWER );
    assert( cublas_const( MagmaNonUnit       ) == CUBLAS_DIAG_NON_UNIT   );
    assert( cublas_const( MagmaUnit          ) == CUBLAS_DIAG_UNIT       );
    assert( cublas_const( MagmaLeft          ) == CUBLAS_SIDE_LEFT       );
    assert( cublas_const( MagmaRight         ) == CUBLAS_SIDE_RIGHT      );
    #endif
    
    #ifdef HAVE_CBLAS
    printf( "testing magma -> CBLAS constants\n" );
    assert( cblas_const( MagmaRowMajor      ) == CblasRowMajor  );
    assert( cblas_const( MagmaColMajor      ) == CblasColMajor  );
    assert( cblas_const( MagmaNoTrans       ) == CblasNoTrans   );
    assert( cblas_const( MagmaTrans         ) == CblasTrans     );
    assert( cblas_const( MagmaConjTrans     ) == CblasConjTrans );
    assert( cblas_const( MagmaUpper         ) == CblasUpper     );
    assert( cblas_const( MagmaLower         ) == CblasLower     );
    assert( cblas_const( MagmaNonUnit       ) == CblasNonUnit   );
    assert( cblas_const( MagmaUnit          ) == CblasUnit      );
    assert( cblas_const( MagmaLeft          ) == CblasLeft      );
    assert( cblas_const( MagmaRight         ) == CblasRight     );
    #endif
}
