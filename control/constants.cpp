#include <assert.h>

#include "magma_types.h"

// ----------------------------------------
// Convert LAPACK character constant to MAGMA constant.
// This is a one-to-many mapping, requiring multiple translators
// (e.g., "N" can be NoTrans or NonUnit).

int magma_trans_const( char lapack_char ) {
    switch( lapack_char ) {
        case 'n':
        case 'N':
            return MagmaNoTrans;
        case 't':
        case 'T':
            return MagmaTrans;
        case 'c':
        case 'C':
            return MagmaConjTrans;
        default:
            assert( 0 );  // complain loudly
            return 0;
    }
}

int magma_side_const( char lapack_char ) {
    switch( lapack_char ) {
        case 'l':
        case 'L':
            return MagmaLeft;
        case 'r':
        case 'R':
            return MagmaRight;
        default:
            assert( 0 );  // complain loudly
            return 0;
    }
}

int magma_diag_const( char lapack_char ) {
    switch( lapack_char ) {
        case 'n':
        case 'N':
            return MagmaNonUnit;
        case 'u':
        case 'U':
            return MagmaUnit;
        default:
            assert( 0 );  // complain loudly
            return 0;
    }
}

int magma_uplo_const( char lapack_char ) {
    switch( lapack_char ) {
        case 'u':
        case 'U':
            return MagmaUpper;
        case 'l':
        case 'L':
            return MagmaLower;
        default:
            return MagmaFull;
    }
}


// ----------------------------------------
// Convert magma constants to lapack constants.
// This list is consistent with plasma/core_blas/global.c

const char *magma2lapack_constants[] =
{
    "",                                      //  0
    "", "", "", "", "", "", "", "", "", "",  //  1-10
    "", "", "", "", "", "", "", "", "", "",  // 11-20
    "", "", "", "", "", "", "", "", "", "",  // 21-30
    "", "", "", "", "", "", "", "", "", "",  // 31-40
    "", "", "", "", "", "", "", "", "", "",  // 41-50
    "", "", "", "", "", "", "", "", "", "",  // 51-60
    "", "", "", "", "", "", "", "", "", "",  // 61-70
    "", "", "", "", "", "", "", "", "", "",  // 71-80
    "", "", "", "", "", "", "", "", "", "",  // 81-90
    "", "", "", "", "", "", "", "", "", "",  // 91-100
    "Row",                                   // 101: MagmaRowMajor
    "Column",                                // 102: MagmaColMajor
    "", "", "", "", "", "", "", "",          // 103-110
    "No transpose",                          // 111: MagmaNoTrans
    "Transpose",                             // 112: MagmaTrans
    "Conjugate transpose",                   // 113: MagmaConjTrans
    "", "", "", "", "", "", "",              // 114-120
    "Upper",                                 // 121: MagmaUpper
    "Lower",                                 // 122: MagmaLower
    "All",                                   // 123: MagmaUpperLower
    "", "", "", "", "", "", "",              // 124-130
    "Non-unit",                              // 131: MagmaNonUnit
    "Unit",                                  // 132: MagmaUnit
    "", "", "", "", "", "", "", "",          // 133-140
    "Left",                                  // 141: MagmaLeft
    "Right",                                 // 142: MagmaRight
    "", "", "", "", "", "", "", "",          // 143-150
    "", "", "", "", "", "", "", "", "", "",  // 151-160
    "", "", "", "", "", "", "", "", "", "",  // 161-170
    "One norm",                              // 171: MagmaOneNorm
    "",                                      // 172: MagmaRealOneNorm
    "",                                      // 173: MagmaTwoNorm
    "Frobenius norm",                        // 174: MagmaFrobeniusNorm
    "Infinity norm",                         // 175: MagmaInfNorm
    "",                                      // 176: MagmaRealInfNorm
    "Maximum norm",                          // 177: MagmaMaxNorm
    "",                                      // 178: MagmaRealMaxNorm
    "", "",                                  // 179-180
    "", "", "", "", "", "", "", "", "", "",  // 181-190
    "", "", "", "", "", "", "", "", "", "",  // 191-200
    "Uniform",                               // 201: MagmaDistUniform
    "Symmetric",                             // 202: MagmaDistSymmetric
    "Normal",                                // 203: MagmaDistNormal
    "", "", "", "", "", "", "",              // 204-210
    "", "", "", "", "", "", "", "", "", "",  // 211-220
    "", "", "", "", "", "", "", "", "", "",  // 221-230
    "", "", "", "", "", "", "", "", "", "",  // 231-240
    "Hermitian",                             // 241 MagmaHermGeev
    "Positive ev Hermitian",                 // 242 MagmaHermPoev
    "NonSymmetric pos sv",                   // 243 MagmaNonsymPosv
    "Symmetric pos sv",                      // 244 MagmaSymPosv
    "", "", "", "", "", "",                  // 245-250
    "", "", "", "", "", "", "", "", "", "",  // 251-260
    "", "", "", "", "", "", "", "", "", "",  // 261-270
    "", "", "", "", "", "", "", "", "", "",  // 271-280
    "", "", "", "", "", "", "", "", "", "",  // 281-290
    "No Packing",                            // 291 MagmaNoPacking
    "U zero out subdiag",                    // 292 MagmaPackSubdiag
    "L zero out superdiag",                  // 293 MagmaPackSupdiag
    "C",                                     // 294 MagmaPackColumn
    "R",                                     // 295 MagmaPackRow
    "B",                                     // 296 MagmaPackLowerBand
    "Q",                                     // 297 MagmaPackUpeprBand
    "Z",                                     // 298 MagmaPackAll
    "", "",                                  // 299-300
    "No vectors",                            // 301 MagmaNoVec
    "Vectors needed",                        // 302 MagmaVec
    "I",                                     // 303 MagmaIVec
    "All",                                   // 304 MagmaAllVec
    "Some",                                  // 305 MagmaSomeVec
    "Overwrite",                             // 306 MagmaOverwriteVec
    "", "", "", "",                          // 307-310
    "", "", "", "", "", "", "", "", "", "",  // 311-320
    "", "", "", "", "", "", "", "", "", "",  // 321-330
    "", "", "", "", "", "", "", "", "", "",  // 331-340
    "", "", "", "", "", "", "", "", "", "",  // 341-350
    "", "", "", "", "", "", "", "", "", "",  // 351-360
    "", "", "", "", "", "", "", "", "", "",  // 361-370
    "", "", "", "", "", "", "", "", "", "",  // 371-380
    "", "", "", "", "", "", "", "", "", "",  // 381-390
    "Forward",                               // 391: MagmaForward
    "Backward",                              // 392: MagmaBackward
    "", "", "", "", "", "", "", "",          // 393-400
    "Columnwise",                            // 401: MagmaColumnwise
    "Rowwise",                               // 402: MagmaRowwise
    "", "", "", "", "", "", "", ""           // 403-410
    // Remember to add a comma!
};

char lapacke_const( int magma_const ) {
    assert( magma_const >= MagmaMinConst );
    assert( magma_const <= MagmaMaxConst );
    return magma2lapack_constants[ magma_const ][0];
}

const char* lapack_const( int magma_const ) {
    assert( magma_const >= MagmaMinConst );
    assert( magma_const <= MagmaMaxConst );
    return magma2lapack_constants[ magma_const ];
}


// ----------------------------------------
// Convert magma constants to clAmdBlas constants.

#ifdef HAVE_clAmdBlas
const int magma2amdblas_constants[] =
{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,                      // 100
    clAmdBlasRowMajor,      // 101: MagmaRowMajor
    clAmdBlasColumnMajor,   // 102: MagmaColMajor
    0, 0, 0, 0, 0, 0, 0, 0,
    clAmdBlasNoTrans,       // 111: MagmaNoTrans
    clAmdBlasTrans,         // 112: MagmaTrans
    clAmdBlasConjTrans,     // 113: MagmaConjTrans
    0, 0, 0, 0, 0, 0, 0,
    clAmdBlasUpper,         // 121: MagmaUpper
    clAmdBlasLower,         // 122: MagmaLower
    0,                      // 123: MagmaUpperLower
    0, 0, 0, 0, 0, 0, 0,
    clAmdBlasNonUnit,       // 131: MagmaNonUnit
    clAmdBlasUnit,          // 132: MagmaUnit
    0, 0, 0, 0, 0, 0, 0, 0,
    clAmdBlasLeft,          // 141: MagmaLeft
    clAmdBlasRight,         // 142: MagmaRight
    0, 0, 0, 0, 0, 0, 0, 0
    // Remember to add a comma!
};

int amdblas_const( int magma_const ) {
    assert( magma_const >= MagmaMinConst );
    assert( magma_const <= MagmaMaxConst );
    return magma2amdblas_constants[ magma_const ];
}

clAmdBlasOrder       amdblas_order_const( int magma_const ) {
    assert( magma_const >= MagmaRowMajor );
    assert( magma_const <= MagmaColMajor );
    return (clAmdBlasOrder)     magma2amdblas_constants[ magma_const ];
}

clAmdBlasTranspose   amdblas_trans_const( int magma_const ) {
    assert( magma_const >= MagmaNoTrans   );
    assert( magma_const <= MagmaConjTrans );
    return (clAmdBlasTranspose) magma2amdblas_constants[ magma_const ];
}

clAmdBlasSide        amdblas_side_const ( int magma_const ) {
    assert( magma_const >= MagmaLeft  );
    assert( magma_const <= MagmaRight );
    return (clAmdBlasSide)      magma2amdblas_constants[ magma_const ];
}

clAmdBlasDiag        amdblas_diag_const ( int magma_const ) {
    assert( magma_const >= MagmaNonUnit );
    assert( magma_const <= MagmaUnit    );
    return (clAmdBlasDiag)      magma2amdblas_constants[ magma_const ];
}

clAmdBlasUplo        amdblas_uplo_const ( int magma_const ) {
    assert( magma_const >= MagmaUpper );
    assert( magma_const <= MagmaLower );
    return (clAmdBlasUplo)      magma2amdblas_constants[ magma_const ];
}
#endif


// ----------------------------------------
// Convert magma constants to Nvidia CUBLAS constants.

#ifdef HAVE_CUBLAS
const int magma2cublas_constants[] =
{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,                      // 100
    0,                      // 101: MagmaRowMajor
    0,                      // 102: MagmaColMajor
    0, 0, 0, 0, 0, 0, 0, 0,
    CUBLAS_OP_N,            // 111: MagmaNoTrans
    CUBLAS_OP_T,            // 112: MagmaTrans
    CUBLAS_OP_C,            // 113: MagmaConjTrans
    0, 0, 0, 0, 0, 0, 0,
    CUBLAS_FILL_MODE_UPPER, // 121: MagmaUpper
    CUBLAS_FILL_MODE_LOWER, // 122: MagmaLower
    0,                      // 123: MagmaUpperLower
    0, 0, 0, 0, 0, 0, 0,
    CUBLAS_DIAG_NON_UNIT,   // 131: MagmaNonUnit
    CUBLAS_DIAG_UNIT,       // 132: MagmaUnit
    0, 0, 0, 0, 0, 0, 0, 0,
    CUBLAS_SIDE_LEFT,       // 141: MagmaLeft
    CUBLAS_SIDE_RIGHT,      // 142: MagmaRight
    0, 0, 0, 0, 0, 0, 0, 0
    // Remember to add a comma!
};

int cublas_const( int magma_const ) {
    assert( magma_const >= MagmaMinConst );
    assert( magma_const <= MagmaMaxConst );
    return magma2cublas_constants[ magma_const ];
}

cublasOperation_t    cublas_trans_const ( int magma_const ) {
    assert( magma_const >= MagmaNoTrans   );
    assert( magma_const <= MagmaConjTrans );
    return (cublasOperation_t)  magma2cublas_constants[ magma_const ];
}

cublasSideMode_t     cublas_side_const  ( int magma_const ) {
    assert( magma_const >= MagmaLeft  );
    assert( magma_const <= MagmaRight );
    return (cublasSideMode_t)   magma2cublas_constants[ magma_const ];
}

cublasDiagType_t     cublas_diag_const  ( int magma_const ) {
    assert( magma_const >= MagmaNonUnit );
    assert( magma_const <= MagmaUnit    );
    return (cublasDiagType_t)   magma2cublas_constants[ magma_const ];
}

cublasFillMode_t     cublas_uplo_const  ( int magma_const ) {
    assert( magma_const >= MagmaUpper );
    assert( magma_const <= MagmaLower );
    return (cublasFillMode_t)   magma2cublas_constants[ magma_const ];
}
#endif


// ----------------------------------------
// Convert magma constants to CBLAS constants.
// We assume that magma constants are consistent with cblas constants,
// so verify that with asserts.

#ifdef HAVE_CBLAS
enum CBLAS_ORDER     cblas_order_const  ( int magma_const ) {
    assert( magma_const >= MagmaRowMajor );
    assert( magma_const <= MagmaColMajor );
    assert( MagmaRowMajor == CblasRowMajor );
    return (enum CBLAS_ORDER)     magma_const;
}

enum CBLAS_TRANSPOSE cblas_trans_const  ( int magma_const ) {
    assert( magma_const >= MagmaNoTrans   );
    assert( magma_const <= MagmaConjTrans );
    assert( MagmaNoTrans == CblasNoTrans );
    return (enum CBLAS_TRANSPOSE) magma_const;
}

enum CBLAS_SIDE      cblas_side_const   ( int magma_const ) {
    assert( magma_const >= MagmaLeft  );
    assert( magma_const <= MagmaRight );
    assert( MagmaLeft == CblasLeft );
    return (enum CBLAS_SIDE)      magma_const;
}

enum CBLAS_DIAG      cblas_diag_const   ( int magma_const ) {
    assert( magma_const >= MagmaNonUnit );
    assert( magma_const <= MagmaUnit    );
    assert( MagmaUnit == CblasUnit );
    return (enum CBLAS_DIAG)      magma_const;
}

enum CBLAS_UPLO      cblas_uplo_const   ( int magma_const ) {
    assert( magma_const >= MagmaUpper );
    assert( magma_const <= MagmaLower );
    assert( MagmaUpper == CblasUpper );
    return (enum CBLAS_UPLO)      magma_const;
}
#endif
