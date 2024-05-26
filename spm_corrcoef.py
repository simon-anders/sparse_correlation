import math
import numpy
import numba
import scipy

_jitclass_spec_CscSparseMatrixJit = [
    ('data',    numba.types.float64[:]),
    ('indices', numba.types.int32[:]),
    ('indptr',  numba.types.int32[:]),
    ('shape',   numba.types.UniTuple(numba.types.int32, 2) )
]

@numba.experimental.jitclass(_jitclass_spec_CscSparseMatrixJit)
class CscSparseMatrixJit:
    def __init__(self, data, indices, indptr, shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

def to_csc_jitclass( m ):
    if not scipy.sparse.isspmatrix_csc(m):
        raise TypeError( "not as CSC sparse matrix" )
    return CscSparseMatrixJit( m.data, m.indices, m.indptr, m.shape )


@numba.njit((CscSparseMatrixJit.class_type.instance_type, 
        numba.types.int32, numba.types.int32))
def csc_corrcoef( spm, col1, col2 ):
    """Given a CSC sparse matrix 'spm', this calculates the Pearson
    correlation coefficient between columns 'col1' and 'col2'.
    Use the function 'to_csc_jitclass' to convert you scipy CSC sparse
    matrix to the class that this function expects.
    """
    col1_mean = 0
    for i in range( spm.indptr[col1], spm.indptr[col1+1] ):
        col1_mean += spm.data[i]
    col1_mean /= spm.shape[0]    

    col1_var = 0
    for i in range( spm.indptr[col1], spm.indptr[col1+1] ):
        col1_var += ( spm.data[i] - col1_mean )**2
    col1_var += col1_mean**2 * ( spm.shape[0] - spm.indptr[col1+1] + spm.indptr[col1] )
    col1_var /= spm.shape[0]    
    
    col2_mean = 0
    for i in range( spm.indptr[col2], spm.indptr[col2+1] ):
        col2_mean += spm.data[i]
    col2_mean /= spm.shape[0]    

    col2_var = 0
    for i in range( spm.indptr[col2], spm.indptr[col2+1] ):
        col2_var += ( spm.data[i] - col2_mean )**2
    col2_var += col2_mean**2 * ( spm.shape[0] - spm.indptr[col2+1] + spm.indptr[col2] )
    col2_var /= spm.shape[0]    

    if col1_var == 0. or col2_var == 0.:
        return numpy.nan
    
    ptr1 = spm.indptr[col1]
    ptr2 = spm.indptr[col2]
    prodsum = 0
    prodcount = 0
    while True:
        if spm.indices[ptr1] == spm.indices[ptr2]:
            prodsum += ( spm.data[ptr1] - col1_mean ) * ( spm.data[ptr2] - col2_mean )
            prodcount += 1
            ptr1 += 1
            ptr2 += 1
        elif spm.indices[ptr1] < spm.indices[ptr2]:
            prodsum -= ( spm.data[ptr1] - col1_mean ) * col2_mean
            prodcount += 1
            ptr1 += 1
        else: # spm.indices[ptr1] > spm.indices[ptr2]
            prodsum -= col1_mean * ( spm.data[ptr2] - col2_mean )
            prodcount += 1
            ptr2 += 1
        if ptr1 >= spm.indptr[col1+1] or ptr2 >= spm.indptr[col2+1]:
            break
    
    prodsum += ( spm.shape[0] - prodcount ) * col1_mean * col2_mean 
    return prodsum / ( spm.shape[0] * math.sqrt( col1_var * col2_var ) )
