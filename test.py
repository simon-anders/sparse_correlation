import scipy
import sparse_correlation

m = scipy.sparse.csc_array(
[[1,2,0,3.],
 [2,1,0,0],
 [0,0,1,0]])

print( sparse_correlation.csc_corrcoef( to_csc_jitclass(m), 0, 1 ) )
