import scipy
from spm_corrcoef import *


m = scipy.sparse.csc_matrix(
[[1,2,0,3.],
 [2,1,0,0],
 [0,0,1,0]])

print( csc_corrcoef( to_csc_jitclass(m), 0, 1 ) )
