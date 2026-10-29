"""Independent numerical references; not a GPU trainer or performance benchmark."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def distributed_scalar_terms(local_sums,local_counts):
    sums=np.asarray(local_sums,dtype=np.float64)
    counts=np.asarray(local_counts,dtype=np.float64)
    if sums.ndim!=1 or sums.shape!=counts.shape or not len(sums) or not np.isfinite(sums).all() or not np.isfinite(counts).all() or np.any(counts<0):raise ValueError('Invalid scalar reductions')
    # The production implementation all-reduces detached counts; DDP averages gradients.
    return len(sums)*sums/max(float(counts.sum()),1)

def linear_flow_pair(noise,data,t):
    x0=np.asarray(noise,dtype=np.float64);x1=np.asarray(data,dtype=np.float64)
    if x0.shape!=x1.shape or not np.isfinite(x0).all() or not np.isfinite(x1).all() or not 0<=t<=1:raise ValueError('Invalid flow pair')
    return (1-t)*x0+t*x1,x1-x0

def match_noise(noise,data):
    x0=np.asarray(noise,dtype=np.float64);x1=np.asarray(data,dtype=np.float64)
    if x0.shape!=x1.shape or x0.ndim<2:raise ValueError('Batch shapes must match')
    flat0=x0.reshape(len(x0),-1);flat1=x1.reshape(len(x1),-1)
    cost=((flat1[:,None,:]-flat0[None,:,:])**2).sum(-1)
    rows,cols=linear_sum_assignment(cost)
    if not np.array_equal(rows,np.arange(len(x0))):raise ValueError('Unexpected matching rows')
    return x0[cols]
