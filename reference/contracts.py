"""Independent NumPy reference for recorded policy contracts, not a robot SDK.

This intentionally rejects invalid input rotations; the copied PyTorch production
decoder additionally has its own guarded fallback logic. Do not interchange them.
"""
import numpy as np
from scipy.spatial.transform import Rotation

HAND_NAMES=(
 'hand_thumb_base','hand_thumb_abd','hand_thumb_horiz_abd','hand_thumb_tip',
 'hand_index_abd','hand_index_base','hand_index_tip',
 'hand_middle_abd','hand_middle_base','hand_middle_tip',
 'hand_ring_abd','hand_ring_base','hand_ring_tip',
 'hand_pinky_abd','hand_pinky_base','hand_pinky_tip',
)

def finite(values,width):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim<1 or x.shape[-1]!=width or not np.isfinite(x).all():
        raise ValueError(f'Expected finite (...,{width}) values')
    return x

def matrix_to_rot6(matrix):
    m=np.asarray(matrix,dtype=np.float64)
    if m.shape[-2:]!=(3,3) or not np.isfinite(m).all():raise ValueError('Expected finite rotation matrices')
    return np.concatenate([m[..., :,0],m[..., :,1]],axis=-1)

def rot6_to_matrix(values):
    x=finite(values,6)
    first=x[...,:3]; n1=np.linalg.norm(first,axis=-1,keepdims=True)
    if np.any(n1<1e-10):raise ValueError('Degenerate first rotation column')
    b1=first/n1
    b2=x[...,3:]-np.sum(x[...,3:]*b1,axis=-1,keepdims=True)*b1
    n2=np.linalg.norm(b2,axis=-1,keepdims=True)
    if np.any(n2<1e-10):raise ValueError('Degenerate second rotation column')
    b2=b2/n2
    return np.stack([b1,b2,np.cross(b1,b2)],axis=-1)

def quaternion_wxyz_to_rot6(q):
    q=finite(q,4)
    if np.any(np.linalg.norm(q,axis=-1)<1e-10):raise ValueError('Zero quaternion')
    shape=q.shape[:-1]
    matrix=Rotation.from_quat(q[..., [1,2,3,0]].reshape(-1,4)).as_matrix().reshape(*shape,3,3)
    return matrix_to_rot6(matrix)

def named_compact_to_control(values,names):
    x=finite(values,25);names=tuple(names)
    required={'ee_valid','ee_x','ee_y','ee_z','ee_qw','ee_qx','ee_qy','ee_qz','hand_valid',*HAND_NAMES}
    if len(names)!=25 or len(set(names))!=25 or set(names)!=required:raise ValueError('Named compact schema mismatch')
    idx={name:i for i,name in enumerate(names)}
    if np.any(x[...,idx['ee_valid']]<=.5) or np.any(x[...,idx['hand_valid']]<=.5):raise ValueError('Invalid measured state')
    xyz=x[..., [idx[n] for n in ['ee_x','ee_y','ee_z']]]
    q=x[..., [idx[n] for n in ['ee_qw','ee_qx','ee_qy','ee_qz']]]
    hand=x[..., [idx[n] for n in HAND_NAMES]]
    return np.concatenate([xyz,quaternion_wxyz_to_rot6(q),hand],axis=-1)

def absolute_to_relative(anchor,chunk):
    anchor=finite(anchor,25);chunk=finite(chunk,25)
    if anchor.ndim!=1 or chunk.ndim!=2:raise ValueError('Reference expects one anchor and a (T,25) chunk')
    ra=rot6_to_matrix(anchor[3:9]);rt=rot6_to_matrix(chunk[:,3:9])
    out=chunk.copy();out[:,:3]-=anchor[:3]
    out[:,3:9]=matrix_to_rot6(rt@ra.T)
    return out

def relative_to_absolute(anchor,chunk):
    anchor=finite(anchor,25);chunk=finite(chunk,25)
    if anchor.ndim!=1 or chunk.ndim!=2:raise ValueError('Reference expects one anchor and a (T,25) chunk')
    ra=rot6_to_matrix(anchor[3:9]);rd=rot6_to_matrix(chunk[:,3:9])
    out=chunk.copy();out[:,:3]+=anchor[:3]
    out[:,3:9]=matrix_to_rot6(rd@ra)
    return out

def to_quaternion_boundary(control):
    """Return xyz+xyzw+canonical hand. Hardware ordering/units/IK are NOT supplied."""
    x=finite(control,25);r=rot6_to_matrix(x[...,3:9]);shape=x.shape[:-1]
    xyzw=Rotation.from_matrix(r.reshape(-1,3,3)).as_quat().reshape(*shape,4)
    return np.concatenate([x[...,:3],xyzw,x[...,9:]],axis=-1)

def split_position_force(prediction):
    """Separate control25 and predicted future measurement5; sends no commands."""
    x=finite(prediction,30)
    return x[...,:25].copy(),x[...,25:].copy()
