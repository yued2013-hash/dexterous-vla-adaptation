import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
from reference.contracts import (HAND_NAMES,matrix_to_rot6,rot6_to_matrix,quaternion_wxyz_to_rot6,
    named_compact_to_control,absolute_to_relative,relative_to_absolute,to_quaternion_boundary,split_position_force)
from reference.training_math import distributed_scalar_terms,linear_flow_pair,match_noise

class Contracts(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(19)
        self.anchor=np.r_[np.array([.1,.2,.3]),matrix_to_rot6(Rotation.from_euler('xyz',[30,20,10],degrees=True).as_matrix()),np.arange(16.)]
        self.chunk=np.tile(self.anchor,(16,1));self.chunk[:,:3]+=self.rng.normal(0,.03,(16,3))
        self.chunk[:,3:9]=matrix_to_rot6(Rotation.random(16,random_state=self.rng).as_matrix())
    def test_rotation_roundtrip(self):
        r=Rotation.random(100,random_state=self.rng).as_matrix()
        np.testing.assert_allclose(rot6_to_matrix(matrix_to_rot6(r)),r,atol=1e-12)
    def test_quaternion_sign_equivalence(self):
        q=np.array([.5,.5,.5,.5]);np.testing.assert_allclose(quaternion_wxyz_to_rot6(q),quaternion_wxyz_to_rot6(-q))
    def test_degenerate_rotation_rejected(self):
        with self.assertRaises(ValueError):rot6_to_matrix(np.zeros(6))
    def test_fixed_anchor_roundtrip_and_hand(self):
        relative=absolute_to_relative(self.anchor,self.chunk)
        np.testing.assert_allclose(relative_to_absolute(self.anchor,relative),self.chunk,atol=1e-12)
        np.testing.assert_array_equal(relative[:,9:],self.chunk[:,9:])
    def test_world_frame_order_is_not_body_frame(self):
        relative=absolute_to_relative(self.anchor,self.chunk)
        ra=rot6_to_matrix(self.anchor[3:9]);rt=rot6_to_matrix(self.chunk[0,3:9])
        np.testing.assert_allclose(rot6_to_matrix(relative[0,3:9]),rt@ra.T,atol=1e-12)
        self.assertGreater(np.max(np.abs(rt@ra.T-ra.T@rt)),.01)
    def test_named_compact_reorders_channels(self):
        names=['ee_valid','ee_x','ee_y','ee_z','ee_qw','ee_qx','ee_qy','ee_qz','hand_valid',*reversed(HAND_NAMES)]
        values=np.array([1,.1,.2,.3,1,0,0,0,1,*range(16)])
        x=named_compact_to_control(values,names)
        np.testing.assert_array_equal(x[9:],np.arange(16)[::-1])
    def test_duplicate_names_rejected(self):
        with self.assertRaises(ValueError):named_compact_to_control(np.ones(25),['same']*25)
    def test_boundary_shape_and_orientation(self):
        x=to_quaternion_boundary(self.chunk)
        self.assertEqual(x.shape,(16,23))
        np.testing.assert_allclose(Rotation.from_quat(x[:,3:7]).as_matrix(),rot6_to_matrix(self.chunk[:,3:9]),atol=1e-12)
    def test_force_is_separate_measurement(self):
        p=np.c_[self.chunk,np.full((16,5),3.)];control,force=split_position_force(p)
        self.assertEqual(control.shape,(16,25));self.assertEqual(force.shape,(16,5));np.testing.assert_array_equal(control,self.chunk)
    def test_ddp_unequal_counts(self):
        terms=distributed_scalar_terms([6,10],[3,2])
        self.assertAlmostEqual(terms.mean(),16/5)
        self.assertNotAlmostEqual(terms.mean(),np.mean([6/3,10/2]))
    def test_flow_endpoints(self):
        x0=self.rng.normal(size=(3,16,30));x1=self.rng.normal(size=x0.shape)
        np.testing.assert_allclose(linear_flow_pair(x0,x1,0)[0],x0)
        np.testing.assert_allclose(linear_flow_pair(x0,x1,1)[0],x1)
    def test_noise_matching(self):
        data=self.rng.normal(size=(5,16,30));noise=data[[3,1,4,0,2]]
        np.testing.assert_array_equal(match_noise(noise,data),data)

if __name__=='__main__':unittest.main()
