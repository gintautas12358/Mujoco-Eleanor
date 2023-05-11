import numpy as np

import sys
import os
sys.path.append(os.getcwd())

from kuka.utils.quaternion import subQuat, mat2Quat, quat2eul
from kuka.utils.kinematics import forwardKinSite, forwardKinJacobianSite

from .MujocoPDController import MujocoPDController


class CartesianController(MujocoPDController):
    '''
    An inverse dynamics controller that used PD gains to compute a desired acceleration.
    '''

    def __init__(self,
                 sim_model, sim_data,
                 kp = 300, kd=None,
                 site_name='ee_site',
                ):
        super(CartesianController, self).__init__(sim_model, sim_data, kp, kd)

        self.site_name = site_name
        
        self.nominal_qpos = np.zeros(7)

        self.pos_set = None
        self.quat_set = None

    def error(self):
        return self.pose_error()

    def force_feedback(self):
        r_pseudo_J = self.right_pseudo_Jac()

        # solve: tau = (Jac)^T * F
        return r_pseudo_J.T @ self.sim_data.qfrc_constraint

    def pose_error(self):
        if self.pos_set is None or self.quat_set is None:
            raise ValueError("Function set_action was not called first")

        pos, quat = self._fk()
        
        dx = self.pos_set - pos
        dr = subQuat(self.quat_set, quat) # Original
        dframe = np.concatenate((dx,dr))
        return dframe
    
    
    def fk(self):
        pos, quat = self._fk()
        return np.append(pos, quat2eul(quat))
    
    def _fk(self):
        pos, mat = forwardKinSite(self.sim_model, self.sim_data, self.site_name, recompute=False)
        return pos, mat2Quat(mat)

    def Jac(self):
        jpos, jrot = forwardKinJacobianSite(self.sim_model, self.sim_data, self.site_name, recompute=False)
        J = np.vstack((jpos, jrot)) # full jacobian
        return J

    def right_pseudo_Jac(self, eps=0):
        J = self.Jac()
        
        pJ = J.T @ np.linalg.inv(J @ J.T + eps*np.eye(6)) 
        return pJ

    def left_pseudo_Jac(self, eps=0):
        J = self.Jac()
        
        pJ = np.linalg.inv(J.T @ J + eps*np.eye(7)) @ J.T
        return pJ