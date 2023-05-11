import numpy as np
import mujoco

import sys
import os
sys.path.append(os.getcwd())

from kuka.utils.quaternion import eul2quat

from .CartesianController import CartesianController


class ElasticCartesianPDController(CartesianController):
    '''
    A base for all joint based controllers
    '''

    def __init__(self,
                    sim_model, sim_data,
                    kp=300,
                    kd=None,
                    null_space_damping=10,
                    null_space_stiffness=100,
                    site_name='ee_site',
                    ):

        super(ElasticCartesianPDController, self).__init__(sim_model, sim_data, kp, kd, site_name)

        self.nominal_qpos = np.zeros(7)

        self.null_space_damping = null_space_damping
        self.null_space_stiffness = null_space_stiffness

    def set_gains(self, kp, kd):
        self.kp = kp
        if kd is None:
            self.kd = 2 * np.sqrt(kp)
        else:
            self.kd = kd

    def set_action(self, action):
        '''
        Set the setpoint.
        '''
        self.scale = 1
        action = action * self.scale

        dx = action[0:3].astype(np.float64)
        dr = action[3:6].astype(np.float64)

        self.pos_set = dx
        self.quat_set = eul2quat(dr)

    def get_torque(self):
        '''
        Update the PD setpoint and compute the torque.
        '''
        self.sim_data.qacc = self.controlLaw()


        # # gravity compensation
        # G = self.sim_data.qfrc_bias

        # # Sum the torques.
        # out_torque = torque + G
        # self.sim_data.ctrl = out_torque
        # # self.sim_data.ctrl = torque

        mujoco.mj_inverse(self.sim_model, self.sim_data)
        id_torque = self.sim_data.qfrc_inverse[self.sim_actuators_idx].copy()

        self.sim_data.ctrl = id_torque

        return id_torque
    
    # http://www.diag.uniroma1.it/deluca/rob2_en/13_CartesianControl.pdf
    # slide 3 elastic
    def controlLaw(self):
        return self.right_pseudo_Jac(eps=0) @ (self.kp * self.pose_error()) - self.kd * self.sim_data.qvel
    
    
    

    
