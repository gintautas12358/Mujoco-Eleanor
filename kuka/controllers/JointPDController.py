import numpy as np

import sys
import os
sys.path.append(os.getcwd())

from kuka.utils.mujoco_utils import kuka_subtree_mass

from .JointController import JointController


class JointPDController(JointController):
    '''
    A base for all joint based controllers
    '''

    def __init__(self,
                    sim_model, sim_data,
                    kp=np.array([200, 600, 200, 500, 50, 50, 10]),
                    kd=np.array([40, 60, 5, 35, 5, 5, 0.01]),
                    site_name='ee_site'
                    ):

        super(JointPDController, self).__init__(sim_model, sim_data, kp, kd)

        self.site_name = site_name

        # Initialize setpoint.
        self.sim_qpos_set = sim_data.qpos[self.sim_qpos_idx].copy()
        self.sim_qvel_set = np.zeros(len(self.sim_qvel_idx))

    def set_action(self, action):
        '''
        Set the setpoint.
        '''

        self.sim_qpos_set = action

    def get_torque(self):
        '''
        Update the PD setpoint and compute the torque.
        '''

        # PD law
        torque = self.controlLaw()

        # gravity compensation
        G = self.sim_data.qfrc_bias

        # Sum the torques.
        out_torque = torque + G

        self.sim_data.ctrl = out_torque
        
        return out_torque
    
    def controlLaw(self):
        return self.kp * self.joint_error() + self.kd * self.joint_vel_error()
    
    def set_gains(self, kp, kd):
        self.kp = kp
        if kd is None:
            # calc kd for critically damped 
            mass = kuka_subtree_mass(self.sim_model)
            print(kd, kp, mass)
            self.kd = 2 * np.sqrt(mass * kp)
        else:
            self.kd = kd
