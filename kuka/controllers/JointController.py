import numpy as np

import sys
import os
sys.path.append(os.getcwd())

from .MujocoPDController import MujocoPDController


class JointController(MujocoPDController):
    '''
    A base for all joint based controllers
    '''

    def __init__(self,
                    sim_model, sim_data,
                    kp=3.0,
                    kd=None,
                    ):

        super(JointController, self).__init__(sim_model, sim_data, kp, kd)

        # Initialize setpoint.
        self.sim_qpos_set = sim_data.qpos[self.sim_qpos_idx].copy()
        self.sim_qvel_set = np.zeros(len(self.sim_qvel_idx))

    def joint_error(self):
        return self.sim_qpos_set - self.sim_data.qpos[self.sim_qpos_idx]

    def joint_vel_error(self):
        return self.sim_qvel_set - self.sim_data.qvel[self.sim_qvel_idx]
    
    def error(self):
        return self.joint_error()