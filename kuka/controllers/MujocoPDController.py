import abc
import mujoco

import sys
import os
sys.path.append(os.getcwd())

from .Controller import Controller


class MujocoPDController(Controller, abc.ABC):
    '''
    An abstract base class for low level controllers.
    '''

    # def __init__(self, sim_model, sim_data):
    #     super(MujocoController, self).__init__(sim_model, sim_data)

    def __init__(self, sim_model, sim_data, kp=3, kd=None):
        self.sim_model, self.sim_data = sim_model, sim_data

        # Get the position, velocity, and actuator indices for the model.
        self.init_indices()

        # PD parameters
        self.set_gains(kp, kd)

        mujoco.mj_forward(sim_model, sim_data)

        
    def init_indices(self): 
        self.sim_qpos_idx = range(self.sim_model.nq)
        self.sim_qvel_idx = range(self.sim_model.nv)
        self.sim_actuators_idx = range(self.sim_model.nu)
        self.sim_joint_idx = range(self.sim_model.nu)

    @abc.abstractmethod
    def set_gains(self, kp, kd):
        pass

    @abc.abstractmethod
    def controlLaw(self):
        pass
    