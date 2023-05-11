import numpy as np

import sys
import os
sys.path.append(os.getcwd())


class PathController:
    '''
    A base for all joint based controllers
    '''

    def __init__(self, controller, planner, vmax=1, amax=1, jmax=1) -> None:
        super().__init__()

        self.controller = controller
        self.planner = planner
        self.vmax = vmax
        self.amax = amax
        self.jmax = jmax
        
        self.targetPose = None
        self.path = None

    def initStartPose(self, pose):
        self.targetPose = pose
        self.path = self.planner.genPath(0, 1, self.targetPose, self.targetPose)

    def newPathTo(self, target, tt):
        if self.targetPose is None:
            raise Exception("no current pose for the robot given")
        
        duration = self.calcDuration(self.targetPose, target, self.vmax, self.amax, self.jmax)
        self.path = self.planner.genPath(tt, tt+duration, self.targetPose, target)
        self.targetPose = target

    def calcDuration(self, x0, x1, vmax, amax, jmax):
        x = x1 - x0

        times = []
        for i in range(x.size):
            tv = x[i] / vmax
            ta = 2 * pow(x[i]/amax, 1.0/2)
            tj = 2 * pow(4*x[i]/jmax, 1.0/3)
            times += [tv, ta, tj]

        print(times)
        return max(times)


    def set_time(self, t):
        '''
        Set the setpoint.
        '''
        if self.path is None:
            raise Exception("no path was generated")

        pose = self.path.atTime(t)
        self.controller.set_action(pose)

    def get_torque(self):
        return self.controller.get_torque()
        

    def error(self):
        err = self.targetPose - self.controller.fk()
        for i in range(3,6):
            err[i] = err[i] - 2*np.pi if err[i] > np.pi else err[i]
        return err
    
