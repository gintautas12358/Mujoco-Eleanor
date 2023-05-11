import numpy as np

from .ElasticCartesianPDController import ElasticCartesianPDController
from .JointPDController import JointPDController


class NullSpaceElasticCartesianPDController(ElasticCartesianPDController, JointPDController):
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

        super(NullSpaceElasticCartesianPDController, self).__init__(sim_model, sim_data, kp, kd, site_name)
    
    def controlLaw(self):
        return ElasticCartesianPDController.controlLaw(self) + self.null_space_proj_m() @ JointPDController.controlLaw(self)
    

    def null_space_proj_m(self):
        J = self.Jac()

        # p = 1 - J^T * (J^+)^T
        projection_matrix = np.eye(7) - J.T @ self.left_pseudo_Jac(eps=1e-6).T

        return projection_matrix
    
