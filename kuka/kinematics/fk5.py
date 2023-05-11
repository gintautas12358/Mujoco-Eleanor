
from math import sin, cos
import numpy as np

def fk5(qpos):
	tt1, tt2, tt3, tt4, tt5, tt6, tt7 = qpos
	x = -0.1845*(-sin(tt1)*sin(tt3) + cos(tt1)*cos(tt2)*cos(tt3))*sin(tt4) + 0.1845*sin(tt2)*cos(tt1)*cos(tt4) + 0.42*sin(tt2)*cos(tt1)
	y = -0.1845*(sin(tt1)*cos(tt2)*cos(tt3) + sin(tt3)*cos(tt1))*sin(tt4) + 0.1845*sin(tt1)*sin(tt2)*cos(tt4) + 0.42*sin(tt1)*sin(tt2)
	z = 0.1845*sin(tt2)*sin(tt4)*cos(tt3) + 0.1845*cos(tt2)*cos(tt4) + 0.42*cos(tt2) + 0.36
	pos = np.array([x, y, z])
	mat = np.zeros((3,3))
	mat[0,0] = ((-sin(tt1)*sin(tt3) + cos(tt1)*cos(tt2)*cos(tt3))*cos(tt4) + sin(tt2)*sin(tt4)*cos(tt1))*cos(tt5) + (-sin(tt1)*cos(tt3) - sin(tt3)*cos(tt1)*cos(tt2))*sin(tt5)
	mat[0,1] = -((-sin(tt1)*sin(tt3) + cos(tt1)*cos(tt2)*cos(tt3))*cos(tt4) + sin(tt2)*sin(tt4)*cos(tt1))*sin(tt5) + (-sin(tt1)*cos(tt3) - sin(tt3)*cos(tt1)*cos(tt2))*cos(tt5)
	mat[0,2] = -(-sin(tt1)*sin(tt3) + cos(tt1)*cos(tt2)*cos(tt3))*sin(tt4) + sin(tt2)*cos(tt1)*cos(tt4)
	mat[1,0] = ((sin(tt1)*cos(tt2)*cos(tt3) + sin(tt3)*cos(tt1))*cos(tt4) + sin(tt1)*sin(tt2)*sin(tt4))*cos(tt5) + (-sin(tt1)*sin(tt3)*cos(tt2) + cos(tt1)*cos(tt3))*sin(tt5)
	mat[1,1] = -((sin(tt1)*cos(tt2)*cos(tt3) + sin(tt3)*cos(tt1))*cos(tt4) + sin(tt1)*sin(tt2)*sin(tt4))*sin(tt5) + (-sin(tt1)*sin(tt3)*cos(tt2) + cos(tt1)*cos(tt3))*cos(tt5)
	mat[1,2] = -(sin(tt1)*cos(tt2)*cos(tt3) + sin(tt3)*cos(tt1))*sin(tt4) + sin(tt1)*sin(tt2)*cos(tt4)
	mat[2,0] = (-sin(tt2)*cos(tt3)*cos(tt4) + sin(tt4)*cos(tt2))*cos(tt5) + sin(tt2)*sin(tt3)*sin(tt5)
	mat[2,1] = -(-sin(tt2)*cos(tt3)*cos(tt4) + sin(tt4)*cos(tt2))*sin(tt5) + sin(tt2)*sin(tt3)*cos(tt5)
	mat[2,2] = sin(tt2)*sin(tt4)*cos(tt3) + cos(tt2)*cos(tt4)
	return pos, mat