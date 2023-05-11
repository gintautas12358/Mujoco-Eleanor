
from math import sin, cos
import numpy as np

def fk4cm(qpos):
	tt1, tt2, tt3, tt4, tt5, tt6, tt7 = qpos
	x = -0.067*(-sin(tt1)*sin(tt3) + cos(tt1)*cos(tt2)*cos(tt3))*sin(tt4) + 0.034*sin(tt1)*cos(tt3) + 0.067*sin(tt2)*cos(tt1)*cos(tt4) + 0.42*sin(tt2)*cos(tt1) + 0.034*sin(tt3)*cos(tt1)*cos(tt2)
	y = -0.067*(sin(tt1)*cos(tt2)*cos(tt3) + sin(tt3)*cos(tt1))*sin(tt4) + 0.067*sin(tt1)*sin(tt2)*cos(tt4) + 0.42*sin(tt1)*sin(tt2) + 0.034*sin(tt1)*sin(tt3)*cos(tt2) - 0.034*cos(tt1)*cos(tt3)
	z = -0.034*sin(tt2)*sin(tt3) + 0.067*sin(tt2)*sin(tt4)*cos(tt3) + 0.067*cos(tt2)*cos(tt4) + 0.42*cos(tt2) + 0.36
	pos = np.array([x, y, z])
	mat = np.zeros((3,3))
	mat[0,0] = -(-sin(tt1)*sin(tt3) + cos(tt1)*cos(tt2)*cos(tt3))*sin(tt4) + sin(tt2)*cos(tt1)*cos(tt4)
	mat[0,1] = -(-sin(tt1)*sin(tt3) + cos(tt1)*cos(tt2)*cos(tt3))*cos(tt4) - sin(tt2)*sin(tt4)*cos(tt1)
	mat[0,2] = sin(tt1)*cos(tt3) + sin(tt3)*cos(tt1)*cos(tt2)
	mat[1,0] = -(sin(tt1)*cos(tt2)*cos(tt3) + sin(tt3)*cos(tt1))*sin(tt4) + sin(tt1)*sin(tt2)*cos(tt4)
	mat[1,1] = -(sin(tt1)*cos(tt2)*cos(tt3) + sin(tt3)*cos(tt1))*cos(tt4) - sin(tt1)*sin(tt2)*sin(tt4)
	mat[1,2] = sin(tt1)*sin(tt3)*cos(tt2) - cos(tt1)*cos(tt3)
	mat[2,0] = sin(tt2)*sin(tt4)*cos(tt3) + cos(tt2)*cos(tt4)
	mat[2,1] = sin(tt2)*cos(tt3)*cos(tt4) - sin(tt4)*cos(tt2)
	mat[2,2] = -sin(tt2)*sin(tt3)
	return pos, mat