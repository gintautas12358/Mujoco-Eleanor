
from math import sin, cos
import numpy as np

def fk3cm(qpos):
	tt1, tt2, tt3, tt4, tt5, tt6, tt7 = qpos
	x = -0.03*sin(tt1)*cos(tt3) + 0.3345*sin(tt2)*cos(tt1) - 0.03*sin(tt3)*cos(tt1)*cos(tt2)
	y = 0.3345*sin(tt1)*sin(tt2) - 0.03*sin(tt1)*sin(tt3)*cos(tt2) + 0.03*cos(tt1)*cos(tt3)
	z = 0.03*sin(tt2)*sin(tt3) + 0.3345*cos(tt2) + 0.36
	pos = np.array([x, y, z])
	mat = np.zeros((3,3))
	mat[0,0] = -sin(tt1)*cos(tt3) - sin(tt3)*cos(tt1)*cos(tt2)
	mat[0,1] = sin(tt1)*sin(tt3) - cos(tt1)*cos(tt2)*cos(tt3)
	mat[0,2] = sin(tt2)*cos(tt1)
	mat[1,0] = -sin(tt1)*sin(tt3)*cos(tt2) + cos(tt1)*cos(tt3)
	mat[1,1] = -sin(tt1)*cos(tt2)*cos(tt3) - sin(tt3)*cos(tt1)
	mat[1,2] = sin(tt1)*sin(tt2)
	mat[2,0] = sin(tt2)*sin(tt3)
	mat[2,1] = sin(tt2)*cos(tt3)
	mat[2,2] = cos(tt2)
	return pos, mat