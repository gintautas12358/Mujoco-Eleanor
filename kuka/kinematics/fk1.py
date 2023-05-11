
from math import sin, cos
import numpy as np

def fk1(qpos):
	tt1, tt2, tt3, tt4, tt5, tt6, tt7 = qpos
	x = 0
	y = 0
	z = 0.157500000000000
	pos = np.array([x, y, z])
	mat = np.zeros((3,3))
	mat[0,0] = cos(tt1)
	mat[0,1] = -sin(tt1)
	mat[0,2] = 0
	mat[1,0] = sin(tt1)
	mat[1,1] = cos(tt1)
	mat[1,2] = 0
	mat[2,0] = 0
	mat[2,1] = 0
	mat[2,2] = 1
	return pos, mat