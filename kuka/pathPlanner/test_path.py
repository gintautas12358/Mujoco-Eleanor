import numpy as np
from CubicPathPlanner import CubicPolyPathGenerator, SingleCubicPolyPathGenerator

x0 = 0
x1 = 10

t0 = 0
t1 = 10

polyGen = SingleCubicPolyPathGenerator()
path = polyGen.genPath(t0, t1, x0, x1)

for t in range(t0, t1):
    print(path.atTime(t))




x0 = np.array([0,0,0, 0,0,0])
x1 = np.array([10,10,10, 8,10,10])

t0 = 0
t1 = 10

polyGen = CubicPolyPathGenerator(x0.size)
path = polyGen.genPath(t0, t1, x0, x1)

for t in range(t0, t1):
    # print(path)
    print(path.atTime(t))
