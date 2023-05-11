import numpy as np

class Polynomial:
        
    def T0(self, t):
        #[1, t1, t1**2, t1**3, t1**4, t1**5]
        return np.array([pow(t, x) for x in range(6)])

    def T1(self, t):
        #[0 , 1, 2*t1, 3*t1**2, 4*t1**3, 5*t1**4]
        return np.array([(x+1)*pow(t, x) if x > -1 else 0 for x in range(-1,5)])

    def T2(self, t):
        #[0 , 0, 2, 6*t1, 12*t1**2, 20*t1**3]
        return np.array([(x+1)*(x+2)*pow(t, x) if x > -1 else 0 for x in range(-2,4)])


class SingleCubicPolyPathGenerator(Polynomial):

    def __init__(self) -> None:
        super().__init__()



    def T(self, t0, t1):
        return np.array([self.T0(t0),
                        self.T0(t1),
                        self.T1(t0),
                        self.T1(t1),
                        self.T2(t0),
                        self.T2(t1)])

    def genPath(self, t0, t1, x0, x1, dx0=0, dx1=0, ddx0=0, ddx1=0):
        x = np.array([[x0], [x1], [dx0], [dx1], [ddx0], [ddx1]])

        param = np.linalg.inv(self.T(t0, t1)) @ x
        return Path(param, t0, t1)


class Path(Polynomial):

    def __init__(self, param, t0, t1) -> None:
        self.param = param
        self.t0 = t0
        self.t1 = t1

    def atTime(self, t):
        if t < self.t0:
            t = self.t0
        if t > self.t1:
            t = self.t1

        T = np.array([self.T0(t)])
        x = T @ self.param
        return x[0]
    

class CubicPolyPathGenerator(SingleCubicPolyPathGenerator):

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def genPath(self, t0, t1, x0, x1, dx0=None, dx1=None, ddx0=None, ddx1=None):
        if dx0 is None:
            dx0 = np.zeros(self.dim)
        if dx1 is None:
            dx1 = np.zeros(self.dim)
        if ddx0 is None:
            ddx0 = np.zeros(self.dim)
        if ddx1 is None:
            ddx1 = np.zeros(self.dim)
        
        params = None
        for i in range(x0.size):
            x = np.array([[x0[i]], [x1[i]], [dx0[i]], [dx1[i]], [ddx0[i]], [ddx1[i]]])
            param = np.linalg.inv(self.T(t0, t1)) @ x

            if params is None:
                params = param
            else:
                params = np.concatenate((params, param), axis=1)

        return Path(params, t0, t1)
    
