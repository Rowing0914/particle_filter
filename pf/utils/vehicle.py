import numpy as np

DT = 0.1  # time tick [s]
R = np.diag([2.0, np.deg2rad(40.0)]) ** 2  # input error

class Vehicle(object):
    """ Vehicle class """

    def __init__(self, init_X=np.zeros(shape=[4, 1])):
        self._X = init_X

    def auto_pilot(self):
        """ Follow the predefined trajectory """
        u = self._calc_input()
        F, B = self._make_F_B()
        self._X = self._motion_model(X=np.asarray(self._get_state()), u=u, F=F, B=B)

    def move_particles(self, X):
        """ Move particles """
        u = self._calc_input()
        u = self._make_input_noisy(u=u)
        u = np.tile(u, reps=[1, X.shape[1]])  # 2 x num_particle
        F, B = self._make_F_B()  # 4 x 4, 2 x 4
        _X = self._motion_model(X=X, u=u, F=F, B=B)  # 4 x num_particle
        return _X

    def _make_F_B(self):
        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])  # 4 x 4

        B = np.array([[DT * np.cos(self._X[2, 0]), 0],
                      [DT * np.sin(self._X[2, 0]), 0],
                      [0.0, DT],
                      [1.0, 0.0]])  # 2 x 4
        return F, B

    def observe(self):
        """ observe the position: x, y """
        return np.array([self._X[:-2, 0]]).T

    def _calc_input(self):
        """ temporarily we assume the fixed trajectory """
        v = 1.0  # [m/s]
        yaw_rate = 0.1  # [rad/s]
        u = np.array([[v, yaw_rate]]).T  # 2 x 1
        return u

    def _make_input_noisy(self, u):
        ud1 = u[0, 0] + np.random.randn() * R[0, 0] ** 0.5
        ud2 = u[1, 0] + np.random.randn() * R[1, 1] ** 0.5
        ud = np.array([[ud1, ud2]]).T  # 2 x 1
        return ud

    def _motion_model(self, X, u, F, B):
        """ Motion Model """
        return F.dot(X) + B.dot(u)

    def _get_state(self):
        """ to access the state of the vehicle """
        return self._X

    def get_state(self):
        """ returns the string which describes the state of the vehicle """
        X = self._get_state()
        return "x: {:.4f}, y: {:.4f}, yaw: {:.4f}, v: {:.4f}".format(X[0, 0], X[1, 0], X[2, 0], X[3, 0])

    def get_position(self):
        """ returns x, y coordinate of the vehicle """
        return self._X[0, 0], self._X[1, 0]
