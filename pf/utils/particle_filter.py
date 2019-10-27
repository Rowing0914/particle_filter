import numpy as np
from pf.utils.window import WINDOW_XMIN, WINDOW_XMAX, WINDOW_YMIN, WINDOW_YMAX

SENSOR_NOISE = 0.05


def gaussian_prob(mu, sigma, x):
    # calculate the probability of x for 1-dim Gaussian with mean mu and var sigma
    return np.exp(-((mu - x) ** 2) / (sigma ** 2) / 2.) / np.sqrt(2. * np.pi * (sigma ** 2))


class PF(object):
    """ Particle Filter class """

    def __init__(self, num_particle, vehicle):
        self._particles = list()
        self._num_particle = num_particle
        self._vehicle = vehicle
        self._particles = self._init_particles()
        self._weight = self._init_weights()

        self.X_true = self._vehicle.get_state()
        self.X_est = self.estimate_state()

    def _init_particles(self):
        """ Init particles in the predefined 2D space equally """
        # sample particles
        _x = np.random.uniform(low=WINDOW_XMIN, high=WINDOW_XMAX, size=self._num_particle)
        _y = np.random.uniform(low=WINDOW_YMIN, high=WINDOW_YMAX, size=self._num_particle)
        _yaw = np.zeros(shape=(self._num_particle,))
        _v = np.zeros(shape=(self._num_particle,))
        return np.vstack([_x, _y, _yaw, _v])  # 4 x num_particle

    def _init_weights(self):
        """ uniformly initialise weights for particles """
        return np.ones(shape=(self._num_particle, 1)) / self._num_particle  # num_particle x 1

    def get_vehicle_position(self):
        """ returns x, y coordinate of the vehicle """
        return self._vehicle.get_position()

    def get_particles_position(self):
        """ returns x, y coordinate of each particle """
        return self._particles[0, :], self._particles[1, :]

    def get_estimate_position(self):
        """ returns x, y coordinate of estimate position """
        return self.X_est[0, 0], self.X_est[1, 0]

    def predict(self):
        """ Move particles using the same motion model as Vehicle """
        # move vehicle along with the predefined trajectory by one step
        self._vehicle.auto_pilot()

        # estimate the vehicle's pose
        self.X_est = self.estimate_state()

    def estimate_state(self):
        """ Estimate the pose of the vehicle using survived particles """
        # return self._particles.dot(self._weight)  # 4 x 1
        return self._particles.mean(axis=-1).reshape(4, 1)  # 4 x 1

    def update(self):
        # move particles
        X = self._vehicle.move_particles(X=self._particles)

        # observation: x, y
        z = self._vehicle.observe()  # 2 x 1
        z = np.tile(z, reps=[1, self._num_particle])  # 2 x num_particle

        #  Calc Importance Weight
        _x, _y = X[0, :], X[1, :]
        _x_w = gaussian_prob(mu=_x, sigma=SENSOR_NOISE, x=z[0, :])
        _y_w = gaussian_prob(mu=_y, sigma=SENSOR_NOISE, x=z[1, :])
        weight = _x_w * _y_w + np.spacing(1)
        weight /= np.sum(weight)
        self._weight = np.reshape(weight, [self._num_particle, 1])  # num_particle x 1
        self._particles = X

    def resample(self):
        """ Low Variance Resampling
            Page 86, Table 4.4 in Probabilistic Robotics by S.Thrun et al.
        """
        r = np.random.uniform(low=0., high=self._num_particle ** -1)
        c = self._weight[0]
        i = 0
        X_t = list()
        for m in range(self._num_particle):
            u = r + (m - 1) * (self._num_particle ** -1)
            while u > c:
                i += 1
                c += self._weight[i]
            X_t.append(self._particles[:, i])

        self._particles = np.asarray(X_t).T