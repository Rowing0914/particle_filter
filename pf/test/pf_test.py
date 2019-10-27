"""
test for particle filter
"""

import numpy as np
from pf.utils.window import Window
from pf.utils.vehicle import Vehicle
from pf.utils.particle_filter import PF

NUM_PARTICLES = 100

window = Window()

init_X = np.zeros(shape=[4, 1])
pf = PF(num_particle=NUM_PARTICLES,
        vehicle=Vehicle(init_X=init_X))

for i in range(100):
    x, y = pf.get_estimate_position()
    window.plot(x=x, y=y, id=i, _type="estimate")
    x, y = pf.get_vehicle_position()
    window.plot(x=x, y=y, id=i, _type="vehicle")
    x, y = pf.get_particles_position()
    window.plot(x=x, y=y, id=i)
    window.update_canvas()
    pf.predict()
    pf.update()
    pf.resample()
