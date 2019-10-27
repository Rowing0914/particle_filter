"""
test for moving around a vehicle
"""

import numpy as np
from pf.utils.vehicle import Vehicle

init_X = np.zeros(shape=[4, 1])

vehicle = Vehicle(init_X=init_X)
print("Before move: {}".format(vehicle.get_state()))

for i in range(100):
    vehicle.auto_pilot()
    print("Iter {} {}".format(i, vehicle.get_state()))
