"""
test for checking the animation with a vehicle
"""

import numpy as np
from pf.utils.window import Window
from pf.utils.vehicle import Vehicle

window = Window()

init_X = np.zeros(shape=[4, 1])

vehicle = Vehicle(init_X=init_X)
print("Before move: {}".format(vehicle.get_state()))

for i in range(100):
    vehicle.auto_pilot()
    print("Iter {} {}".format(i, vehicle.get_state()))
    x, y = vehicle.get_position()
    window.plot(x=x, y=y, id=i, _type="vehicle")
    window.update_canvas()
