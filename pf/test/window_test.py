"""
test for a window
"""

import numpy as np
from pf.utils.window import Window, WINDOW_XMIN, WINDOW_XMAX, WINDOW_YMIN, WINDOW_YMAX

window = Window()

for i in range(3):
    data = np.random.random()
    x = np.random.uniform(low=WINDOW_XMIN, high=WINDOW_XMAX, size=1000)
    y = np.random.uniform(low=WINDOW_YMIN, high=WINDOW_YMAX, size=1000)

    window.plot(x=x, y=y, id=i, _type="")
    window.update_canvas()
