import matplotlib.pyplot as plt
import numpy as np

# define WINDOW_SIZE
WINDOW_XMIN = -0.1
WINDOW_XMAX = 10.0
WINDOW_YMIN = -0.1
WINDOW_YMAX = 5.0
PAUSE_SEC = 0.0001


class Window(object):
    def __init__(self):
        plt.ion()  # activate the tkinter mode

    def plot(self, x, y, id=None, _type=""):
        if _type == "vehicle":
            plt.scatter(x, y, marker="o", c="red", label="vehicle")
            plt.title("Iter: {}, x: {:.3f}, y: {:.3f}".format(id, x, y))
        elif _type == "estimate":
            plt.scatter(x, y, marker="o", c="blue", label="estimate")
            plt.title("Iter: {}, x: {:.3f}, y: {:.3f}".format(id, x, y))
        else:
            plt.scatter(x, y, marker="*", s=2.0, c="green", label="particles")

        plt.xlim(WINDOW_XMIN, WINDOW_XMAX)
        plt.ylim(WINDOW_YMIN, WINDOW_YMAX)

    def update_canvas(self):
        plt.legend()
        plt.draw()
        plt.pause(PAUSE_SEC)
        plt.clf()
