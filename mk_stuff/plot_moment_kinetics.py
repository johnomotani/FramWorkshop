import animatplot as amp
import h5py
from matplotlib import pyplot as plt
import numpy as np
import sys

def load_moment_kinetics_output(data_path):
    with h5py.File(data_path, "r") as dataset:
        time = dataset["dynamic_data/time"][:]
        grid = dataset["coords/z/grid"][:]
        pressure = dataset["dynamic_data/pressure"][:,0,0,:]

    return time, grid, pressure

def animate_results(time, grid, pressure, fps=60):
    timestep_size = time[1] - time[0]
    steps_per_output = max(int(1 / (timestep_size * fps)), 1)
    blocks = [amp.blocks.Line(grid, pressure[::steps_per_output,:], label="pressure")]
    plt.xlabel("x")
    plt.ylabel("pressure")
    d_max = np.max(pressure)
    d_min = np.min(pressure)
    extra_space = 0.1 * (d_max - d_min)
    d_max += extra_space
    d_min -= extra_space
    plt.ylim(d_min, d_max)
    plt.legend(loc="lower right")
    timeline = amp.Timeline(time[::steps_per_output], fps=fps)
    anim = amp.Animation(blocks, timeline)
    anim.controls()
    plt.show()
    return

time, grid, pressure = load_moment_kinetics_output(sys.argv[1])
animate_results(time, grid, pressure)
