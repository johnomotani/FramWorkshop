import sys
import numpy as np
import h5py
from matplotlib import pyplot as plt
import animatplot as amp

def load_moment_kinetics_output(data_path):
    print("Load moment_kinetics results.")

    with h5py.File(data_path, "r") as dataset:
        time = dataset["dynamic_data/time"][:]
        grid = dataset["coords/z/grid"][:]
        pressure = dataset["dynamic_data/pressure"][:,0,0,:]

    return time, grid, pressure

def animate_results(time, grid, pressure, timesteps_per_frame):
    print("Animate moment_kinetics results.")

    fig = plt.figure(figsize=(8,6))
    blocks = [amp.blocks.Line(grid, pressure[::timesteps_per_frame,:], label="pressure")]
    p_max = np.max(pressure[::timesteps_per_frame,:])
    p_min = np.min(pressure[::timesteps_per_frame,:])
    extra_space = 0.1 * (p_max - p_min)
    p_max += extra_space
    p_min -= extra_space
    plt.ylim(p_min, p_max)
    timeline = amp.Timeline(time[::timesteps_per_frame], fps=60)
    anim = amp.Animation(blocks, timeline, fig=fig)
    plt.xlabel("x")
    plt.ylabel("pressure")
    anim.controls()
    plt.show()
    return

time, grid, pressure = load_moment_kinetics_output(sys.argv[1])
animate_results(time, grid, pressure, 1)
