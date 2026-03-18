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
        v_grid = dataset["coords/vpa/grid"][:]
        pressure = dataset["dynamic_data/pressure"][:,0,0,:]
        if "f" in dataset["dynamic_data"]:
            f = dataset["dynamic_data/f"][:,0,0,:,0,:]
        else:
            f = None

    return time, grid, v_grid, pressure, f

def animate_results(time, grid, vgrid, pressure, f, timesteps_per_frame):
    print("Animate moment_kinetics results.")

    # Hard to see anything in f, so subtract the backround, non-wave-y, part.
    f_without_background = f[::timesteps_per_frame,:,:] - np.mean(f[::timesteps_per_frame,:,:], axis=(0, 1))
    flim = np.max(np.abs(f_without_background))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    blocks = [amp.blocks.Line(grid, pressure[::timesteps_per_frame,:], label="pressure", ax=ax1)]
    p_max = np.max(pressure[::timesteps_per_frame,:])
    p_min = np.min(pressure[::timesteps_per_frame,:])
    extra_space = 0.1 * (p_max - p_min)
    p_max += extra_space
    p_min -= extra_space
    ax1.set_ylim(p_min, p_max)
    ax1.set_xlabel("x")
    ax1.set_ylabel("pressure")
    blocks.append(amp.blocks.Pcolormesh(v_grid, grid, f_without_background, shading="gouraud", vmin=-flim, vmax=flim, cmap="PuOr", ax=ax2))
    ax2.set_xlabel("v")
    ax2.set_ylabel("x")
    timeline = amp.Timeline(time[::timesteps_per_frame], fps=60)
    anim = amp.Animation(blocks, timeline, fig=fig)
    anim.controls()
    plt.show()
    return

def animate_results_p(time, grid, pressure, timesteps_per_frame):
    print("Animate moment_kinetics results.")

    fig = plt.figure(figsize=(8,6))
    blocks = [amp.blocks.Line(grid, pressure[::timesteps_per_frame,:], label="pressure")]
    p_max = np.max(pressure[::timesteps_per_frame,:])
    p_min = np.min(pressure[::timesteps_per_frame,:])
    extra_space = 0.1 * (p_max - p_min)
    p_max += extra_space
    p_min -= extra_space
    plt.ylim(p_min, p_max)
    plt.xlabel("x")
    plt.ylabel("pressure")
    timeline = amp.Timeline(time[::timesteps_per_frame], fps=60)
    anim = amp.Animation(blocks, timeline, fig=fig)
    anim.controls()
    plt.show()
    return

def animate_results_f(time, grid, v_grid, f, timesteps_per_frame):
    print("Animate moment_kinetics results.")

    # Hard to see anything in f, so subtract the backround, non-wave-y, part.
    f_without_background = f[::timesteps_per_frame,:,:] - np.mean(f[::timesteps_per_frame,:,:], axis=(0, 1))
    flim = np.max(np.abs(f_without_background))

    fig = plt.figure(figsize=(8,6))
    blocks = [amp.blocks.Pcolormesh(v_grid, grid, f_without_background, shading="gouraud", vmin=-flim, vmax=flim, cmap="PuOr")]
    timeline = amp.Timeline(time[::timesteps_per_frame], fps=60)
    anim = amp.Animation(blocks, timeline, fig=fig)
    plt.xlabel("v")
    plt.ylabel("x")
    anim.controls()
    plt.show()
    return

time, grid, v_grid, pressure, f = load_moment_kinetics_output(sys.argv[1])
if f is not None:
    animate_results(time, grid, v_grid, pressure, f, 1)
else:
    animate_results_p(time, grid, pressure, 1)
