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

def animate_results(time1, grid1, vgrid1, pressure1, f1, time2, grid2, vgrid2, pressure2, f2, timesteps_per_frame):
    print("Animate moment_kinetics results.")

    if not np.max(np.abs(time1 - time2)) < 1e-10:
        raise ValueError("time1 and time2 are not the same")

    # Hard to see anything in f, so subtract the backround, non-wave-y, part.
    f_without_background1 = f1[::timesteps_per_frame,:,:] - np.mean(f1[::timesteps_per_frame,:,:], axis=(0, 1))
    f_without_background2 = f2[::timesteps_per_frame,:,:] - np.mean(f2[::timesteps_per_frame,:,:], axis=(0, 1))
    flim = max(np.max(np.abs(f_without_background1)), np.max(np.abs(f_without_background2)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
    blocks = [amp.blocks.Line(grid1, pressure1[::timesteps_per_frame,:], label="pressure1", ax=ax1),
              amp.blocks.Line(grid2, pressure2[::timesteps_per_frame,:], label="pressure2", ax=ax1)]
    p_max = max(np.max(pressure1[::timesteps_per_frame,:]), np.max(pressure2[::timesteps_per_frame,:]))
    p_min = min(np.min(pressure1[::timesteps_per_frame,:]), np.min(pressure2[::timesteps_per_frame,:]))
    extra_space = 0.1 * (p_max - p_min)
    p_max += extra_space
    p_min -= extra_space
    ax1.set_ylim(p_min, p_max)
    ax1.set_xlabel("x")
    ax1.set_ylabel("pressure")
    ax1.legend()
    blocks.append(amp.blocks.Pcolormesh(v_grid1, grid1, f_without_background1, shading="gouraud", vmin=-flim, vmax=flim, cmap="PuOr", ax=ax2))
    ax2.set_xlabel("v")
    ax2.set_ylabel("x")
    ax2.set_title("f1")
    blocks.append(amp.blocks.Pcolormesh(v_grid2, grid2, f_without_background2, shading="gouraud", vmin=-flim, vmax=flim, cmap="PuOr", ax=ax3))
    ax3.set_xlabel("v")
    ax3.set_ylabel("x")
    ax3.set_title("f2")
    timeline = amp.Timeline(time1[::timesteps_per_frame], fps=60)
    anim = amp.Animation(blocks, timeline, fig=fig)
    anim.controls()
    plt.show()
    return

time1, grid1, v_grid1, pressure1, f1 = load_moment_kinetics_output(sys.argv[1])
time2, grid2, v_grid2, pressure2, f2 = load_moment_kinetics_output(sys.argv[2])
animate_results(time1, grid1, v_grid1, pressure1, f1, time2, grid2, v_grid2, pressure2, f2, 1)
