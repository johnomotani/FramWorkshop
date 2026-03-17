import h5py

def load_moment_kinetics_output():
    data_path = "../../moment_kinetics-test3/runs/sound-wave/sound-wave.moments.h5"
    with h5py.Dataset(data_path, "r") as dataset:
        time = dataset["dynamic_data/time"][:]
        grid = dataset["coords/z/grid"][:]
        pressure = dataset["dynamic_data/pressure"][:,0,0,:]

    return time, grid, pressure

def animate_results(time, grid, pressure, fps=60):
    timestep_size = time[1] - time[0]
    steps_per_output = max(int(1 / (timestep_size * fps)), 1)
    blocks = [amp.blocks.Line(grid, pressure[::steps_per_output,:], label="pressure")]
    d_max = np.max(pressure[0,:])
    extra_space = 0.1 * d_max
    d_max += extra_space
    d_min = np.min(pressure[0,:]) - extra_space
    plt.ylim(d_min, d_max)
    plt.legend(loc="lower right")
    timeline = amp.Timeline(time[::steps_per_output], fps=fps)
    anim = amp.Animation(blocks, timeline)
    anim.controls()
    plt.show()
    return

def load_and_plot():
    time, grid, pressure = load_moment_kinetics_output()
    animate_results(time, grid, pressure)
