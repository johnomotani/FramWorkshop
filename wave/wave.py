import numpy as np
from matplotlib import pyplot as plt
import animatplot as amp

def simulate_wave(wave_speed, timestep_size, n_gridpoints):
    # Length of the string.
    L = 1.0

    # Total time for the simulation.
    total_time = 10.0

    print("Initialising arrays...")
    # Array with time for each step.
    n_t = int(total_time / timestep_size)
    time = np.array([i_t * timestep_size for i_t in range(n_t)])

    # Create equally-spaced grid of points.
    grid_spacing = L / n_gridpoints
    grid = np.array([i * grid_spacing for i in range(n_gridpoints)])

    # Array that will store values of the displacement d for all time points.
    d = np.zeros([n_t, n_gridpoints])

    # Array that will store values of the velocity v for all time points.
    v = np.zeros([n_t, n_gridpoints])

    # Set up initial profile of d and v
    # This function produces a peak at L=0.25
    def initial_d(x):
        return np.sin(np.pi * (x / L + 0.25))**16

    d[0,:] = initial_d(grid)

    # v is chosen to make a right-moving disturbance at the initial time.
    initial_v_grid = (grid + wave_speed * 0.5 * timestep_size) % L
    v[0,:] = -16 * wave_speed * np.pi / L * np.cos(np.pi * (initial_v_grid / L + 0.25)) * np.sin(np.pi * (initial_v_grid / L + 0.25))**15

    #print(f"timestep_size={timestep_size}, grid_spacing={grid_spacing}, CFL-limit={grid_spacing/wave_speed}")

    # Step through all time points
    print("Start time loop")
    for i_t in range(0, n_t - 1):
        for i in range(0, n_gridpoints):
            # v is changed by forces from the neighbouring grid points
            if i == 0:
                # Periodicity means 'left' point is at the other end of the grid.
                displacement_left = d[i_t,-1] - d[i_t,i]
            else:
                displacement_left = d[i_t,i-1] - d[i_t,i]
            if i == n_gridpoints - 1:
                # Periodicity means 'right' point is at the other end of the grid.
                displacement_right = d[i_t,0] - d[i_t,i]
            else:
                displacement_right = d[i_t,i+1] - d[i_t,i]
            v[i_t+1,i] = v[i_t,i] + wave_speed**2 / grid_spacing**2 * (displacement_left + displacement_right) * timestep_size

        for i in range(0, n_gridpoints):
            # d is moved by the updated v
            d[i_t+1,i] = d[i_t,i] + v[i_t+1,i] * timestep_size

    print("Calculate exact solution")
    exact_d = initial_d(grid[None,:] - wave_speed * time[:,None])

    return time, grid, d, v, exact_d

def animate_results(time, grid, d, timesteps_per_frame):
    print("Animate results.")

    fig = plt.figure(figsize=(8,6))
    blocks = [amp.blocks.Line(grid, d[::timesteps_per_frame,:], label="d")]
    d_max = np.max(d[::timesteps_per_frame,:])
    extra_space = 0.1 * d_max
    d_max += extra_space
    d_min = np.min(d[::timesteps_per_frame,:]) - extra_space
    plt.ylim(d_min, d_max)
    timeline = amp.Timeline(time[::timesteps_per_frame], fps=60)
    anim = amp.Animation(blocks, timeline, fig=fig)
    plt.xlabel("x")
    plt.ylabel("d")
    anim.controls()
    plt.show()
    return

def animate_results_with_error(time, grid, d, exact_d, timesteps_per_frame):
    print("Animate results, showing error.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))

    ax1.set_title("d")
    ax2.set_title("error = d - exact_d")

    blocks = [amp.blocks.Line(grid, d[::timesteps_per_frame,:], label="d", ax=ax1)]
    blocks.append(amp.blocks.Line(grid, exact_d[::timesteps_per_frame,:], label="exact_d", ax=ax1))

    error = d[::timesteps_per_frame,:] - exact_d[::timesteps_per_frame,:]
    blocks.append(amp.blocks.Line(grid, error, label="error", ax=ax2))

    d_max = np.max(d[::timesteps_per_frame,:])
    extra_space = 0.1 * d_max
    d_max += extra_space
    d_min = np.min(d[::timesteps_per_frame,:]) - extra_space
    ax1.set_ylim(d_min, d_max)

    error_max = 1.1 * np.max(np.abs(error))
    ax2.set_ylim(-error_max, error_max)

    plt.legend(loc="lower right")
    timeline = amp.Timeline(time[::timesteps_per_frame], fps=60)
    anim = amp.Animation(blocks, timeline)
    plt.xlabel("x")
    plt.ylabel("d")
    anim.controls()
    plt.show()
    return

wave_speed = 1.0
timestep_size = 0.01
n_gridpoints = 20
timesteps_per_frame = 1

# Run the simulation
time, grid, d, v, exact_d = simulate_wave(wave_speed, timestep_size, n_gridpoints)

# Make a move of the results
animate_results(time, grid, d, timesteps_per_frame)
#animate_results_with_error(time, grid, d, exact_d, timesteps_per_frame)
