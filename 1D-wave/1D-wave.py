import numpy as np
import animatplot as amp
from matplotlib import pyplot as plt

"""
Simulate propagation of a wave in 1d, for example a string that is under tension.

Boundary condition is that there is no displacement at the end points of the grid.

Uses for loops for all operations.
"""
def simulate_wave_with_loops(wave_speed, timestep_size, n_gridpoints):
    L = 1.0
    total_time = 100.0

    n_t = int(total_time / timestep_size)
    time = np.fromiter((i_t * timestep_size for i_t in range(n_t)), float)

    grid_spacing = L / (n_gridpoints - 1)
    grid = np.fromiter((i * grid_spacing for i in range(n_gridpoints)), float)

    # Array that will store values of the displacement d for all time points.
    d = np.zeros([n_t, n_gridpoints])

    # Array that will store values of the velocity v for all time points.
    v = np.zeros([n_t, n_gridpoints])

    # Set up initial profile of d and v
    for i in range(n_gridpoints):
        # Add some small modifications to a bell curve to make sure the initial profile
        # is exactly zero at the boundaries.
        d[0,i] = np.exp(-(grid[i] - L/4)**2 * 16**2) - np.exp(-(0.0 - L/4)**2 * 16**2) - np.exp(-(L - L/4)**2 * 16**2) * grid[i] / L

        # v is chosen to make a right-moving disturbance at the initial time.
        v[0,i] = -wave_speed * (-2 * (grid[i] - L/4) * 16**2 * np.exp(-(grid[i] - L/4)**2 * 16**2) - np.exp(-(L - L/4)**2 * 16**2) / L)

    # Check that boundary conditions are exactly obeyed.
    d[0,0] = 0.0
    d[0,-1] = 0.0
    v[0,0] = 0.0
    v[0,-1] = 0.0

    # Step through all time points
    for i_t in range(0, n_t - 1):
        # Note we do not update the first and last grid points as these are fixed to
        # 0.0.
        for i in range(1,n_gridpoints-1):
            # v is changed by forces from the neighbouring grid points
            displacement_left = d[i_t,i-1] - d[i_t,i]
            displacement_right = d[i_t,i+1] - d[i_t,i]
            v[i_t+1,i] = v[i_t,i] + wave_speed**2 / grid_spacing**2 * (displacement_left + displacement_right) * timestep_size

        for i in range(1,n_gridpoints-1):
            # d is moved by the current v
            d[i_t+1,i] = d[i_t,i] + v[i_t+1,i] * timestep_size

    print(f"timestep_size={timestep_size}, CFL-limit={grid_spacing/wave_speed}")

    return time, grid, d, v

"""
Simulate propagation of a wave in 1d, for example a string that is under tension.

Boundary condition is that there is no displacement at the end points of the grid.

Uses numpy array operations for efficiency.
"""
def simulate_wave_with_array_operations(wave_speed, timestep_size, n_gridpoints):
    L = 1.0
    total_time = 100.0

    n_t = int(total_time / timestep_size)
    time = np.fromiter((i_t * timestep_size for i_t in range(n_t)), float)

    grid_spacing = L / (n_gridpoints - 1)
    grid = np.fromiter((i * grid_spacing for i in range(n_gridpoints)), float)

    # Array that will store values of the displacement d for all time points.
    d = np.zeros([n_t, n_gridpoints])

    # Array that will store values of the velocity v for all time points.
    v = np.zeros([n_t, n_gridpoints])

    # Set up initial profile of d and v
    #
    # Add some small modifications to a bell curve to make sure the initial profile
    # is exactly zero at the boundaries.
    d[0,:] = np.exp(-(grid - L/4)**2 * 16**2) - np.exp(-(0.0 - L/4)**2 * 16**2) - np.exp(-(L - L/4)**2 * 16**2) * grid / L

    # v is chosen to make a right-moving disturbance at the initial time.
    v[0,:] = -wave_speed * (-2 * (grid - L/4) * 16**2 * np.exp(-(grid - L/4)**2 * 16**2) - np.exp(-(L - L/4)**2 * 16**2) / L)

    # Check that boundary conditions are exactly obeyed.
    d[0,0] = 0.0
    d[0,-1] = 0.0
    v[0,0] = 0.0
    v[0,-1] = 0.0

    # Step through all time points
    for i_t in range(0, n_t - 1):
        #print(time[i_t])
        # Note we do not update the first and last grid points as these are fixed to
        # 0.0.

        # v is changed by forces from the neighbouring grid points
        displacement_left = d[i_t,:-2] - d[i_t,1:-1]
        displacement_right = d[i_t,2:] - d[i_t,1:-1]
        v[i_t+1,1:-1] = v[i_t,1:-1] + wave_speed**2 / grid_spacing**2 * (displacement_left + displacement_right) * timestep_size

        # d is moved by the updated v
        d[i_t+1,1:-1] = d[i_t,1:-1] + v[i_t+1,1:-1] * timestep_size

    print(f"timestep_size={timestep_size}, CFL-limit={grid_spacing/wave_speed}")

    return time, grid, d, v

def simulate_wave_with_array_operations_periodic(wave_speed, timestep_size, n_gridpoints):
    L = 1.0
    total_time = 100.0

    n_t = int(total_time / timestep_size)
    time = np.fromiter((i_t * timestep_size for i_t in range(n_t)), float)

    grid_spacing = L / n_gridpoints
    grid = np.fromiter((i * grid_spacing for i in range(n_gridpoints)), float)

    # Array that will store values of the displacement d for all time points.
    d = np.zeros([n_t, n_gridpoints])

    # Array that will store values of the velocity v for all time points.
    v = np.zeros([n_t, n_gridpoints])

    # Set up initial profile of d and v
    #
    # Add some small modifications to a bell curve to make sure the initial profile
    # is exactly zero at the boundaries.
    peak_sharpness_setting = 16
    def initial_d(x):
        #return np.exp(-(x - L/4)**2 * peak_sharpness_setting**2) - np.exp(-(0.0 - L/4)**2 * peak_sharpness_setting**2) - np.exp(-(L - L/4)**2 * peak_sharpness_setting**2) * x / L
        #return np.sin(2 * np.pi * x / L)
        return np.sin(np.pi * (x / L + 0.25))**peak_sharpness_setting

    d[0,:] = initial_d(grid)

    # v is chosen to make a right-moving disturbance at the initial time.
    #v[0,:] = -wave_speed * (-2 * (grid - L/4) * peak_sharpness_setting**2 * np.exp(-(grid - L/4)**2 * peak_sharpness_setting**2) - np.exp(-(L - L/4)**2 * peak_sharpness_setting**2) / L)
    #v[0,:] = -wave_speed * 2 * np.pi / L * np.cos(2 * np.pi * grid / L)
    initial_v_grid = (grid + wave_speed * 0.5 * timestep_size) % L
    v[0,:] = -peak_sharpness_setting * wave_speed * np.pi / L * np.cos(np.pi * (initial_v_grid / L + 0.25)) * np.sin(np.pi * (initial_v_grid / L + 0.25))**(peak_sharpness_setting - 1)

    # Step through all time points
    for i_t in range(0, n_t - 1):
        #print(time[i_t])

        # v is changed by forces from the neighbouring grid points
        displacement_left = np.roll(d[i_t,:], 1) - d[i_t,:]
        displacement_right = np.roll(d[i_t,:], -1) - d[i_t,:]
        v[i_t+1,:] = v[i_t,:] + wave_speed**2 / grid_spacing**2 * (displacement_left + displacement_right) * timestep_size

        # d is moved by the updated v
        d[i_t+1,:] = d[i_t,:] + v[i_t+1,:] * timestep_size

    exact_d = initial_d(grid[None,:] - wave_speed * time[:,None])

    print(f"timestep_size={timestep_size}, CFL-limit={grid_spacing/wave_speed}")

    return time, grid, d, v, exact_d

def animate_results(time, grid, d, timesteps_per_frame):
    blocks = [amp.blocks.Line(grid, d[::timesteps_per_frame,:], label="d")]
    d_max = np.max(d[::timesteps_per_frame,:])
    extra_space = 0.1 * d_max
    d_max += extra_space
    d_min = np.min(d[::timesteps_per_frame,:]) - extra_space
    plt.ylim(d_min, d_max)
    timeline = amp.Timeline(time[::timesteps_per_frame], fps=60)
    anim = amp.Animation(blocks, timeline)
    anim.controls()
    plt.show()
    return

def animate_results_with_error(time, grid, d, exact_d, timesteps_per_frame):
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
    anim.controls()
    plt.show()
    return

def run_and_plot(wave_speed, timestep_size, n_gridpoints, timesteps_per_frame):
    # Run the simulation
    #time, grid, d, v = simulate_wave_with_loops(wave_speed, timestep_size, n_gridpoints)
    #time, grid, d, v = simulate_wave_with_array_operations(wave_speed, timestep_size, n_gridpoints)
    time, grid, d, v, exact_d = simulate_wave_with_array_operations_periodic(wave_speed, timestep_size, n_gridpoints)
    #time, grid, d, v = simulate_gas_wave_with_array_operations_periodic(wave_speed, timestep_size, n_gridpoints)
    #time, grid, d, v = simulate_gas_wave_with_array_operations_periodic_staggered(wave_speed, timestep_size, n_gridpoints)

    # Make a move of the results
    animate_results(time, grid, d, timesteps_per_frame)
    #animate_results_with_error(time, grid, d, exact_d, timesteps_per_frame)
