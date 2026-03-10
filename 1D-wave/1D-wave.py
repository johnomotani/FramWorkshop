import numpy as np
import animatplot as amp
from matplotlib import pyplot as plt

"""
Simulate propagation of a wave in 1d, for example a string that is under tension.

Boundary condition is that there is no displacement at the end points of the grid.

Uses for loops for all operations.
"""
def simulate_wave_with_loops(timestep_size, n_gridpoints):
    L = 1.0
    total_time = 10.0
    wave_speed = 1.0

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
            # d is moved by the current v
            d[i_t+1,i] = d[i_t,i] + v[i_t,i] * timestep_size

            # v is changed by forces from the neighbouring grid points
            displacement_left = d[i_t,i-1] - d[i_t,i]
            displacement_right = d[i_t,i+1] - d[i_t,i]
            v[i_t+1,i] = v[i_t,i] + wave_speed**2 / grid_spacing**2 * (displacement_left + displacement_right) * timestep_size

    print(f"timestep_size={timestep_size}, CFL-limit={grid_spacing/wave_speed}")

    return time, grid, d, v

"""
Simulate propagation of a wave in 1d, for example a string that is under tension.

Boundary condition is that there is no displacement at the end points of the grid.

Uses numpy array operations for efficiency.
"""
def simulate_wave_with_array_operations(timestep_size, n_gridpoints):
    L = 1.0
    total_time = 10.0
    wave_speed = 1.0

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
        print(time[i_t])
        # Note we do not update the first and last grid points as these are fixed to
        # 0.0.

        # d is moved by the current v
        d[i_t+1,1:-1] = d[i_t,1:-1] + v[i_t,1:-1] * timestep_size

        # v is changed by forces from the neighbouring grid points
        displacement_left = d[i_t,:-2] - d[i_t,1:-1]
        displacement_right = d[i_t,2:] - d[i_t,1:-1]
        v[i_t+1,1:-1] = v[i_t,1:-1] + wave_speed**2 / grid_spacing**2 * (displacement_left + displacement_right) * timestep_size

    print(f"timestep_size={timestep_size}, CFL-limit={grid_spacing/wave_speed}")

    return time, grid, d, v

def simulate_wave_with_array_operations_periodic(timestep_size, n_gridpoints):
    L = 1.0
    total_time = 10.0
    wave_speed = 1.0

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
    #d[0,:] = np.exp(-(grid - L/4)**2 * 16**2) - np.exp(-(0.0 - L/4)**2 * 16**2) - np.exp(-(L - L/4)**2 * 16**2) * grid / L
    d[0,:] = np.sin(2 * np.pi * grid / L)

    # v is chosen to make a right-moving disturbance at the initial time.
    #v[0,:] = -wave_speed * (-2 * (grid - L/4) * 16**2 * np.exp(-(grid - L/4)**2 * 16**2) - np.exp(-(L - L/4)**2 * 16**2) / L)
    v[0,:] = -wave_speed * 2 * np.pi / L * np.cos(2 * np.pi * grid / L)

    # Step through all time points
    for i_t in range(0, n_t - 1):
        print(time[i_t])

        # d is moved by the current v
        d[i_t+1,:] = d[i_t,:] + v[i_t,:] * timestep_size

        # v is changed by forces from the neighbouring grid points
        displacement_left = np.roll(d[i_t,:], 1) - d[i_t,:]
        displacement_right = np.roll(d[i_t,:], -1) - d[i_t,:]
        v[i_t+1,:] = v[i_t,:] + wave_speed**2 / grid_spacing**2 * (displacement_left + displacement_right) * timestep_size

    print(f"timestep_size={timestep_size}, CFL-limit={grid_spacing/wave_speed}")

    return time, grid, d, v

"""
Simulate propagation of a wave in 1d, for example a gas in a pipe.

Boundary condition is that there is no displacement at the end points of the grid.

Uses numpy array operations for efficiency.
"""
def simulate_gas_wave_with_array_operations(timestep_size, n_gridpoints):
    L = 1.0
    total_time = 10.0
    wave_speed = 1.0

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
        print(time[i_t])
        # Note we do not update the first and last grid points as these are fixed to
        # 0.0.

        # d is moved by the current v
        d[i_t+1,1:-1] = d[i_t,1:-1] + v[i_t,1:-1] * timestep_size

        # v is changed by forces from the neighbouring grid points
        displacement_left = d[i_t,:-2] - d[i_t,1:-1]
        displacement_right = d[i_t,2:] - d[i_t,1:-1]
        v[i_t+1,1:-1] = v[i_t,1:-1] + wave_speed**2 / grid_spacing**2 * (displacement_left + displacement_right) * timestep_size

    print(f"timestep_size={timestep_size}, CFL-limit={grid_spacing/wave_speed}")

    return time, grid, d, v

def simulate_gas_wave_with_array_operations_periodic(timestep_size, n_gridpoints):
    L = 1.0
    total_time = 10.0
    wave_speed = 1.0

    n_t = int(total_time / timestep_size)
    time = np.fromiter((i_t * timestep_size for i_t in range(n_t)), float)

    grid_spacing = L / n_gridpoints
    grid = np.fromiter((i * grid_spacing for i in range(n_gridpoints)), float)

    # Array that will store values of the pressure p for all time points.
    p = np.zeros([n_t, n_gridpoints])

    # Array that will store values of the velocity v for all time points.
    v = np.zeros([n_t, n_gridpoints])

    # Set up initial profile of p and v
    #
    # Add some small modifications to a bell curve to make sure the initial profile
    # is exactly zero at the boundaries.
    #p[0,:] = np.exp(-(grid - L/4)**2 * 16**2) - np.exp(-(0.0 - L/4)**2 * 16**2) - np.exp(-(L - L/4)**2 * 16**2) * grid / L
    p[0,:] = np.sin(2 * np.pi * grid / L)

    # v is chosen to make a right-moving disturbance at the initial time.
    #v[0,:] = -wave_speed * (-2 * (grid - L/4) * 16**2 * np.exp(-(grid - L/4)**2 * 16**2) - np.exp(-(L - L/4)**2 * 16**2) / L)
    #v[0,:] = -wave_speed * 2 * np.pi / L * np.cos(2 * np.pi * grid / L)

    # Step through all time points
    for i_t in range(0, n_t - 1):
        print(time[i_t])

        # p is changed by the compression of v
        p[i_t+1,:] = p[i_t,:] - 0.5 * (np.roll(v[i_t,:], -1) - np.roll(v[i_t,:], 1)) / grid_spacing * timestep_size

        # v is changed by difference in pressure from the neighbouring grid points
        v[i_t+1,:] = v[i_t,:] - wave_speed**2 * 0.5 * (np.roll(p[i_t,:], -1) - np.roll(p[i_t,:], 1)) / grid_spacing * timestep_size

    print(f"timestep_size={timestep_size}, CFL-limit={grid_spacing/wave_speed}")

    return time, grid, p, v

def simulate_gas_wave_with_array_operations_periodic_staggered(timestep_size, n_gridpoints):
    L = 1.0
    total_time = 10.0
    wave_speed = 1.0

    n_t = int(total_time / timestep_size)
    time = np.fromiter((i_t * timestep_size for i_t in range(n_t)), float)

    grid_spacing = L / n_gridpoints
    grid = np.fromiter((i * grid_spacing for i in range(n_gridpoints)), float)

    # Array that will store values of the pressure p for all time points.
    p = np.zeros([n_t, n_gridpoints])

    # Array that will store values of the velocity v for all time points.
    v = np.zeros([n_t, n_gridpoints])

    # Set up initial profile of p and v
    #
    # Add some small modifications to a bell curve to make sure the initial profile
    # is exactly zero at the boundaries.
    #p[0,:] = np.exp(-(grid - L/4)**2 * 16**2) - np.exp(-(0.0 - L/4)**2 * 16**2) - np.exp(-(L - L/4)**2 * 16**2) * grid / L
    p[0,:] = np.sin(2 * np.pi * grid / L)

    # v is chosen to make a right-moving disturbance at the initial time.
    #v[0,:] = -wave_speed * (-2 * (grid - L/4) * 16**2 * np.exp(-(grid - L/4)**2 * 16**2) - np.exp(-(L - L/4)**2 * 16**2) / L)
    #v[0,:] = -wave_speed * 2 * np.pi / L * np.cos(2 * np.pi * grid / L)

    # Step through all time points
    for i_t in range(0, n_t - 1):
        print(time[i_t])

        # p is changed by the compression of v
        p[i_t+1,:] = p[i_t,:] - (np.roll(v[i_t,:], -1) - v[i_t,:]) / grid_spacing * timestep_size

        # v is changed by difference in pressure from the neighbouring grid points
        v[i_t+1,:] = v[i_t,:] - wave_speed**2 * (p[i_t,:] - np.roll(p[i_t,:], 1)) / grid_spacing * timestep_size

    print(f"timestep_size={timestep_size}, CFL-limit={grid_spacing/wave_speed}")

    return time, grid, p, v

def animate_results(time, grid, d, v, fps=30):
    timestep_size = time[1] - time[0]
    steps_per_output = max(int(1 / (timestep_size * fps)), 1)
    block = amp.blocks.Line(grid, d[::steps_per_output,:])
    d_max_absolute_value = 1.1 * np.max(np.abs(d[0,:]))
    plt.ylim(-d_max_absolute_value, d_max_absolute_value)
    timeline = amp.Timeline(time[::steps_per_output], fps=fps)
    anim = amp.Animation([block], timeline)
    anim.controls()
    plt.show()
    return

def run_and_plot(timestep_size, n_gridpoints):
    # Run the simulation
    #time, grid, d, v = simulate_wave_with_loops(timestep_size, n_gridpoints)
    #time, grid, d, v = simulate_wave_with_array_operations(timestep_size, n_gridpoints)
    #time, grid, d, v = simulate_wave_with_array_operations_periodic(timestep_size, n_gridpoints)
    #time, grid, d, v = simulate_gas_wave_with_array_operations_periodic(timestep_size, n_gridpoints)
    time, grid, d, v = simulate_gas_wave_with_array_operations_periodic_staggered(timestep_size, n_gridpoints)

    # Make a move of the results
    animate_results(time, grid, d, v)
