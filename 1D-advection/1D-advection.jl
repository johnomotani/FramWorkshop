using GLMakie
using Printf

function animate_array(t, x, array)
    @assert size(x) == size(array)[1:end-1]
    @assert size(t) == size(array)[end:end]

    i = Observable(1)
    n_t = length(t)
    d_t = t[2] - t[1]

    # To make animation reasonable, only draw a frame about every 0.02 seconds.
    steps_per_frame = max(floor(Int64, 0.02 / d_t), 1)
    animation_d_t = steps_per_frame * d_t

    d_x = x[2] - x[1]
    n_x = length(x)

    # Positions of cell edges
    x_edges = zeros(2 * n_x)
    for i_x in 1:n_x
        x_edges[2*i_x-1] = x[i_x] - 0.5 * d_x
        x_edges[2*i_x] = x[i_x] + 0.5 * d_x
    end

    function get_stepped_array_slice(i_t)
        array_slice = @view array[:,i_t]
        return [array_slice[div(i_x,2)] for i_x in 2:2*n_x+1]
    end

    array_observable = @lift get_stepped_array_slice($i)
    title = @lift string(@sprintf "t = %.3f" t[$i])

    fig = Figure()
    ax = Axis(fig[1,1]; title=title)
    lines!(ax, x_edges, array_observable)
    display(fig)

    for i[] in 1:steps_per_frame:n_t
        sleep(animation_d_t)
    end
    i[] = n_t

    return nothing
end

"""
    simulate_advection(v, dt, n_grid)

Advect an initial profile with a constant velocity `v`, using a timestep `dt` and `n_grid`
grid points.
"""
function simulate_advection(v, d_t, n_grid)
    # Time that simulation runs for
    t_total = 10.0

    # Length of simulation domain
    L = 1.0

    # Total number of time steps
    n_t = floor(Int64, t_total / d_t) + 1

    # Time of each time step
    t = [i * d_t for i in 0:n_t-1]

    # Width of grid cells
    d_x = L / n_grid

    # Positions of centres of grid cells
    x = [(i + 0.5) * d_x for i in 0:n_grid-1]

    # Check whether time step size is stable
    if d_t > d_x / v
        println("Warning: time step too large, simulation will be unstable")
    end

    # Array to store the calculated values at every time step
    f = zeros(n_grid, n_t)

    # Create initial profile in first column of f.
    # Profile is a peak at L/2.
    for i_x in 1:n_grid
        f[i_x,1] = sin(pi * x[i_x] / L)^4
    end

    # Advance profile in time.
    for i_t in 2:n_t
        for i_x in 1:n_grid
            if i_x == 1
                # Makes grid periodic.
                i_previous = n_grid
            else
                i_previous = i_x - 1
            end

            # Flux arriving in cell through left boundary
            flux_left = d_t * v * f[i_previous,i_t-1]

            # Flux leaving in cell through right boundary
            flux_right = d_t * v * f[i_x,i_t-1]

            # Total f after timestep in this cell
            f[i_x,i_t] = f[i_x,i_t-1] + (flux_left - flux_right) / d_x
        end
    end

    return t, x, f
end

function run_and_animate(v, d_t, n_grid)
    t, x, f = simulate_advection(v, d_t, n_grid)
    animate_array(t, x, f)

    return nothing
end

# Example resolutions:
# * quite a bit of dissipation, v=0.2, d_t=2.0e-2, n_grid=100
# * little dissipation, v=0.2, d_t=2.0e-3, n_grid=1000
# * almost invisible dissipation, v=0.2, d_t=5.0e-4, n_grid=4000
