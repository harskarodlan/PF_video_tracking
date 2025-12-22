import numpy as np

# Bounds pf the Frame
X_LEFT = 0
X_RIGHT = 1280
Y_LOWER = 720
Y_UPPER = 0



def wrap_state(state:np.ndarray) -> np.ndarray:
    """
    Ensure all the particles lie in the frame.
    Params:
        state : NumPy array (4,M), the particles, rows are respectively x, y, v_x, v_y
    Return:
        state_wrapped : NumPy array (4,M), particle wrapped in the frame
    """
    state_wrapped = np.vstack((
        np.clip( state[0,:], X_LEFT, X_RIGHT ),
        np.clip( state[1,:], Y_UPPER, Y_LOWER ), # Y-axis is upside down, so Y_LOWER > Y_UPPER
        state[2:4,:]
    ))

    return state_wrapped



def random_particles(M:int, std_v:float) -> np.ndarray:
    """
    Returns M particles with uniform distribution on the position (bounds inferred from frame size)
    and normal distribution with zero mean on the velocities.
    Params:
        M : int, the number of particles
        std_v : float, the standard deviation for the velocities
    Return:
        state : NumPy array (4,M), the generated particles, rows are respectively x, y, v_x, v_y
    """
    state = np.vstack((
        np.random.uniform(X_LEFT, X_RIGHT, (1, M)),
        np.random.uniform(Y_UPPER, Y_LOWER, (1, M)), # Y-axis is upside down, so Y_LOWER > Y_UPPER
        np.random.normal(0.,std_v,(2,M))
    ))

    return state



def predict(state:np.ndarray, std_p:float, std_v:float, speed:float) -> np.ndarray:
    """
    Predicts the new position of the bird for the next frame.
    Params:
        state : NumPy array (4,M), the particles, rows are respectively x, y, v_x, v_y
        std_p : float, standard deviation for x and y
        std_v : float, standard deviation for v_x and v_y
        speed : float, playback speed
    Return:
        state_bar : NumPy array (4,M), particle values for the next frame
    """
    M = state.shape[1] # number of particles

    fps = 60 * speed # frames per second
    delta_t = int(1000/fps) # milliseconds between two frames

    state_bar = np.vstack((
        state[0:2,:] + delta_t * state[2:4,:],
        state[2:4,:]
    ))

    diffusion = np.vstack((
        np.random.normal(0,std_p,(2,M)),
        np.random.normal(0,std_v,(2,M))
    ))

    state_bar = state_bar + diffusion

    return state_bar



def weight_particles(state_bar:np.ndarray, z:np.ndarray, std_q:float, threshold:float) -> np.ndarray:
    """
    Computes the weights of the particles.
    Params:
        state_bar : NumPy array (4,M), the particles, rows are respectively x, y, v_x, v_y
        z : NumPy array (2,1), the measurement for this frame
        std_q: float, standard deviation for the measurement model
        threshold: float, the threshold to detect outlier measurements
    Return:
        weights : NumPy array (M), the weights of the particles
    """
    M = state_bar.shape[1]  # number of particles

    z_extended = np.tile(z, (1,M))
    eta = state_bar[0:2,:] - z_extended # innovation

    psi = -0.5 * np.square(eta) / (std_q ** 2)
    psi = np.exp( np.sum(psi,0) )
    psi = psi / (2 * np.pi * (std_q ** 2)) # gaussian normalization

    # Outlier Detection
    mean_psi = np.mean(psi)
    if mean_psi <= threshold:
        psi = np.ones(M)

    weights = psi / np.sum(psi) # normalized weights

    return weights



def multinomial_resample(state_bar:np.ndarray, weights:np.ndarray) -> np.ndarray:
    """
    Performs multinomial resample on the particles.
    Params:
        state_bar : NumPy array (4,M), the particles, rows are respectively x, y, v_x, v_y
        weights : NumPy array (M), the weights of the particles
    Return:
        state : NumPy array (4,M), resampled particle values
    """
    M = state_bar.shape[1]  # number of particles
    cdf = np.cumsum(weights)
    state = np.zeros(state_bar.shape)

    for m in range(M):
        r_m = np.random.uniform(0,1)
        i = np.where(cdf >= r_m)[0][0]
        state[:,m] = state_bar[:,i]

    return state



def systematic_resample(state_bar:np.ndarray, weights:np.ndarray) -> np.ndarray:
    """
    Performs systematic resample on the particles.
    Params:
        state_bar : NumPy array (4,M), the particles, rows are respectively x, y, v_x, v_y
        weights : NumPy array (M), the weights of the particles
    Return:
        state : NumPy array (4,M), resampled particle values
    """
    M = state_bar.shape[1]  # number of particles
    cdf = np.cumsum(weights)
    state = np.zeros(state_bar.shape)
    r_0 = np.random.uniform(0,1) / M

    for m in range(M):
        i = np.where(cdf >= r_0)[0][0]
        state[:,m] = state_bar[:,i]
        r_0 = r_0 + 1/M

    return state



def systematic_resample_new_size(state_bar:np.ndarray, weights:np.ndarray, new_M:int) -> np.ndarray:
    """
    Performs systematic resample on the particles, the new number of particles will be new_M.
    Params:
        state_bar : NumPy array (4,M), the particles, rows are respectively x, y, v_x, v_y
        weights : NumPy array (M), the weights of the particles
        new_M : int, the new number of particles
    Return:
        state : NumPy array (4,new_M), resampled particle values
    """
    cdf = np.cumsum(weights)
    state = np.zeros((4, new_M))
    r_0 = np.random.uniform(0, 1) / new_M

    for m in range(new_M):
        i = np.where(cdf >= r_0)[0][0]
        state[:, m] = state_bar[:, i]
        r_0 = r_0 + 1 / new_M

    return state



def injection_resample(state_bar:np.ndarray, weights:np.ndarray, injection_ratio:float=0., std_v:float=0.) -> np.ndarray:
    """
    Performs systematic resample on the particles, but a fraction of them equal to
    injection_ratio will be randomly generated instead of sampled.
    Params:
        state_bar : NumPy array (4,M), the particles, rows are respectively x, y, v_x, v_y
        weights : NumPy array (M), the weights of the particles
        injection_ratio : float in [0,1], the fraction of particles to generate instead of sample
        std_v : float, the standard deviation to sample the velocities
    Return:
        state : NumPy array (4,M), resampled particle values
    """
    M = state_bar.shape[1] # number of particles
    gen_M = int(M * injection_ratio) # number of particles to generate
    draw_M = M - gen_M # number of particles to draw

    if draw_M <= 0: # injection_ratio is 1 or greater, we generate everything randomly
        state = random_particles(M, std_v)
    elif draw_M >= M: # injection_ratio is 0 or negative, it's a normal systematic resample
        state = systematic_resample(state_bar,weights)
    else: # injection_ratio is in (0,1), so we partially resample and partially generate
        old_particles = systematic_resample_new_size(state_bar,weights,draw_M)
        new_particles = random_particles(gen_M, std_v)

        state = np.hstack((
            old_particles, new_particles
        ))

    return state


def resample(
        state:np.ndarray,
        M:int,
        z_t:np.ndarray,
        std_p:float,
        std_v:float,
        std_q:float,
        threshold:float,
        injection_ratio:float,
        speed:float
):
    
    no_measurement = -1 * np.ones((2,1))

    state_bar = predict(state,std_p,std_v,speed)
    z_t = np.resize(z_t,(2,1)) # numpy returns a (2) we need a (2,1)

    # We have a measurement, the filter is in normal mode and all particles are resampled
    if np.array_equal(z_t,no_measurement):
        weights = np.ones(M) / M
        state = injection_resample(state_bar, weights, injection_ratio, std_v)
    # We have no measurement, the filter is in recover mode and some particles are injected
    else:
        weights = weight_particles(state_bar, z_t, std_q, threshold)
        state = systematic_resample(state_bar, weights)

    state = wrap_state(state)

    return state



def particle_filter(
        M:int,
        z:np.ndarray,
        std_p:float,
        std_v:float,
        std_q:float,
        threshold:float,
        injection_ratio:float,
        speed:float
) -> np.ndarray:
    """
    The entire Particle Filter that tracks the bird.
    Params:
        M : int, the number of particles
        z : NumPy array (2,T), the measurements
        std_p : float, standard deviation for x and y
        std_v : float, the standard deviation to sample the velocities
        std_q: float, standard deviation for the measurement model
        threshold: float, the threshold to detect outlier measurements
        injection_ratio : float in [0,1], the fraction of particles to generate when the filter is in
        recover mode
        speed : float, playback speed
    Return:
        simulation : NumPy array (4,M,T), the particles, organized by frames
    """
    state = random_particles(M,std_v)
    T = z.shape[1] # number of measurements

    simulation = np.zeros((4,M,T))

    for t in range(T):
        z_t = np.resize(z_t,(2,1)) # numpy returns a (2) we need a (2,1)  
        state = resample(state, M, z_t, std_p, std_v, std_q, threshold. injection_ratio, speed)
        simulation[:,:,t] = state

    return simulation



def extract_pose(simulation:np.ndarray) -> np.ndarray:
    """
    Extracts the estimated bird position as the mean of the particles for each frame.
    Params:
        simulation : NumPy array (4,M,T), the particles, organized by frames
    Return:
        pose : NumPy array (2,T), the estimated bird position for each frame
    """
    pose = simulation[0:2,:,:]
    pose = np.mean(pose,1)
    return pose
