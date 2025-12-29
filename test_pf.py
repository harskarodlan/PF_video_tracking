from math import trunc

import numpy as np
from particle_filter import wrap_state, random_particles, predict, weight_particles
from particle_filter import systematic_resample, systematic_resample_new_size, multinomial_resample, injection_resample

from particle_filter import particle_filter, extract_pose
from computer_vision import get_measurements



print("--- Testing wrap_state ---")
state = np.hstack((
    -5 * np.ones((4,3)), 25 * np.ones((4,6)), 8000 * np.ones((4,2))
))
print(state)
state = wrap_state(state)
print(state)
print("--- --- ---\n")

print("--- Testing random_particles ---")
state = random_particles(10,1.)
print(state)
print("--- --- ---\n")

print("--- Testing predict ---")
state = np.zeros((4,10))
state[1:3,:] = np.ones((2,10))
state[3,:] = 2 * np.ones((1,10))
print(state)
state_bar = predict(state,0.001,0.1)
print(state_bar)
print("--- --- ---\n")

print("--- Testing weight_particles ---")
state_bar = np.zeros((4,10))
for i in range(10):
    state_bar[:,i] = i * np.ones(4)
print(state_bar)
z = np.array([[1.],[2.]])
print(z)
weights = weight_particles(state_bar,z, 5., 0.002, True)
print(weights)
print("--- --- ---\n")



print("--- Testing resamples ---")
print(state_bar)

weights = 0.05 * np.ones(10)
weights[0] = 0.55
print(weights)

print("Multinomial")
state = multinomial_resample(state_bar,weights)
print(state)

print("Systematic")
state = systematic_resample(state_bar,weights)
print(state)

print("Systematic New Size")
state = systematic_resample_new_size(state_bar,weights,15)
print(state)

print("Injection")
state = injection_resample(state_bar,weights,0.4,1.)
print(state)
print("--- --- ---\n")



print("--- Testing particle_filter ---")
z = get_measurements('annoying_bird.mov', 1, 5, False)
print(z.shape)
simulation = particle_filter(10,z,0.001,0.01,5.,0.002,0.4)
print(simulation.shape)
pose = extract_pose(simulation)
print(pose.shape)
print("--- --- ---\n")
