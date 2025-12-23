from computer_vision import get_measurements
from visualizer import visualize_sim_z_given

z = get_measurements('annoying_bird.mov', 1., 15, False)
visualize_sim_z_given(100,z,0.001,0.01,5.,0.002,0.4,1.)
visualize_sim_z_given(100,z,5.,0.1,20.,0.0,0.0,1.)
