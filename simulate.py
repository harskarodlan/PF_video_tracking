import matplotlib.pyplot as plt
from visualizer import visualize_sim, plot_errors, plot_errors_interval, clean_errors



errors, errors_meas, poses_true, measures, poses_pf = visualize_sim(
    M = 1000,
    std_p = 5.,
    std_v = 10.,
    std_q = 20.,
    threshold = 0,
    injection_ratio = 0.005,
    injection_distance = 100,
    speed = 0.5,
    file = 'annoying_bird.mov',
    std_e = 20.,
    play = True
)

print(poses_true.shape)
print(measures.shape)
print(poses_pf.shape)

print(errors.shape)
plot_errors(errors, "Error for Each Frame")

print(errors_meas.shape)
plot_errors(errors_meas, "Measurement Error for Each Frame")

errors_1 = clean_errors(errors, 1)
plot_errors(errors_1, "Error After Pruning 1")
errors_2 = clean_errors(errors, 5)
plot_errors(errors_2, "Error After Pruning 2")
errors_3 = clean_errors(errors, 10)
plot_errors(errors_3, "Error After Pruning 3")

plot_errors_interval(errors, "Error for Initial Frames", 1, 201)
plot_errors_interval(errors_meas, "Measurement Error for Initial Frames", 1, 201)

plt.show()
