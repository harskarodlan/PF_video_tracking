import matplotlib.pyplot as plt
from visualizer import visualize_sim, plot_errors, plot_errors_interval, clean_errors



errors, errors_meas, poses_true, measures, poses_pf = visualize_sim(
    M = 5000,
    std_p = 5.,
    std_v = 10.,
    std_q = 20.,
    threshold = 1e-3, # for threshold detection
    # threshold = 0.65, # for Neff detection
    injection_ratio = 0.005,
    injection_distance = 100,
    speed = 2.,
    file = 'annoying_bird.mov',
    std_e = 20.,
    p_outlier = 0.1,
    detection_type = 'threshold',
    save_frames = None, # (0,200),
    play = True
)



print(poses_true)
print(measures)
print(poses_pf)

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

plot_errors_interval(errors, "Error for Initial Frames", 5, 200)
plot_errors_interval(errors_meas, "Measurement Error for Initial Frames", 5, 200)

plt.show()
