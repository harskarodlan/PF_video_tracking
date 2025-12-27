import matplotlib.pyplot as plt
from visualizer import visualize_sim, plot_errors, clean_errors



errors = visualize_sim(
    M = 500,
    std_p = 10.,
    std_v = 10.,
    std_q = 20.,
    threshold = 0,
    injection_ratio = 0.01,
    injection_distance = 100,
    speed = 0.5,
    file = 'annoying_bird.mov',
    std_e = 4,
    play = True
)

print(errors.shape)
plot_errors(errors, "Error for Each Frame")

errors_1 = clean_errors(errors, 1)
plot_errors(errors_1, "Error After Pruning 1")
errors_2 = clean_errors(errors, 5)
plot_errors(errors_2, "Error After Pruning 2")
errors_3 = clean_errors(errors, 10)
plot_errors(errors_3, "Error After Pruning 3")

plt.show()
