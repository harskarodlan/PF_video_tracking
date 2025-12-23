from computer_vision import get_and_play, get_measurements, get_measurements_robust

get_and_play('annoying_bird.mov', 0.5)

z = get_measurements('annoying_bird.mov', 1., 5, True)
print(z)
print(z.shape)

z = get_measurements_robust('annoying_bird.mov', 1., 2, True)
