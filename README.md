# PF_video_tracking

Final project for the Applied Estimation course at KTH.
We use a Particle Filter estimation algorithm to track the head of "The Annoying Bird" charter in a scene of "Kalle Ankas Jul".

## Files in this Repositories

In this repository you will find the following files:

- `annoying_bird.mov`: The video we are going to analyze.

- `computer_vision.py`: Contains the implementation of the computer vision functions used to extract the measurements.

- `particle_filter.py`: Contains the implementation of the functions related to the Particle Filter.

- `visualizer.py`: Contains the implementation of the functions that invokes the Particle Filter on the video.

- `simulate.py`: The main function that runs the algorithm with the selected parameters.

- `test_cv.py`: tester script used to debug `computer_vision.py`. 

- `test_pf.py`: tester script used to debug `particle_filter.py`. 

- `test_vis.py`: tester script used to debug `visualizer.py`. 

## How to run the Code

To run the project, open `simulate.py` and change the parameters at your pleasure.
Then, run the same file with

``
python3 ./simulate.py
``

## Maintainers

Alex Pegoraro

Theresa Johansson
