import cv2
import numpy as np
from PIL import Image

from particle_filter import next_frame, random_particles

from computer_vision import get_measurements, current_measurement



def visualize_sim(
        M: int,
        std_p: float,
        std_v: float,
        std_q: float,
        threshold: float,
        injection_ratio: float,
        speed: float,
        file: str = 'annoying_bird.mov',
        h_range: int = 15,
        play:bool = True
) -> float:
    """
    On screen video playback and visualizer of Particle Filter that tracks the bird.
    Params:
        M : int, the number of particles
        std_p : float, standard deviation for x and y
        std_v : float, the standard deviation to sample the velocities
        std_q: float, standard deviation for the measurement model
        threshold: float, the threshold to detect outlier measurements
        injection_ratio : float in [0,1], the fraction of particles to generate when the filter is in
        recover mode
        speed : float, playback speed
        file : string, video file name
        h_range : int, sensitivity in HSV H-value for tracking red color
        play : boolean, if True, plays video on screen
    Return:
        mae: float, the Mean Absolute Error of the simulation
    """
    no_measurement = -1 * np.ones((2, 1))
    cap = cv2.VideoCapture(file)

    # 1x frame rate = 60
    fps = 60 * speed
    # time between frames
    t = int(1000 / fps)

    state = random_particles(M, std_v)

    k = 0
    cum_err = 0
    while True:
        # ret: Bool, True if frame successfully opened
        # frame: NumPy array of image frame
        ret, frame = cap.read()
        if not ret:  # if eg. end of video
            break

        particles = state[:2, :]
        z_k = current_measurement(frame, h_range)

        pose_predicted = np.mean(particles,1)
        pose_predicted = np.resize(pose_predicted, (2, 1))

        if not np.array_equal(z_k, no_measurement):
            cum_err += np.linalg.norm(z_k - pose_predicted)

        if play:
            if not np.array_equal(z_k, no_measurement):
                cv2.circle(frame, (int(z_k[0, 0]), int(z_k[1, 0])), radius=8, color=(0, 255, 0), thickness=-1)
            cv2.circle(frame, (int(pose_predicted[0, 0]), int(pose_predicted[1, 0])), radius=8, color=(255, 0, 255), thickness=-1)

            for i in range(M):
                x, y = np.int16(particles[:, i])
                cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

            # show frame in window "Frame"
            cv2.imshow("Frame", frame)

            # wait t ms or until esc is pressed
            key = cv2.waitKey(t)
            if key == 27:  # esc = 27
                break

        state = next_frame(state, M, z_k, std_p, std_v, std_q, threshold, injection_ratio)
        k = k + 1

    mae = cum_err / k
    cap.release()
    cv2.destroyAllWindows()

    return mae



def visualize_sim_z_given(
        M:int,
        z:np.ndarray,
        std_p:float,
        std_v:float,
        std_q:float,
        threshold:float,
        injection_ratio:float,
        speed:float,
        file = 'annoying_bird.mov'
) -> None:
    """
    On screen video playback and visualizer of Particle Filter that tracks the bird,
    z is given already to increase speed.
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
        file : string, video file name
    Return:
        None
    """
    cap = cv2.VideoCapture(file)
    
    # 1x frame rate = 60
    fps = 60*speed
    # time between frames
    t = int(1000/fps)

    state = random_particles(M,std_v)

    k = 0
    while True:
        # ret: Bool, True if frame successfully opened
        # frame: NumPy array of image frame
        ret, frame = cap.read()
        if not ret: # if eg. end of video
            break

        particles = state[:2, :]

        for i in range(M):
            x,y = np.int16(particles[:,i])
            cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1) 

        cv2.circle(frame, (int(z[0,k]), int(z[1,k])), radius=8, color=(0, 255, 0), thickness=-1)

        # show frame in window "Frame"
        cv2.imshow("Frame", frame)

        # wait t ms or until esc is pressed
        key = cv2.waitKey(t)
        if key == 27: # esc = 27
            break

        z_k = np.resize(z[:, k], (2, 1))  # numpy returns a (2) we need a (2,1)
        state = next_frame(state, M, z_k, std_p, std_v, std_q, threshold, injection_ratio)
        k = k+1

    cap.release()
    cv2.destroyAllWindows()



# z = get_measurements('annoying_bird.mov', 1, 15, False)
# visualize_sim_z_given(100,z,0.001,0.01,5.,0.002,0.4,1.)
# visualize_sim_z_given(100,z,5.,0.1,20.,0.0,0.0,1.)

error = visualize_sim(
    M = 500,
    std_p = 10.,
    std_v = 10.,
    std_q = 20.,
    threshold = 0,
    injection_ratio = 0.01,
    speed = 0.5,
    file = 'annoying_bird.mov',
    h_range = 15,
    play = True
)

print(error)
