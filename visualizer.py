import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from numpy.ma.extras import apply_along_axis

from particle_filter import next_frame, random_particles
from computer_vision import get_measurements, current_measurement, current_measurement_robust, current_ground_truth



def plot_errors(errors: np.ndarray, title:str) -> None:
    """
    Plots the errors
    Params:
        errors : NumPy array (T), the error for each frame
        title: string, the title of the plot
    Return:
        None
    """
    plt.figure(num=title)
    mae = int(np.mean(errors))

    plt.plot(errors)
    plt.grid()

    plt.suptitle(title, fontweight="bold")
    plt.title("MAE: " + str(mae) + " pixels")
    plt.xlabel("Frame Number")
    plt.ylabel("Frame Error")

    # plt.show() # We call just one plt.show() at the end
    return



def clean_errors(errors: np.ndarray, stride: int) -> np.ndarray:
    """
    Removes errors right after an invalid measurement section, to take into
    account only the regions where the filter converged already.
    Params:
        errors : NumPy array (T), the error for each frame
        stride: int, how many frames to neglect after the invalid measurement section
    Return:
        errors_pruned : NumPy array (T), the errors pruned
    """
    errors_pruned = np.zeros(errors.size)

    for i in range(errors.size - stride):
        if errors[i] == 0.:
            errors_pruned[i + stride] = 0.
        else:
            errors_pruned[i + stride] = errors[i + stride]
    for i in range(stride):
        errors_pruned[-stride] = errors[-stride]

    return errors_pruned



def visualize_sim(
        M: int,
        std_p: float,
        std_v: float,
        std_q: float,
        threshold: float,
        injection_ratio: float,
        injection_distance: int,
        speed: float = 1.,
        file: str = 'annoying_bird.mov',
        std_e: int = 4,
        play:bool = True
) -> (float, np.ndarray):
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
        injection_distance: int, if two consecutive measurements are farther than this value,
        we still inject some particles even if the measurement is valid
        speed : float, playback speed
        file : string, video file name
        std_e : int, measurement error standard deviation
        play : boolean, if True, plays video on screen
    Return:
        errors: NumPy array (T), the error for each frame
    """
    no_measurement = -1 * np.ones((2, 1))
    cap = cv2.VideoCapture(file)

    # 1x frame rate = 60
    fps = 60 * speed
    # time between frames
    t = int(1000 / fps)

    state = random_particles(M, std_v)

    k = 0
    errors = np.zeros(1000)
    z_prev = np.zeros((2, 1))

    while True:
        # ret: Bool, True if frame successfully opened
        # frame: NumPy array of image frame
        ret, frame = cap.read()
        if not ret:  # if eg. end of video
            break


        particles = state[:2, :]
        z_k = current_measurement(frame, std_e)
        true_pos = current_ground_truth(frame)

        pose_predicted = np.mean(particles,1)
        pose_predicted = np.resize(pose_predicted, (2, 1))

        if not np.array_equal(z_k, no_measurement):
            errors[k] = np.linalg.norm(true_pos - pose_predicted)

        if play:
            if not np.array_equal(z_k, no_measurement):
                # draw measurement in green
                cv2.circle(frame, (int(z_k[0, 0]), int(z_k[1, 0])), radius=8, color=(0, 255, 0), thickness=-1)
                # draw true position in blue
                cv2.circle(frame, (int(true_pos[0,0]), int(true_pos[1,0])), radius=8, color=(255, 0, 0), thickness=-1)
            # draw predicted position in magenta
            cv2.circle(frame, (int(pose_predicted[0, 0]), int(pose_predicted[1, 0])), radius=8, color=(255, 0, 255), thickness=-1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            # Use putText() method for
            # inserting text on video
            frame_count = str(k)
            cv2.putText(frame, 
                        frame_count, 
                        (50, 50), 
                        font, 1, 
                        (0, 255, 255), 
                        2, 
                        cv2.LINE_4)

            for i in range(M):
                x, y = np.int16(particles[:, i])
                # draw particles in red
                cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

            # show frame in window "Frame"
            cv2.imshow("Frame", frame)

            # wait t ms or until esc is pressed
            key = cv2.waitKey(t)
            if key == 27:  # esc = 27
                break

        measurement_distance = int(np.linalg.norm(z_prev - z_k))
        z_prev = z_k

        if measurement_distance >= injection_distance:
            state = next_frame(state, M, z_k, std_p, std_v, std_q, threshold, injection_ratio, True)
        else:
            state = next_frame(state, M, z_k, std_p, std_v, std_q, threshold, injection_ratio, False)
        k = k + 1

    errors = errors[:k]

    cap.release()
    cv2.destroyAllWindows()

    return errors



def visualize_sim_z_given(
        M:int,
        z:np.ndarray,
        std_p:float,
        std_v:float,
        std_q:float,
        threshold:float,
        injection_ratio:float,
        speed:float,
        file:str = 'annoying_bird.mov'
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
        state = next_frame(state, M, z_k, std_p, std_v, std_q, threshold, injection_ratio, False)
        k = k+1

    cap.release()
    cv2.destroyAllWindows()
