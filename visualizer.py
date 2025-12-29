import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from numpy.ma.extras import apply_along_axis

from particle_filter import next_frame, random_particles
from computer_vision import current_measurement_rand_jump, current_ground_truth





#
# PLOTTING FUNCTIONS
#



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



def plot_errors_interval(errors: np.ndarray, title:str, start:int, end:int) -> None:
    """
    Plots the errors in a given interval [start,end)
    Params:
        errors : NumPy array (T), the error for each frame
        title: string, the title of the plot
        start: int, starting frame, included
        start: int, ending frame, excluded
    Return:
        None
    """
    errors_interval = errors[start:end]

    plt.figure(num=title)
    mae = int(np.mean(errors_interval))

    plt.plot(range(start,end), errors_interval)
    plt.grid()

    plt.suptitle(title, fontweight="bold")
    plt.title("MAE in [" + str(start) + "," + str(end) + "): " + str(mae) + " pixels")
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





#
# SIMULATION FUNCTION
#



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
        std_e: float = 4.,
        dropout: float = 0.,
        play:bool = True
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
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
        dropout: float in [0,1], probability that the measurement model fails
        play : boolean, if True, plays video on screen
    Return:
        errors: NumPy array (T), the prediction error for each frame
        errors_meas: NumPy array (T), the measurement error for each frame
        poses_true: NumPy array (2,T), the ground truth poses for each frame
        measures: NumPy array (2,T), the measures for each frame
        poses_pf: NumPy array (2,T), the predicted poses for each frame
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
    errors_meas = np.zeros(1000)
    poses_true = np.zeros((2,1000))
    measures = np.zeros((2, 1000))
    poses_pf = np.zeros((2, 1000))
    z_prev = np.zeros((2, 1))

    while True:
        # ret: Bool, True if frame successfully opened
        # frame: NumPy array of image frame
        ret, frame = cap.read()
        if not ret:  # if eg. end of video
            break


        particles = state[:2, :]
        z_k = current_measurement_rand_jump(frame, std_e, dropout)
        true_pos = current_ground_truth(frame)
        pose_predicted = np.mean(particles,1)

        poses_true[:,k] = np.resize(true_pos, 2)
        measures[:,k] = np.resize(z_k, 2)
        poses_pf[:,k] = pose_predicted

        pose_predicted = np.resize(pose_predicted, (2, 1))

        if not np.array_equal(z_k, no_measurement):
            errors[k] = np.linalg.norm(true_pos - pose_predicted)
            errors_meas[k] = np.linalg.norm(true_pos - z_k)

        if play:
            for i in range(M):
                x, y = np.int16(particles[:, i])
                # draw particles in red
                cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

            if not np.array_equal(z_k, no_measurement):
                # draw measurement in green
                cv2.circle(frame, (int(z_k[0, 0]), int(z_k[1, 0])), radius=8, color=(0, 255, 0), thickness=-1)
                # draw true position in blue
                cv2.circle(frame, (int(true_pos[0,0]), int(true_pos[1,0])), radius=8, color=(255, 0, 0), thickness=-1)
            # draw predicted position in yellow
            cv2.circle(frame, (int(pose_predicted[0, 0]), int(pose_predicted[1, 0])), radius=8, color=(0, 255, 255), thickness=-1)

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

            # show frame in window "Frame"
            cv2.imshow("Frame", frame)

            # wait t ms or until esc is pressed
            key = cv2.waitKey(t)
            if key == 27:  # esc = 27
                break
            if key == 32:  # space = 32
                key2 = cv2.waitKey(10000)
                if key2 == 115: # s = 115
                    cv2.imwrite("./frames/frame_"+str(k)+".png", frame)

            # ATTENTION: uncomment the following line will save every single frame
            #cv2.imwrite("./movie/frame_" + str(k) + ".png", frame)

        measurement_distance = int(np.linalg.norm(z_prev - z_k))
        z_prev = z_k

        outlier_det = not np.array_equal(z_prev,no_measurement)

        if measurement_distance >= injection_distance:
            state = next_frame(state, M, z_k, std_p, std_v, std_q, threshold, injection_ratio, True, outlier_det)
        else:
            state = next_frame(state, M, z_k, std_p, std_v, std_q, threshold, injection_ratio, False, outlier_det)
        k = k + 1

    errors = errors[:k]
    errors_meas = errors_meas[:k]
    poses_true = poses_true[:,:k]
    measures = measures[:,:k]
    poses_pf = poses_pf[:,:k]

    cap.release()
    cv2.destroyAllWindows()

    return errors, errors_meas, poses_true, measures, poses_pf










#
# PREVIOUS VERSIONS
#
# The following functions have been used during the project development
# but have been discarded or replaced by modified versions for the final delivery.
# This means we report them here for completeness, but they are not invoked by
# the final simulation.
#



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
        state = next_frame(state, M, z_k, std_p, std_v, std_q, threshold, injection_ratio, False, False)
        k = k+1

    cap.release()
    cv2.destroyAllWindows()

    return



def compute_errors(predicted:np.ndarray, ground:np.ndarray) -> np.ndarray:
    """
    Computes the error between two pose sequences, as the Euclidean distance between the pixel positions.
    Params:
        predicted : NumPy array (2, T), the predicted pose
        ground : NumPy array (2, T), the ground truth
    Return:
        errors: NumPy array (T), the error
    """
    diff = predicted - ground
    diff = np.square(diff)
    diff = np.sum(diff, 0)
    diff = np.sqrt(diff)

    return diff
