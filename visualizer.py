import cv2
import numpy as np
from PIL import Image

from particle_filter import resample, random_particles

from computer_vision import get_measurements


def visualize_sim(
        M:int,
        z:np.ndarray,
        std_p:float,
        std_v:float,
        std_q:float,
        threshold:float,
        injection_ratio:float,
        speed:float,
        file = 'annoying_bird.mov'
):
    """
    On screen video playback and vizualizer of Particle Filter that tracks the bird.
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
    """

    cap = cv2.VideoCapture(file)
    
    # 1x frame rate = 60
    fps = 60*speed
    # time between frames
    t = int(1000/fps)

    state = random_particles(M,std_v)
    #print(state.shape)

    k = 0

    while True:
        # ret: Bool, True if frame successfully opened
        # frame: NumPy array of image frame
        ret, frame = cap.read()
        if not ret: # if eg. end of video
            break

        particles = state[:2, :]
        print(state.shape)

        for i in range(M):
            #print(particles[:,i])
            x,y = np.int16(particles[:,i])
            print(z.shape)
            cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1) 
        
        cv2.circle(frame, (int(z[0,k]), int(z[1,k])), radius=8, color=(0, 255, 0), thickness=-1) 

        # show frame in window "Frame"
        cv2.imshow("Frame", frame)

        # wait t ms or until esc is pressed
        key = cv2.waitKey(t)
        if key == 27: # esc = 27
            break

        state = resample(state, M, z, std_p, std_v, std_q, threshold, injection_ratio, speed)
        k = k+1

    cap.release()
    cv2.destroyAllWindows()


z = get_measurements('annoying_bird.mov', 1, 15, False)

visualize_sim(100,z,0.001,0.01,5.,0.002,0.4,1.)

