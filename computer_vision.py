import cv2
import numpy as np
from PIL import Image





#
# COMPUTER VISION FUNCTIONS
#



def hsv_limits(h_range):
    """
    Get HSV range limits for red colour.
    """
    lsat = 200
    uval = 200
    lower_red_0 = np.array([0, lsat, 100])
    upper_red_0 = np.array([h_range, 255, uval])
    lower_red_1 = np.array([180 - h_range, lsat, 100])
    upper_red_1 = np.array([180, 255, uval])

    return lower_red_0, upper_red_0, lower_red_1, upper_red_1



def color_mask(frame, h_range):
    """
    Create mask for red color given sensitivity in H-value.
    """

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l0, u0, l1, u1 = hsv_limits(h_range)

    mask0 = cv2.inRange(frame_hsv, l0, u0)
    mask1 = cv2.inRange(frame_hsv, l1, u1)
    mask = cv2.bitwise_or(mask0, mask1)

    return mask



def current_measurement_robust(frame, h_range) -> np.ndarray:
    """
    Get measurements z_x, z_y of bird's position for a given frame.
    Filters small components for robustness.
    Params:
        frame : the frame
        h_range : int, sensitivity in HSV H-value for tracking red color
    Return:
        z : NumPy array (2,1), of z_x, z_y for this frame.
            Values of -1 indicate no measurements (bird out of frame)
    """
    z = np.zeros((2, 1))

    # blur for noise reduction
    frame_blurred = cv2.GaussianBlur(frame, (15, 15), 0)

    # mask for red colours
    mask = color_mask(frame_blurred, h_range)

    if mask.any():  # if there is red on screen

        # We want to remove small contours (not the big red blob that is the birds hair)

        # Get connected components
        _, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(mask)

        sizes = stats[:, cv2.CC_STAT_AREA]  # get areas of components in order of size

        min_size = sizes[1]  # bg gives sizes[0], hair given as second larges blob i.e. sizes[1]

        # filter away small components
        mask = np.where(sizes[im_with_separated_blobs] >= min_size, mask, 0)

        # Find indices where we have mass
        mass_y, mass_x = np.where(mask >= 255)
        # mass_x and mass_y are the list of x indices and y indices of mass pixels

        # find center of mass_curr
        cent_x = int(np.average(mass_x))
        cent_y = int(np.average(mass_y))

        # plot center
        cv2.circle(frame, (cent_x, cent_y), radius=8, color=(0, 255, 0), thickness=-1)

        # position is center of mass
        z[0, :] = cent_x
        z[1, :] = cent_y

    else:
        # if no red on screen, set measurement -1 (invalid pos)
        z[0, :] = -1
        z[1, :] = -1

    return z



def current_ground_truth(frame):
    """
    Get ground truth position x,y of bird for a given frame.
    Params:
        frame : the frame
    Return:
        z : NumPy array (2,1), of x, y for this frame.
            Values of -1 indicate no position (bird out of frame)
    """
    pos = current_measurement_robust(frame, 2)
    return pos



def current_measurement_rand_jump(frame, std, p):
    """
    Get simulated measurement position z_x,z_y of bird for a given frame.
    Implemented as ground truth with added gaussian noise as error.
    Params:
        frame : the frame
        std : standard deviation of measurement error
        p: float in [0,1], probability that the measurement model returns an outlier
    Return:
        z : NumPy array (2,1), of measurement z_x, z_y for this frame.
            Values of -1 indicate no position (bird out of frame)
    """
    r = 2000

    true_pos = current_ground_truth(frame)

    no_measurement = -1 * np.ones((2, 1))

    z = true_pos
    x = z[0,0]
    y = z[1,0]

    if not np.array_equal(true_pos, no_measurement):
        z = z + np.random.normal(0,std,(2,1))
        if np.random.uniform(0.,1.) <= p:
            x0 = max(0, x-r)
            x1 = min(1280, x+r)
            y0 = max(0, y-r)
            y1 = min(720, y+r)
            z = np.random.randint([x0, y0], [x1, y1]).reshape((2,1))
            print("Outlier incoming")

    return z










#
# PREVIOUS VERSIONS
#
# The following functions have been used during the project development
# but have been discarded or replaced by modified versions for the final delivery.
# This means we report them here for completeness, but they are not invoked by
# the final simulation.
#



def get_and_play(file='annoying_bird.mov', speed=1):
    """
    Basic video player.
    """

    cap = cv2.VideoCapture(file)

    # 1x frame rate = 60
    fps = 60 * speed
    # time between frames
    t = int(1000 / fps)

    while True:
        # ret: Bool, True if frame successfully opened
        # frame: NumPy array of image frame
        ret, frame = cap.read()
        if not ret:  # if eg. end of video
            break

        frame = cv2.GaussianBlur(frame, (15, 15), 0)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # show frame in window "Frame"
        cv2.imshow("Frame", frame)

        # wait t ms or until esc is pressed
        key = cv2.waitKey(t)
        if key == 27:  # esc = 27
            break

    cap.release()
    cv2.destroyAllWindows()



def current_measurement_old(frame, h_range = 15) -> np.ndarray:
    """
    Get measurements z_x, z_y of bird's position for a given frame.
    Params:
        frame : the frame
        h_range : int, sensitivity in HSV H-value for tracking red color
    Return:
        z : NumPy array (2,1), of z_x, z_y the frame.
            Values of -1 indicate no measurements (bird out of frame)
    """
    z = np.zeros((2,1))

    # blur for noise reduction
    frame_blurred = cv2.GaussianBlur(frame, (15, 15), 0)

    # mask for red colours
    mask = color_mask(frame_blurred, h_range)

    # get bounding box for red objects
    mask_pil = Image.fromarray(mask)
    bbox = mask_pil.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        # position measurements are center of box
        z[0,:] = ((x1 + x2) / 2)
        z[1,:] = ((y1 + y2) / 2)
    else:
        # if no red on screen, set measurement to -1 (invalid pos)
        z[0, :] = -1
        z[1, :] = -1

    return z



def get_measurements(file = 'annoying_bird.mov', speed = 1., h_range = 15, play=True) -> np.ndarray:
    """
    Get measurements z_x, z_y of bird's position.
    Params:
        file : string, video file name
        speed : float, playback speed
        h_range : int, sensitivity in HSV H-value for tracking red color
        play : boolean, if True, plays video on screen
    Return:
        z : NumPy array (2,T), of z_x, z_y for each frame.
            Values of -1 indicate no measurements (bird out of frame)
    """
    cap = cv2.VideoCapture(file)
    
    # 1x frame rate = 60
    fps = 60*speed
    # time between frames
    t = int(1000/fps)

    # x, y position measurements
    z_x = []
    z_y = []

    while True:
        # ret: Bool, True if frame successfully opened
        # frame: NumPy array of image frame
        ret, frame = cap.read()
        if not ret: # if eg. end of video
            break

        # blur for noise reduction
        frame_blurred = cv2.GaussianBlur(frame, (15, 15), 0)  
        
        # mask for red colours
        mask = color_mask(frame_blurred, h_range)

        # get bounding box for red objects
        mask_pil = Image.fromarray(mask)
        bbox = mask_pil.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),5) # draw box

            # position measurements are center of box
            z_x.append((x1+x2)/2)
            z_y.append((y1+y2)/2)
        else:
            # if no red on screen, set measurement -1 (invalid pos)
            z_x.append(-1)
            z_y.append(-1)

        if play is True:
            cv2.imshow("Frame", frame)

            # wait t ms or until esc is pressed
            key = cv2.waitKey(t)
            if key == 27: # esc = 27
                break
    
    z_x = np.array(z_x)
    z_y = np.array(z_y)
    z = np.vstack((z_x, z_y))

    cap.release()
    cv2.destroyAllWindows()

    return z



def get_measurements_robust(file = 'annoying_bird.mov', speed = 1., h_range = 2, play=True):
    """
    Get measurements z_x, z_y of bird's position.
    Filters small components for robustness.
    Params:
        file : video file
        speed : float, playback speed
        h_range : int, sensitivity in HSV H-value for tracking red color
        play : boolean, if True, plays video on screen
    Return:
        z : 2xN NumPy array of z_x, z_y for each frame.
            Values of -1 indicate no measurements (bird out of frame)
    """
    cap = cv2.VideoCapture(file)
    
    # 1x frame rate = 60
    fps = 60*speed
    # time between frames
    t = int(1000/fps)

    # x, y position measurements
    z_x = []
    z_y = []

    while True:
        # ret: Bool, True if frame successfully opened
        # frame: NumPy array of image frame
        ret, frame = cap.read()
        if not ret: # if eg. end of video
            break

        z_curr = current_measurement_robust(frame, h_range)
        z_x.append(z_curr[0])
        z_y.append(z_curr[1])


        if play is True:
            cv2.imshow("Frame", frame)
            #cv2.imshow("Frame", mask)

            # wait t ms or until esc is pressed
            key = cv2.waitKey(t)
            if key == 27: # esc = 27
                break
    
    z_x = np.array(z_x)
    z_y = np.array(z_y)
    z = np.vstack((z_x, z_y))

    cap.release()
    cv2.destroyAllWindows()

    return z



def current_measurement_dropout(frame, std, dropout):
    """
    Get simulated measurement position z_x,z_y of bird for a given frame.
    Implemented as ground truth with added gaussian noise as error.
    Params:
        frame : the frame
        std : standard deviation of measurement error
        dropout: float in [0,1], probability that the measurement model fails
    Return:
        z : NumPy array (2,1), of measurement z_x, z_y for this frame.
            Values of -1 indicate no position (bird out of frame)
    """
    true_pos = current_ground_truth(frame)

    no_measurement = -1 * np.ones((2, 1))

    z = true_pos

    if not np.array_equal(true_pos, no_measurement):
        z = z + np.random.normal(0,std,(2,1))

        if np.random.uniform(0.,1.) <= dropout:
            z = no_measurement

    return z
