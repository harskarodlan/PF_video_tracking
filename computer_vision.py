import cv2
import numpy as np
from PIL import Image



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


def get_measurements(file = 'annoying_bird.mov', speed = 1, h_range = 15, play=True):
    """
    Get measurments z_x, z_y of position of bird.
    Params:
        file : video file
        speed : playback speed
        h_range : sensitivity in HSV H-value for tracking red color
        play : if True, plays video on screen
    Return:
        z : 2xN NumPy array of z_x, z_y for each frame
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
            # if no red on screen, set mesurement -1 (invalid pos)
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


def get_and_play(file = 'annoying_bird.mov', speed = 1):
    """
    Basic video player.
    """

    cap = cv2.VideoCapture(file)
    
    # 1x frame rate = 60
    fps = 60*speed
    # time between frames
    t = int(1000/fps)

    while True:
        # ret: Bool, True if frame successfully opened
        # frame: NumPy array of image frame
        ret, frame = cap.read()
        if not ret: # if eg. end of video
            break

        frame = cv2.GaussianBlur(frame, (15, 15), 0)  
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        
        # show frame in window "Frame"
        cv2.imshow("Frame", frame)

        # wait t ms or until esc is pressed
        key = cv2.waitKey(t)
        if key == 27: # esc = 27
            break

    cap.release()
    cv2.destroyAllWindows()



#get_and_play('annoying_bird.mov', 0.5)

#z = get_measurements('annoying_bird.mov', 1, 5, False)
#print(z)
#print(z.shape)