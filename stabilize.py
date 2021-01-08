import cv2
import numpy as np



def stabilize_v1(vid,dx_list,dy_list,smoothing_length):

    # Calculer dx et dy moyens en fonction du temps
    kernel = np.ones((1,smoothing_length))
    dx_list_lisse = np.convolve(dx_list,kernel,mode='same')
    dy_list_lisse = np.convolve(dy_list,kernel,mode='same')

    # En déduire le tremblement à chaque frame

    err_dx = dx_list - dx_list_lisse
    err_dy = dy_list - dy_list_lisse

    print(err_dx,err_dy)

    cap = cv2.VideoCapture(vid)
    # Read first frame
    ret, first_frame = cap.read()
    # Scale and resize image
    resize_dim = 900
    max_dim = max(first_frame.shape)
    print(max_dim)
    scale = resize_dim/max_dim
    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
 



    out = cv2.VideoWriter(vid_str.split('.')[-2] + 'stable.mp4',-1,1,(600, 600))

    nb_pts = 15
    while cap.isOpened():
        # Read a frame from video
        ret, frame = cap.read()
        if(ret):
            first_frame
        
        

    return out
