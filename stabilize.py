import cv2
import numpy as np

from smoothing import smooth



def stabilize_v1(vid,dx_list,dy_list,smoothing_length):

    dx_list_lisse = smooth(dx_list,window_len=smoothing_length)
    dy_list_lisse = smooth(dy_list,window_len=smoothing_length)


    # Calculer dx et dy moyens en fonction du temps
    # kernel = np.ravel((1/smoothing_length) * np.ones((1,smoothing_length)))
    # dx_list_lisse = np.convolve(dx_list,kernel,mode='same')
    # dy_list_lisse = np.convolve(dy_list,kernel,mode='same')

    # En déduire le tremblement à chaque frame

    err_dx = dx_list - dx_list_lisse
    err_dy = dy_list - dy_list_lisse

    xmax = int(max(abs(err_dx))) + 1

    ymax = int(max(abs(err_dy))) + 1



    cap = cv2.VideoCapture(vid)
    # Read first frame
    ret, first_frame = cap.read()
    # Scale and resize image
    resize_dim = 900
    max_dim = max(first_frame.shape)
    scale = resize_dim/max_dim
    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter(vid.split('.')[-2] + str(smoothing_length) +'_frames_stable.avi',fourcc, fps,(900, 900))
    out = cv2.VideoWriter('../vids/NEW_smooth_'+ vid.split('.')[-2].split('/')[-1] +'_'+ str(smoothing_length) + '.avi' ,fourcc, fps,(first_frame.shape[1],first_frame.shape[0]))

    
    frame_count = 0
    while cap.isOpened():
        # Read a frame from video
        ret, frame = cap.read()
        if(ret):
            dx = err_dx[frame_count]
            dy = err_dy[frame_count]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            shifted_frame = shiftFrame(frame,-dx,-dy)
            
            rogned = rogner(shifted_frame,xmax,ymax)
            # zoomer ? 
            # ecrire dans out
            out.write(rogned)

            frame_count +=1
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return 0




def shiftFrame(img,tx,ty):
    h, w = img.shape[:2] 
    T = np.float32([[1, 0, tx], [0, 1, ty]])
    img_translation = cv2.warpAffine(img, T, (w, h))
    return(img_translation)


def rogner(frame,x,y):

    h,w,_ = frame.shape
    frame[:,0:x,:] = 0
    frame[:,w-x:w,:] = 0
    frame[0:y,:,:] = 0
    frame[h-y:h,:,:] = 0
    return frame