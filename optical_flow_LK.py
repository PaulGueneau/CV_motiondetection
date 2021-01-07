import cv2
import numpy as np

# Get a VideoCapture object from video and store it in vs
cap = cv2.VideoCapture("basketball.mp4")
# Read first frame
ret, first_frame = cap.read()
# Scale and resize image
resize_dim = 600
max_dim = max(first_frame.shape)
scale = resize_dim/max_dim
first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
# Convert to gray scale 
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)


# Create HSV Space
hsv = np.zeros_like(first_frame)

# Sets image saturation to maximum
hsv[..., 1] = 255


#out = cv2.VideoWriter('video.mp4',-1,1,(600, 600))

while(cap.isOpened()):
    # Read a frame from video
    ret, frame = cap.read()
    if(ret):
        
    
    # Convert new frame format`s to gray scale and resize gray frame obtained
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale)
    cv2.imshow("frame", gray)
    # Calculate dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
 

    hsv[..., 0] = angle * 180 / np.pi
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print(hsv)
    # Resize frame size to match dimensions
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    
    # Open a new window and displays the output frame
    dense_flow = cv2.addWeighted(frame, 1,rgb, 4, 0)
    cv2.imshow("Dense optical flow", dense_flow)
    #out.write(dense_flow)
    prev_gray = gray
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()


