import cv2
import numpy as np

# Get a VideoCapture object from video and store it in vs
#cap = cv2.VideoCapture("./vids/Camera fixe/football.mp4")
cap = cv2.VideoCapture("../vids/VID_1.mp4")


# Read first frame
ret, first_frame = cap.read()
# Scale and resize image
resize_dim = 900
max_dim = max(first_frame.shape)
print(max_dim)
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
        corners = cv2.goodFeaturesToTrack(gray,200,0.90,12)
        print(corners.shape)

        grayRGB = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

        for [[x,y]] in corners:
            grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (0,0,255), 1)


        cv2.imshow("Dense optical flow", grayRGB)
        #out.write(dense_flow)
        prev_gray = gray
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()

cv2.destroyAllWindows()
exit()

