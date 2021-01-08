import cv2
import numpy as np

from stabilize import stabilize_v1

# Get a VideoCapture object from video and store it in vs
vid_str = "../vids/VID_1.mp4"

#cap = cv2.VideoCapture("./vids/Camera fixe/football.mp4")
cap = cv2.VideoCapture(vid_str)


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




#out = cv2.VideoWriter('video.mp4',-1,1,(600, 600))

dx_list=list()
dy_list=list()

nb_pts = 15
while cap.isOpened():
    # Read a frame from video
    ret, frame = cap.read()
    if(ret):
        
    
        # Convert new frame format`s to gray scale and resize gray frame obtained
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
        

        corners = cv2.goodFeaturesToTrack(prev_gray,nb_pts,0.1,40)

        next_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,gray, corners, None)
        
        sum_dx = 0
        sum_dy = 0
        for i in range(len(corners)):
            sum_dx += next_corners[i][0][0] - corners[i][0][0]
            sum_dy += next_corners[i][0][1] - corners[i][0][1]
        dx = sum_dx/nb_pts
        dy = sum_dy/nb_pts

        dx_list.append(dx)
        dy_list.append(dy)


        
        grayRGB = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

        for [[x,y]] in corners:
            grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (0,0,255), 1) #red

        for [[x,y]] in next_corners:
            grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (255,0,0), 1) #blue

        
        grayRGB	= cv2.arrowedLine(	grayRGB, (200,200), (int(200+6*dx),int(200+6*dy)), (0,255,0),thickness=3)


        
        cv2.imshow(".", grayRGB)
        #out.write(dense_flow)
        prev_gray = gray
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1) #wait until any key is pressed
    else:
        break
     


print(dx_list,dy_list)

cap.release()

cv2.destroyAllWindows()

stabilize_v1(vid_str,dx_list,dy_list,15)

