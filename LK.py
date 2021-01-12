import cv2
import numpy as np
import matplotlib.pyplot as plt
from stabilize import stabilize_v1
from stabilize import fixBorder
from smoothing import movingAverage,smooth
# Get a VideoCapture object from video and store it in vs
vid_str = "VID_1.mp4"

#cap = cv2.VideoCapture("./vids/Camera fixe/football.mp4")
cap = cv2.VideoCapture(vid_str)

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Read first frame
ret, first_frame = cap.read()
# Scale and resize image
resize_dim = 900
max_dim = max(first_frame.shape)
scale = resize_dim/max_dim
first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

fps = int(cap.get(cv2.CAP_PROP_FPS))
out =cv2.VideoWriter('video_out.mp4', fourcc, fps, (width, height)) 
# Convert to gray scale 
_,prev = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)




#out = cv2.VideoWriter('video.mp4',-1,1,(600, 600))

#dx_list=list()
#dy_list=list()

nb_pts = 350
transforms = np.zeros((n_frames-1,3),np.float32)
frames_tab = []
for i in range(n_frames-2):
    corners = cv2.goodFeaturesToTrack(prev_gray,nb_pts,0.01,30,3)
    # Read a frame from video
    ret, frame = cap.read()
    if(ret):
        
    
        # Convert new frame format`s to gray scale and resize gray frame obtained
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
        
        
        

        next_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,gray, corners, None)

        idx = np.where(status==1)[0]
        corners = corners[idx]
        next_corners = next_corners[idx]

        a = np.array([[corners[0][0][0],-corners[0][0][1],1,0],[corners[0][0][1],corners[0][0][0],0,1],[corners[1][0][0],-corners[1][0][1],1,0],[corners[1][0][1],corners[1][0][0],0,1]])
        b = np.array([next_corners[0][0][0],next_corners[0][0][1],next_corners[1][0][0],next_corners[1][0][1]])
        m = np.linalg.solve(a,b) 
        M = [m[0],-m[1],m[2]],[m[1],m[0],m[3]]
        dx = M[0][2]
        dy = M[1][2]
        da = np.arctan2(M[1][0],M[0][0])
        transforms[i] = [dx,dy,da]
        prev_gray = gray
        trajectory = np.cumsum(transforms,axis=0)
        smooth_trajectory = smooth(trajectory)
        frames_tab.append(i)
        #plt.plot(trajectory)
        '''sum_dx = 0
        sum_dy = 0
        
        sum_dx += next_corners[i][0][0] - corners[i][0][0]
        sum_dy += next_corners[i][0][1] - corners[i][0][1]
        dx = sum_dx/nb_pts
        dy = sum_dy/nb_pts

        dx_list.append(dx)
        dy_list.append(dy)'''
        
        diff_trajectory  = smooth_trajectory - trajectory

        transforms_smooths = transforms + diff_trajectory

        

        
        grayRGB = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

        for [[x,y]] in corners:
            grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (0,0,255), 1) #red

        for [[x,y]] in next_corners:
            grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (255,0,0), 1) #blue

        
        grayRGB	= cv2.arrowedLine(	grayRGB, (200,200), (int(200+6*dx),int(200+6*dy)), (0,255,0),thickness=3)

frames_tab.append(n_frames-2)
'''
plt.figure()
plt.plot(frames_tab,transforms[:,0],'r',label="dx")
plt.plot(frames_tab,transforms[:,1],'g',label="dy")
plt.plot(frames_tab,transforms[:,2],'b',label="da")
plt.legend()
plt.savefig('dx_dy_da')

plt.figure()
plt.plot(frames_tab,trajectory[:,0],'r',label="trajectory_x")
plt.plot(frames_tab,smooth_trajectory[:,0],'b',label="smooth_trajectory_x")
plt.legend()
plt.savefig('Trajectories_x')

plt.figure()
plt.plot(frames_tab,trajectory[:,1],'r',label="trajectory_y")
plt.plot(frames_tab,smooth_trajectory[:,1],'b',label="smooth_trajectory_y")
plt.legend()
plt.savefig('Trajectories_y')'''


cap.set(cv2.CAP_PROP_POS_FRAMES,0)
for i in range(n_frames-2):
    success, frame = cap.read()
    if not success:
        break
    dx= transforms_smooths[i,0]
    dy= transforms_smooths[i,1]  
    da= transforms_smooths[i,2]
    M_t = np.zeros((2,3),np.float32)
    M_t[0,0]= np.cos(da)
    M_t[0,1]= -np.sin(da)
    M_t[1,0]= np.sin(da)
    M_t[1,1]= np.cos(da)
    M_t[0,2]= dx
    M_t[1,2]= dy



    frame_stable = cv2.warpAffine(frame,M_t,(width,height))

    frame_stable = fixBorder(frame_stable)

    frame_out = cv2.hconcat([frame,frame_stable])
    if (frame_out.shape[1] > 1920):
        frame_out = cv2.resize(frame_out,((int(frame_out.shape[1]/4)),int(frame_out.shape[0])))

    cv2.imshow(".", frame_out)
        #out.write(dense_flow)

    k= cv2.waitKey(10)
    if k == ord('q'):
        break
     


    out.write(frame_out)
    '''key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed
    else:
        break'''
     

cap.release()

cv2.destroyAllWindows()

#stabilize_v1(vid_str,dx_list,dy_list,51)

