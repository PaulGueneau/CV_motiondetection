import cv2
import numpy as np
<<<<<<< HEAD
import matplotlib.pyplot as plt
from stabilize import stabilize_v1
from stabilize import fixBorder
from smoothing import movingAverage,smooth
# Get a VideoCapture object from video and store it in vs
vid_str = "vid_test.mp4"

#cap = cv2.VideoCapture("./vids/Camera fixe/football.mp4")
cap = cv2.VideoCapture(vid_str)

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Read first frame

# Scale and resize image

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

fps = int(cap.get(cv2.CAP_PROP_FPS))
out =cv2.VideoWriter('video_out_2.avi', fourcc, fps, (width, height)) 

ret, first_frame = cap.read()
# Convert to gray scale first frame 
_,prev = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)




#out = cv2.VideoWriter('video.mp4',-1,1,(600, 600))

#dx_list=list()
#dy_list=list()

nb_pts = 15
dx_list = list()
dy_list = list()
transforms = np.zeros((n_frames-1,2),np.float32)
frames_tab = []
stability = np.zeros((n_frames-1,2),np.float32)
#for i in range(n_frames-2):
while cap.isOpened():
    ret, frame = cap.read()
    if(ret):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(prev_gray,nb_pts,0.01,nb_pts,7,useHarrisDetector=1)
        next_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,gray, corners, None)
        idx = np.where(status==1)[0]
        corners = corners[idx]
        next_corners = next_corners[idx]
    # Read a frame from video
    
    
        
    
        # Convert new frame format`s to gray scale and resize gray frame obtained
        
       # gray = cv2.resize(gray, None, fx=scale, fy=scale)
        
    
=======



def get_dx_dy(vid_str):
    cap = cv2.VideoCapture(vid_str)

    ret, first_frame = cap.read()
    resize_dim = 900
    max_dim = max(first_frame.shape)
    scale = resize_dim/max_dim
    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    dx_list=list()
    dy_list=list()

    nb_pts = 15
    while cap.isOpened():
        ret, frame = cap.read()
        if(ret):
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
            

            cv2.imshow('Frame',gray)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            corners = cv2.goodFeaturesToTrack(prev_gray,nb_pts,0.001,10)
            print(corners)
            next_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,gray, corners, None)
>>>>>>> afe5ef781257b253923f3bc19b327e3967a9e96c
        
            sum_dx = 0
            sum_dy = 0
            for i in range(len(corners)):
                sum_dx += next_corners[i][0][0] - corners[i][0][0]
                sum_dy += next_corners[i][0][1] - corners[i][0][1]
            dx = sum_dx/nb_pts
            dy = sum_dy/nb_pts

<<<<<<< HEAD

       
        #grayRGB = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

        #for [[x,y]] in corners:
         #   grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (255,0,0), 1) #red

        #for [[x,y]] in next_corners:
          #  grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (0,0,255), 1) #blue

        
        mean_square,c = np.linalg.lstsq(corners.reshape(-1,corners.shape[-1]),next_corners.reshape(-1,next_corners.shape[-1]),rcond=None)[0]

        m_th, inliers = cv2.estimateAffinePartial2D(corners, next_corners)
        #M = [m[0],-m[1],m[2]],[m[1],m[0],m[3]]
        dx = m[0,2]
        dy = m[1,2]
        #dx_ms = mean_square[0] + c[0]
        #dy_ms = mean_square[1] + c[1]
        #da = np.arctan2(m_th[1][0],m_th[0][0])
       # grayRGB	= cv2.arrowedLine(grayRGB, (200,200), (int(200+6*dx_ms),int(200+6*dy_ms)), (0,255,0),thickness=3)
        #transforms[i] = [dx,dy]
        

        frames_tab.append(i)
        dx_list.append(dx)
        dy_list.append(dy)
        trajectory = np.cumsum(transforms,axis=0)
        smooth_trajectory = smooth(trajectory)
        
        
        diff_trajectory  = smooth_trajectory - trajectory

        transforms_smooths = transforms + diff_trajectory

        prev_gray = gray 

frames_tab.append(n_frames-2)


cap.set(cv2.CAP_PROP_POS_FRAMES,0)        
#for i in range(n_frames-2):
    success, frame = cap.read()
    if not success:
        break
    dx= transforms_smooths[i,0]
    dy= transforms_smooths[i,1]  
    #da= transforms_smooths[i,2]
    M_t = np.zeros((2,3),np.float32)
    #M_t[0,0]= np.cos(da)
    #M_t[0,1]= -np.sin(da)
    #M_t[1,0]= np.sin(da)
    #M_t[1,1]= np.cos(da)
    M_t[0,0]= 1
    M_t[0,1]= 0
    M_t[0,2]= dx
    M_t[1,0]= 0
    M_t[1,1]= 1
    M_t[1,2]= dy

 
    
   

    frame_stable = cv2.warpAffine(frame,M_t,(width,height))

    frame_stable = fixBorder(frame_stable)

    
    frame_out = cv2.hconcat([frame,frame_stable])
    if (frame_out.shape[1] == 3840):
        frame_out = cv2.resize(frame_out,((int(frame_out.shape[1]/2)),int(frame_out.shape[0])))

    elif (frame_out.shape[1] == 7680):
        frame_out = cv2.resize(frame_out,((int(frame_out.shape[1]/4)),int(frame_out.shape[0])))
=======
            dx_list.append(dx)
            dy_list.append(dy)

            grayRGB = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

            for [[x,y]] in corners:
                grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (0,0,255), 3) #red

            for [[x,y]] in next_corners:
                grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (255,0,0), 3) #blue
>>>>>>> afe5ef781257b253923f3bc19b327e3967a9e96c

            grayRGB	= cv2.arrowedLine(	grayRGB, (200,200), (int(200+6*dx),int(200+6*dy)), (0,255,0),thickness=3)

<<<<<<< HEAD

    if (frame_out.shape[0] > 1080):
        frame_out = cv2.resize(frame_out,((int(frame_out.shape[1]),int(frame_out.shape[0]/2))))

        
    cv2.imshow('.',frame_out)    
    cv2.waitKey(20)
    out.write(frame_out)


plt.figure()
plt.plot(frames_tab,transforms[:,0],'r',label="dx")
plt.plot(frames_tab,transforms[:,1],'g',label="dy")
#plt.plot(frames_tab,transforms[:,2],'b',label="da")
plt.legend()
plt.savefig('figures/dx_dy_da')

plt.figure()
plt.plot(frames_tab,trajectory[:,0],'r',label="trajectory_x")
plt.plot(frames_tab,smooth_trajectory[:,0],'b',label="smooth_trajectory_x")
plt.legend()
plt.savefig('figures/Trajectories_x_ms')

plt.figure()
plt.plot(frames_tab,trajectory[:,1],'r',label="trajectory_y")
plt.plot(frames_tab,smooth_trajectory[:,1],'b',label="smooth_trajectory_y")
plt.legend()
plt.savefig('figures/Trajectories_y_ms')




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
        frame_out = cv2.resize(frame_out,((int(frame_out.shape[1]/2)),int(frame_out.shape[0])))
    #cv2.imshow("Corners",grayRGB)
    cv2.imshow(".", frame_out)
        #out.write(dense_flow)

    cv2.waitKey(10)
    out.write(frame_out)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed
    else:
        break
 
=======
            cv2.imshow(".", grayRGB)
        
            prev_gray = gray
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(-1) 
        else:
            break
        
>>>>>>> afe5ef781257b253923f3bc19b327e3967a9e96c

    cap.release()
    cv2.destroyAllWindows()

    return [np.cumsum(dx_list),np.cumsum(dy_list)]

<<<<<<< HEAD
#stabilize_v1(vid_str,dx_list,dy_list,51)'''
=======

>>>>>>> afe5ef781257b253923f3bc19b327e3967a9e96c
