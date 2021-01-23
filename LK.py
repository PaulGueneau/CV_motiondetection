import cv2
import numpy as np
import matplotlib.pyplot as plt
from smoothing import smooth
from stabilize import shiftFrame
from stabilize import rogner

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

    nb_pts = 50
    while cap.isOpened():
        ret, frame = cap.read()
        if(ret):
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
            

            #cv2.imshow('Frame',gray)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            corners = cv2.goodFeaturesToTrack(prev_gray,nb_pts,0.001,10)
            #print(corners)
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
               grayRGB_c = cv2.circle(grayRGB, (int(x),int(y)), 2, (0,0,255), 3) #red

            for [[x,y]] in next_corners:
                grayRGB_nc = cv2.circle(grayRGB, (int(x),int(y)), 2, (255,0,0), 3) #blue

            grayRGB	= cv2.arrowedLine(	grayRGB, (200,200), (int(200+6*dx),int(200+6*dy)), (0,255,0),thickness=3)






            #cv2.imshow(".", grayRGB)
            #cv2.imwrite("Frames_features/features_c.jpg",grayRGB_c)
            cv2.imwrite("Frames_features/match.jpg",grayRGB)
            prev_gray = gray
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(-1) 
        else:
            break
        
    #Frames = [i for i in range(len(dx_list))]
    cap.release()
    cv2.destroyAllWindows()
    #plt.plot(Frames,dx_list,'r',label="dx")
    #plt.plot(Frames,dy_list,'b',label="dy")
    #plt.legend()
    #plt.savefig('figures/dx_dy_da')


    return [np.cumsum(dx_list),np.cumsum(dy_list)]







'''
def get_dx_dy_auto(vid_str):
    cap = cv2.VideoCapture(vid_str)
    ret, first_frame = cap.read()
    resize_dim = 900
    max_dim = max(first_frame.shape)
    scale = resize_dim/max_dim
    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    dx_list=list()
    dy_list=list()
    frame_count=0
    nb_pts = 15
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame',frame)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
           break

        for i in range(20):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
            corners = cv2.goodFeaturesToTrack(prev_gray,nb_pts,0.001,10)
            print(corners)
            next_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,gray, corners, None)    
            m, inliers = cv2.estimateAffine2D(corners,next_corners)
            dx = m[0,2]
            dy = m[1,2]

            dx_list.append(dx)
            dy_list.append(dy)
            #cv2.imshow('Frame',gray)

            #if cv2.waitKey(30) & 0xFF == ord('q'):
            #    break
            
                
            prev_gray = gray    
            i=i+1

                
                
        dx_list_lisse = smooth(dx_list,17)
        dy_list_lisse = smooth(dy_list,17)

        err_dx = dx_list - dx_list_lisse
        err_dy = dy_list - dy_list_lisse
        xmax = int(max(abs(err_dx))) + 1

        ymax = int(max(abs(err_dy))) + 1
        dx = err_dx[frame_count]
        dy = err_dy[frame_count]

        grayRGB = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

            #for [[x,y]] in corners:
            #    grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (0,0,255), 3) #red

            #for [[x,y]] in next_corners:
            #    grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (255,0,0), 3) #blue

            #grayRGB	= cv2.arrowedLine(	grayRGB, (200,200), (int(200+6*dx),int(200+6*dy)), (0,255,0),thickness=3)
        shifted_frame = shiftFrame(frame,-dx,-dy)
            
        rogned = rogner(shifted_frame,xmax,ymax)
        cv2.imshow("Stable",rogned)
            #cv2.imshow(".", grayRGB)
            
            #frame_count +=1
        key = cv2.waitKey(20)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1) 
        else:
            break
        

    cap.release()
    cv2.destroyAllWindows()

    return [np.cumsum(dx_list),np.cumsum(dy_list)]
'''


























































































'''plt.figure()
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
plt.savefig('figures/Trajectories_y_ms')'''




