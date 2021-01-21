import cv2 
import numpy as np
from smoothing import smooth
from stabilize import shiftFrame
from stabilize import rogner
from stabilize import fixBorder

cap = cv2.VideoCapture(0)

ret, first_frame = cap.read()
resize_dim = 900
max_dim = max(first_frame.shape)
scale = resize_dim/max_dim
first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
K = 29
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
dx_list=list()
dy_list=list()
pile = []
pile.append(first_frame)
max_frames = 700
frame_count=1
nb_pts = 40
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret: 
        cv2.imshow('frame',frame)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
            ## Laisse passer les premières frames pour pouvoir filtrer par la suite
        frame_count=frame_count+1
        k= min(K,frame_count-1)
        #cv2.imshow('frame',frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
        corners = cv2.goodFeaturesToTrack(prev_gray,nb_pts,0.001,10)
        #print(corners)
        next_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,gray, corners, None)    
        m, inliers = cv2.estimateAffine2D(corners,next_corners)
        dx = m[0,2]
        dy = m[1,2]
        pile.append(frame)
        dx_list = np.append(dx_list,dx)  
        dy_list = np.append(dy_list,dy)
        dx_list_sum = np.cumsum(dx_list[frame_count-k-1:k+frame_count-k-1])
        dy_list_sum = np.cumsum(dy_list[frame_count-k-1:k+frame_count-k-1])
        dx_list_lisse = smooth(dx_list_sum,k,window='flat')
        dy_list_lisse = smooth(dy_list_sum,k,window='flat')
        err_dx = dx_list_sum - dx_list_lisse
        err_dy = dy_list_sum - dy_list_lisse
        dx = err_dx[k-1]
        dy = err_dy[k-1]
        xmax = int(max(abs(err_dx))) + 1
        ymax = int(max(abs(err_dy))) + 1 
        shifted_frame = shiftFrame(frame,-dx,-dy)           
        rogned = rogner(shifted_frame,xmax,ymax)    
        cv2.imshow('stable',rogned)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
        prev_gray = gray 
     
 
cap.release()
cv2.destroyAllWindows()     

        
                
'''        pile.append(pile[k-1])
                gray = cv2.cvtColor(pile[k-1], cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=scale, fy=scale)
                corners = cv2.goodFeaturesToTrack(prev_gray,nb_pts,0.001,10)
                next_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,gray, corners, None)    
                m, inliers = cv2.estimateAffine2D(corners,next_corners)
                dx = m[0,2]
                dy = m[1,2]
                dx_list = np.append(dx_list,dx)  
                dy_list = np.append(dy_list,dy)
                dx_list_sum = np.cumsum(dx_list[k-(K+1):k])
                dy_list_sum = np.cumsum(dy_list[k-(K+1):k]) 
                dx_list_lisse = smooth(dx_list_sum,K,window='flat')
                dy_list_lisse = smooth(dy_list_sum,K,window='flat')  
                err_dx = dx_list_sum - dx_list_lisse
                err_dy = dy_list_sum - dy_list_lisse
                dx = err_dx[K]
                dy = err_dy[K]
            
            #xmax = int(max(abs(err_dx))) + 1
            #ymax = int(max(abs(err_dy))) + 1 
        
                shifted_frame = shiftFrame(pile[K],-dx,-dy)
                prev_gray = gray            
                rogned = fixBorder(shifted_frame)
                #cv2.imshow("Frame",pile[k-(K+1)])
                cv2.imshow("Stable",rogned)
                    
                key = cv2.waitKey(1)& 0xFF
                if key==ord('q'):
                    break

    '''
    


        

       



    
    
        
