import cv2
import numpy as np



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
                grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (0,0,255), 3) #red

            for [[x,y]] in next_corners:
                grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (255,0,0), 3) #blue

            grayRGB	= cv2.arrowedLine(	grayRGB, (200,200), (int(200+6*dx),int(200+6*dy)), (0,255,0),thickness=3)

            cv2.imshow(".", grayRGB)
        
            prev_gray = gray
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(-1) 
        else:
            break
        

    cap.release()
    cv2.destroyAllWindows()

    return [np.cumsum(dx_list),np.cumsum(dy_list)]


