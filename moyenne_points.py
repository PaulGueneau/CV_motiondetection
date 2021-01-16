import cv2
import numpy as np




def pos_moyenne_nuage_points(vid_str):
    cap = cv2.VideoCapture(vid_str)

    ret, first_frame = cap.read()
    resize_dim = 900
    max_dim = max(first_frame.shape)
    scale = resize_dim/max_dim
    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)

    dx_list=[0]
    dy_list=[0]

    nb_pts = 300
    old_mean_x = None
    old_mean_y = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if(ret):
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

            corners = cv2.goodFeaturesToTrack(gray,nb_pts,0.001,10)

            mean_x = np.mean(corners[:,0,0])
            mean_y = np.mean(corners[:,0,1])

            if old_mean_x is None:
                old_mean_x = mean_x
                old_mean_y = mean_y
                continue

            dx = mean_x - old_mean_x
            dy = mean_y - old_mean_y
    
            dx_list.append(dx)
            dy_list.append(dy)

            old_mean_x = mean_x
            old_mean_y = mean_y


            grayRGB = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

            for [[x,y]] in corners:
                grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (0,0,255), 4) #red

          

            grayRGB	= cv2.arrowedLine(	grayRGB, (200,200), (int(200+1*dx),int(200+1*dy)), (0,255,0),thickness=3)

            cv2.imshow(".", grayRGB)
        


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


