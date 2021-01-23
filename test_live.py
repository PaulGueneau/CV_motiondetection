import cv2 
import numpy as np
import queue
from smoothing import smooth
from stabilize import shiftFrame
from stabilize import rogner
from stabilize import fixBorder
from random import randrange
[feature_x,feature_y,simil] = [None,None,None]

def get_dx_dy_live(gray,prev_gray):
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.resize(gray, None, fx=scale, fy=scale)
    corners = cv2.goodFeaturesToTrack(prev_gray,50,0.001,10)
    next_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,gray, corners, None)    
    m, inliers = cv2.estimateAffine2D(corners,next_corners)
    dx = m[0,2]
    dy = m[1,2]
    grayRGB = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

    for [[x,y]] in corners:
        grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (0,0,255), 3) #red

    for [[x,y]] in next_corners:
        grayRGB = cv2.circle(grayRGB, (int(x),int(y)), 2, (255,0,0), 3) #blue

    grayRGB	= cv2.arrowedLine(	grayRGB, (200,200), (int(200+6*dx),int(200+6*dy)), (0,255,0),thickness=3)
    cv2.imshow(".", grayRGB)
    return([dx,dy])

def get_dx_dy_tracking(gray,prev_gray,feature_x,feature_y,simil,cout_max=50):

            #gray = cv2.equalizeHist(gray)
            # on travaille sur gray

            # 1 - trouver une feature dans prev_gray
    edges   =   cv2.Canny(cv2.medianBlur(prev_gray,15), 0, 50)
    edges   =   np.multiply(edges,mask_center) 
    print(feature_x)
    if feature_x is not None and mask_center[feature_x,feature_y]>0 and simil < cout_max:
        
        [chosen_x,chosen_y] = [feature_x,feature_y]
        
    else:
        print('sorti')
        [corners_x,corners_y] = np.where(edges>0)
        r_ = randrange(len(corners_x))
        chosen_x = corners_x[r_]
        chosen_y = corners_y[r_]
    #print("chosen_x,chosen_y : " + str([chosen_x,chosen_y]))
    feature = get_feature(chosen_x,chosen_y,prev_gray,region_size)
    # 2 - la chercher dans gray
    [dx,dy,simil] = find_feature(feature,gray,chosen_x,chosen_y,region_size,dx_dy_max)
    
    print('coût  : '+str(simil))
    #cv2.imshow('Canny',edges)
    #if cv2.waitKey(25) & 0xFF == ord('q'):
     #   break
    
    
    grayRGB = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    grayRGB = cv2.circle(grayRGB, (int(chosen_y+dy),int(chosen_x+dx)), 2, (255,0,0), 3) #red
    grayRGB = cv2.circle(grayRGB, (int(chosen_y),int(chosen_x)), 2, (0,0,150), 3) #blue
    grayRGB =   cv2.rectangle(grayRGB, (int(chosen_y) - region_size ,int(chosen_x) - region_size ), (int(chosen_y)+region_size ,int(chosen_x)+region_size),(0,255,0))
    grayRGB = cv2.arrowedLine(  grayRGB, (200,200), (int(200+3*dy),int(200+3*dx)), (0,255,0),thickness=2)
    cv2.imshow(".", grayRGB)
    #out.write(grayRGB)
    prev_gray = gray
    [feature_x,feature_y] = [chosen_x+dx,chosen_y+dy]
    return([dx,dy,feature,feature_x,feature_y,simil])


def get_feature(chosen_x,chosen_y,img,region_size):
    #print(f'je get la fiture de taille {region_size} dans image de taille {img.shape} aux coords {[chosen_x,chosen_y]}')
    return img[chosen_x-region_size:chosen_x+region_size+1,chosen_y-region_size:chosen_y+region_size+1]

def compute_similarity(img1,img2):
    assert img1.shape == img2.shape
    diff = abs(img2-img1)
    #return (np.sum(diff)) / img1.size #L1
    return (np.sum(np.square(diff))) / img1.size #L2
    #return (np.sum(np.square(np.square(diff)))) / img1.size #L4
def find_feature(feature,gray,chosen_x,chosen_y,region_size,dx_dy_max):
    #region_recherche = get_feature(chosen_x,chosen_y,gray,   region_size+dx_dy_max    ) # !!!!! region_size+dx_dy_max !!!!!
    resultat = list()
    for x in range(-dx_dy_max,dx_dy_max,1): #TODO rajouter 1 pour aller jusqu'au bout du carré
        for y in range(-dx_dy_max,dx_dy_max,1):
            resultat.append([x,y,round(compute_similarity(feature,get_feature(chosen_x+x,chosen_y+y,gray,region_size)))])
    resultat = np.array(resultat)    
    indice_meilleur_dx_dy = list(resultat[:,2]).index(min(list(resultat[:,2])))
    return resultat[indice_meilleur_dx_dy,:]






vid_str = 0

cap = cv2.VideoCapture(vid_str)

ret, first_frame = cap.read()
resize_dim = 900
max_dim = max(first_frame.shape)
scale = resize_dim/max_dim
first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
K = 29
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
dx_queue= queue.Queue(maxsize=K)
dy_queue= queue.Queue(maxsize=K)
queue_frame = queue.Queue(maxsize=(K+1)/2)
frame_count=1
nb_pts = 40
region_size = 11
dx_dy_max = 20

mask_center = np.zeros_like(prev_gray)
(h,w) = mask_center.shape
border = 1* (region_size+dx_dy_max)
    
for i in range(1*border,h-1*border):
    for j in range(1*border,w-1*border):
        mask_center[i,j] = 255


while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("dx"+str(dx_queue.qsize()))
        print("dy"+str(dy_queue.qsize()))
        print("queue_frame"+str(queue_frame.qsize()))
        print('frame'+str(frame_count))
        frame_count=frame_count+1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale)

################### CHOIX METHODE ################
        #[dx,dy] = get_dx_dy_live(gray,prev_gray)
        [dx,dy,feature,feature_x,feature_y,simil] = get_dx_dy_tracking(gray,prev_gray,feature_x,feature_y,simil)
        
        
        
        
        dx_queue.put(dx) 
        dy_queue.put(dy)
        if frame_count>(K+1)/2:
            queue_frame.put(frame)
        cv2.imshow('frame',frame)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
        if frame_count>K:
            

            assert dx_queue.qsize() == K
            ## Laisse passer les premières frames pour pouvoir filtrer par la suite
    
                        
            
            #cv2.imshow('frame',frame)
            
            dx_list = list(dx_queue.queue)
            dy_list = list(dy_queue.queue)
            dx_list_sum = np.cumsum(dx_list)
            dy_list_sum = np.cumsum(dy_list)
        

            dx_list_lisse = smooth(dx_list_sum,K,window='flat')
            dy_list_lisse = smooth(dy_list_sum,K,window='flat')
            err_dx = dx_list_sum - dx_list_lisse
            err_dy = dy_list_sum - dy_list_lisse
            dx = err_dx[int((K+1)/2)]
            dy = err_dy[int((K+1)/2)]
            dx_queue.get(0)
            dy_queue.get(0)  
            xmax = int(max(abs(err_dx))) + 1
            ymax = int(max(abs(err_dy))) + 1 
            frame_stable = queue_frame.get(0)
            shifted_frame = shiftFrame(frame_stable,-dx,-dy)           
            cv2.imshow('stable',shifted_frame)
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
    


        

       



    
    
        
