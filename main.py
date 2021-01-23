from LK import get_dx_dy
from canny import pos_moyenne_canny
from moyenne_points import pos_moyenne_nuage_points
from stabilize import stabilize_v1

# Get a VideoCapture object from video and store it in vs
#vid_str = "../vids/Synth_2.mp4"
vid_str = "VID_1.mp4"
#vid_str = "../vids/smooth_51.avi"
#vid_str = "../vids/vidstab.avi"
#vid_str = 0 # webcam
#vid_str = "vid_test.mp4" 


[dx_list,dy_list] = get_dx_dy(vid_str)





#[dx_list,dy_list] = get_dx_dy_auto(vid_str)

#[dx_list,dy_list] = pos_moyenne_nuage_points(vid_str)

#[dx_list,dy_list] = pos_moyenne_canny(vid_str)

stabilize_v1(vid_str,dx_list,dy_list,59)