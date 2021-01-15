from vidstab import VidStab
import cv2

mon_stabilizer = VidStab()
mon_stabilizer.stabilize(input_path='../vids/VID_1.mp4',output_path='vidstab.mp4')
