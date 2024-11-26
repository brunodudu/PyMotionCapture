import cv2 
import numpy as np 
import os 
import glob 
import argparse
import json


parser = argparse.ArgumentParser(description="Calibrar camera a partir de imagens xadrez.")
parser.add_argument("source", type=str, help="Fonte das imagens.")
args = parser.parse_args()
source = args.source
  
  
CHECKERBOARD = (6, 9) 
  
  
criteria = (cv2.TERM_CRITERIA_EPS + 
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
  
  
threedpoints = [] 
twodpoints = [] 
  
  
objectp3d = np.zeros((1, CHECKERBOARD[0]  
                      * CHECKERBOARD[1],  
                      3), np.float32) 
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
                               0:CHECKERBOARD[1]].T.reshape(-1, 2) 
prev_img_shape = None
  
  
images = glob.glob(f'frames_{source}/*.jpg') 
  
for filename in images: 
    image = cv2.imread(filename) 
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
    
    
    
    ret, corners = cv2.findChessboardCorners( 
                    grayColor, CHECKERBOARD,  
                    cv2.CALIB_CB_ADAPTIVE_THRESH  
                    + cv2.CALIB_CB_FAST_CHECK + 
                    cv2.CALIB_CB_NORMALIZE_IMAGE) 
  
    
    
    
    if ret == True: 
        threedpoints.append(objectp3d) 
        corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria) 
        twodpoints.append(corners2) 
        image = cv2.drawChessboardCorners(image,  
                                          CHECKERBOARD,  
                                          corners2, ret) 
  
    rez_img = cv2.resize(image, (800, 600))
    cv2.imshow('img', rez_img) 
    cv2.waitKey(0) 
  
cv2.destroyAllWindows() 
  
h, w = image.shape[:2] 
  
  
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera( 
    threedpoints, twodpoints, grayColor.shape[::-1], None, None) 
  
  
print(" Camera matrix:") 
print(matrix) 
os.makedirs("results", exist_ok=True)
with open(f"results/camera_matrix_{source}.json", "w") as json_file:
    json.dump(matrix.tolist(), json_file)
  
print("\n Distortion coefficient:") 
print(distortion) 
  
print("\n Rotation Vectors:") 
print(r_vecs) 
  
print("\n Translation Vectors:") 
print(t_vecs) 