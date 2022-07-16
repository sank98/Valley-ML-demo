import cv2
import numpy as np
import dlib
from test import evaluate
import os
from skimage import exposure

def resize256(img):
  return cv2.resize(img, (256, 256), interpolation = cv2.INTER_NEAREST)

def affine_pts_gen(dlm, sid, dest_mask):
  pts = pd.read_csv('pts.csv')
  #3 Points for affine tranformation source
  dest_mask = dest_mask.astype('uint8')
  dest_mask_inv = cv2.bitwise_not(dest_mask)
  sp1,sp2,sp3 = eval(pts[sid][0]), eval(pts[sid][1]), eval(pts[sid][2]) 
  sratio = self.face_ratio((sp1,sp2,sp3))
  
  #Calulating points for affine transformation for destination
  tdp1, tdp2 = dlm[1], dlm[15]
  dp1, dp2,= dlm[2], dlm[14]
  x = dlm[27][0]
  y = dlm[1][1]
  while(dest_mask_inv[y][x]):
      y = y-1
  y=y+20
  dp3=x,y
  dw = tdp2[0]-tdp1[0]
  h_opt = int(dw*sratio)
  y = dp3[1]+h_opt

  wl = dp1[0]-tdp1[0]
  hs = y - tdp1[1] 
  hl = dp1[1] - tdp1[1]
  x = dp1[0]+int(wl*hs/hl)
  tdp1 = (x,y)

  wl = dp2[0]-tdp2[0]
  hs = y - tdp2[1]
  hl = dp2[1] - tdp2[1]
  x = dp2[0]+int(wl*hs/hl)
  tdp2 = (x,y)
  
  dp1,dp2 = tdp1,tdp2
  
  #If hairs in forehead style, these points are taken for transformation
  if sid in ['hs1','hs12','hs16','hs20']:
      dp1, dp2, dp3 = dlm[1], dlm[15], dlm[8]

  return np.float32([sp1, sp2, sp3]), np.float32([dp1, dp2, dp3])

def transform_source(img, slm, dlm):
  try:
      rows, cols = img.shape
      ch=1
  except:
      rows, cols, ch = img.shape
  pts1 = np.float32([slm[1], slm[15], slm[8]])
  pts2 = np.float32([dlm[1], dlm[15], dlm[8]])
  M = cv2.getAffineTransform(pts1, pts2)
  dst = cv2.warpAffine(img, M, (cols, rows))
  return dst

def lm_gen(img):
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #Detects all faces in the input image
  faces = detector(gray)
  lm=[]

  #For each face, detect landmarks
  for face in faces:
      landmarks = predictor(gray, face)
      for n in range(0, 68):
          x = landmarks.part(n).x
          y = landmarks.part(n).y
          lm.append((x, y))
  return lm

def maskgen(img, idd):
    #Generating hair mask
    #parsing = segmentation().Bisenet_seg(img, idd)
    cv2.imwrite(str(idd)+'.jpg', img)
    cp = 'models/79999_iter.pth'
    #Running mask_gen.py file on the saved image
    parsing = evaluate(str(idd)+'.jpg', cp)
    os.remove(str(idd)+'.jpg')
    parsing = cv2.resize(parsing, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros_like(parsing).astype('uint8')
    mask[parsing==17] = 255

    #Generating background mask
    mask_bgd = np.zeros_like(parsing).astype('uint8')
    mask_bgd[parsing<=0] = 255
    mask_bgd[parsing>18] = 255

    #Generating skin mask
    mask_skin = np.zeros_like(parsing).astype('uint8')
    mask_skin[parsing>0] = 255
    mask_skin[parsing>16] = 0

    mask_inv = cv2.bitwise_not(mask)
    return(mask, mask_inv, mask_bgd, mask_skin)

def mask_smoothening(mask, sigmax=7.3, sigmay=7.3):
  blur = cv2.GaussianBlur(mask, (0,0), sigmaX=sigmax, sigmaY=sigmay, borderType = cv2.BORDER_DEFAULT)
  res = exposure.rescale_intensity(blur, in_range=(224,225), out_range=(0,255))
  res = res.astype('uint8')
  return res