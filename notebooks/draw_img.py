import sys
#sys.path.append(r"C:\Users\multimaster\documents\retinaFace_dilab")
#from utils.basic_utils import get_bb, draw_bbox
#from retinaface import RetinaFace
import cv2

#model = RetinaFace.detect_faces

img = r"M:\experiment_351\included\__20240130_10102\cam07_frames_p\img_150.jpg"

#bbox = get_bb(model, img)
#pred = draw_bbox(bbox, img)

img = cv2.imread(img)
pred = cv2.circle(img, (978,508), 5, (0,0,255))

cv2.imshow('', pred)
cv2.waitKey(0)