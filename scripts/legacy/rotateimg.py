# usage
# python3 rotateimg.py
# https://www.pyimagesearch.com/2015/02/02/just-open-sourced-personal-imutils-package-series-opencv-convenience-functions/
# i don't see the angels specify, are they random?
import numpy as np
import argparse
import imutils
import cv2


endimg = 100
succeed = 0
for i in range(endimg):
    if i < 9:
        imageID = "00"+str(i+1)+".jpg"
    elif i < 99:
        imageID = "0"+str(i+1)+".jpg"
    else:
        imageID = str(i+1)+".jpg"
    image = cv2.imread(imageID)
    print("[processing]"+imageID)
    for j in range(8):
        rotateID = imageID[:3]+str(j)+imageID[3:]
        rotated = imutils.rotate_bound(image, 45*j)
        cv2.imwrite(rotateID, imutils.resize(rotated, height=500))
        succeed = succeed+1

print("Total image processed:{}, image total(origin included):{}".format(endimg, succeed))
