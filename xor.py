import numpy as np
import image_functions
import cv2
import sys

i = 125
f = 223
for x in range(i,f):
    # Load an color image in grayscale
    img = cv2.imread('images/'+str(x)+'.png')
    img = image_functions.new_select_zone(img)
    image_functions.save_image(img, "pruebas/"+str(x))