import numpy as np
import scipy
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import resize
from PIL import Image
import cv2

light_color = 200
dark_color = [66,74]
dim = (100,100)
directory = "./images/"
zero = 0
one = 1
white = 255

def rgb_2_grayscale(ob):
    img = cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY)
    return img[:404]

def resize_image(img):
    img = scipy.misc.imresize(img, dim)
    # Cambiar aquí para valores finales
    temp = 999
    img[img > 200] = temp
    img[img < 200] = 0
    img[img == temp] = white
    
    return img

def ob_2_gray(ob):
    img = new_select_zone(ob)
    return resize_image(img)

def save_image(img,name):
    cv2.imwrite(directory+name+".png", img)

def new_select_zone(ob):
    # Dimensiones
    
    # Para dejarlo en dos colores
    img = rgb_2_grayscale(ob)
    img[img == dark_color[0]] = one
    img[img == dark_color[1]] = one
    img[img != one] = zero

    # Shapes
    m = img.shape[0]
    n = img.shape[1]

    # Arriba y Abajo, works
    flag = False
    cont = 0
    for i in range(n):
        if img[0][i] == one:
            cont += 1
            if cont == 2:
                flag = True
            elif cont == 4:
                cont = 0
                flag = False
        if flag:
            # Arriba
            img[0][i] = one
            # Abajo
            img[m-1][i] = one

    for i in range(m):
        img[i][0] = one
        img[i][n-1] = one
    
    # Se filean los holes
    img = binary_fill_holes(img).astype(int)
    
    # Se invierte la jugada
    img[img == 1] = white
    return img