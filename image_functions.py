import numpy as np
import scipy
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import resize
from PIL import Image
import cv2

light_color = 200
dark_color = [66,74]
tam_percent = 0.25
directory = "./images/"

def rgb_2_grayscale(ob):
    img = cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY)
    return img[:404]

def resize_image(img):
    img = scipy.misc.imresize(img, tam_percent)
    # Cambiar aquí para valores finales
    img[img < 200] = dark_color[0]
    img[img > 200] = light_color
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
    img[img == dark_color[1]] = dark_color[0]
    img[img != dark_color[0]] = light_color

    # Shapes
    m = img.shape[0]
    n = img.shape[1]

    # Arriba y Abajo, works
    flag = False
    cont = 0
    for i in range(n):
        if img[0][i] == dark_color[0]:
            cont += 1
            if cont == 2:
                flag = True
            elif cont == 4:
                cont = 0
                flag = False
        if flag:
            # Arriba
            img[0][i] = dark_color[0]
            # Abajo
            img[m-1][i] = dark_color[0]

    for i in range(m):
        img[i][0] = dark_color[0]
        img[i][n-1] = dark_color[0]
    # Se pasa a binario y después se inverte otra vez
    img[img == dark_color[0]] = 1
    img[img == light_color] = 0

    #return img

    # Se filean los holes
    img = binary_fill_holes(img).astype(int)
    #return img

    # Se invierte la jugada
    img[img == 1] = dark_color[0]
    img[img == 0] = light_color
    return img