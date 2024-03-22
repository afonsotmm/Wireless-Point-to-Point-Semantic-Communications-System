#https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
from math import log10, sqrt 
import cv2 
import numpy as np 
  
def PSNR(original, generated_resize): 
    mse = np.mean((original - generated_resize) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 
  
def main(): 
    for j in range (1,71):
        original = cv2.imread(f'C:\\Users\\Afonso Mateus\\OneDrive\\Área de Trabalho\\teste\\Image_ori_{j}.png') 
        generated = cv2.imread(f'C:\\Users\\Afonso Mateus\\OneDrive\\Área de Trabalho\\teste\\image_{j}.png') 

        width = 450
        height = 470
        dim = (width, height)
        original_resize = cv2.resize(original, dim, interpolation = cv2.INTER_AREA)
        generated_resize = cv2.resize(generated, dim, interpolation = cv2.INTER_AREA)
        print('Resized Dimensions : ', original_resize.shape)
        print('Resized Dimensions : ', generated_resize.shape)
        value = PSNR(original_resize, generated_resize) 
        print(f"PSNR value is {value} dB") 
       
if __name__ == "__main__": 
    main() 