import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim



def calculate_ssim(original, generated):

  gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
  gray_generated = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)

  return ssim(gray_original, gray_generated, data_range=original.max())

def main():
  for j in range(0, 71):
    original_path = f"Image_ori_{j}.png"
    generated_path = f"image_{j}.png"

    original = cv2.imread(original_path)
    generated = cv2.imread(generated_path)

    width = 470
    height = 450
    dim = (width, height)

    original_resize = cv2.resize(original, dim, interpolation=cv2.INTER_AREA)
    generated_resize = cv2.resize(generated, dim, interpolation=cv2.INTER_AREA)

    ssim_value = calculate_ssim(original_resize, generated_resize)

    print(f"SSIM value for image {j}: {ssim_value}")


if __name__ == "__main__":
  main()



