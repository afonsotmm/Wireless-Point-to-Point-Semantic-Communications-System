#pip install image_similarity_measures

from image_similarity_measures.quality_metrics import psnr

import cv2
for i in range (0,71):
  try:
    original_path = f"Image_ori_{i}.png"
    generated_path = f"image_{i}.png"

    original = cv2.imread(original_path)
    generated = cv2.imread(generated_path)

    width = 470
    height = 450
    dim = (width, height)

    original_resize = cv2.resize(original, dim, interpolation=cv2.INTER_AREA)
    generated_resize = cv2.resize(generated, dim, interpolation=cv2.INTER_AREA)

    print(f"PSNR value for images {i} {psnr(org_img=original_resize, pred_img=generated_resize)}")
  except:
    print("Erro")
