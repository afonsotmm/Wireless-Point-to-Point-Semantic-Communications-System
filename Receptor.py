from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import requests
import PIL.Image
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
import torch

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

data = pd.read_csv('image_captions_PI', usecols=['Image_','Captions'])

for i, line in data.iterrows():
   prompt = line['Captions']
   image = pipe(prompt).images[0]  
   image.save(f'C:\\Users\\Afonso Mateus\\OneDrive\\Área de Trabalho\\PI_08_03\\Imagens geradas\\image_{i}.png')







