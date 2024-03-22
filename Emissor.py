#Conceptual_captions
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import requests
import PIL.Image
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
#import pandas

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent

USER_AGENT = get_datasets_user_agent() # Caracteríticas do utilizador, por ex o sistema operacional, etc...

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large") 
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large") 

def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries+1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            ) # o request é um objeto com os paramteros necessarios para usar a funçao urllib.request.urlopen(...)
            with urllib.request.urlopen(request, timeout=timeout) as req: # solicitacao para aceder a imagem 
                image = PIL.Image.open(io.BytesIO(req.read())) #Ler os dados brutos da imagem e abrir os mesmo como imagem 
            break # Depois de obtida a imagem interrompemos o loop
        except Exception:
            image = None
    return image

def fetch_images(batch, num_threads, timeout=None, retries=0): # funçao que busca por imagens em paralelo
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries) # definimos uma nova funçao que tem como argumentos a funçao fetch_single_image e usa-se a partial para acrescentar à fetch mais argumentos
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch

image_url_vec = []
raw_image = []
caption = []

num_threads = 20
dset = load_dataset("conceptual_captions")
nAmostraImagens = 100

for i in range(0,nAmostraImagens):
    raw_image = 0
    image_url = dset["train"][i]["image_url"]

    try: 
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB') 

    except:
        print("")

    if(raw_image!=0):
        image_url_vec.append(image_url)
        raw_image.save(f"imagem_{i+1}.png")
        text = "a photography of" 
        inputs = processor(raw_image, text, return_tensors="pt")  
        out = model.generate(**inputs) 
        print(processor.decode(out[0], skip_special_tokens=True)) 
        caption.append(processor.decode(out[0], skip_special_tokens=True))
    




#Panda
'''data = {'Image_': image_url_vec, 'Captions': caption}
df = pandas.DataFrame(data)
print(df)
df.to_csv('C:\\Users\\Afonso Mateus\\OneDrive\\Área de Trabalho\\PI_08_03\\image_captions_PI.csv', index=False)'''

        
      
    
