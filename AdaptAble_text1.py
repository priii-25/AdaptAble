from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

def image_to_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    image_path = "/content/autistic test1.jpg"
    caption = image_to_caption(image_path)
    print("Generated Caption:")
    print(caption)