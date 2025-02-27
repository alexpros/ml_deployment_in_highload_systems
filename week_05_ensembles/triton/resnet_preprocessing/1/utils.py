import numpy as np
from PIL import Image

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

def resize_with_aspect_ration(image, target_size=224):
    original_width, original_height = image.size

    if original_width < original_height:
        new_width = target_size
        new_height = int(original_height * (target_size / original_width))
    else:
        new_height = target_size
        new_width = int(original_width * (target_size / original_height))

    resized_image = image.resize((new_width, new_height), Image.BICUBIC)

    return resized_image

def center_crop(image, target_size):
    original_width, original_height = image.size


    left = (original_width - target_size) / 2
    top = (original_height - target_size) / 2
    right = (original_width + target_size) / 2
    bottom = (original_height + target_size) / 2

    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image

def normalize_image(image, MEAN, STD):
    return (image / 255 - MEAN) / STD



def preproc_images(image_filenames, target_size=224):
    preprocessed_images = []
    for i, filename in enumerate(image_filenames):
        image = Image.open(filename)
        resized_image = resize_with_aspect_ration(image, target_size)
        cropped = center_crop(resized_image, target_size)
        norm = normalize_image(np.asarray(cropped), MEAN, STD)
        preprocessed_images.append(norm.transpose((2,0,1)))
    return preprocessed_images
