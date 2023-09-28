import torch
from PIL import Image

from lavis.models import (load_model_and_preprocess,
                          load_model_and_preprocess_my)

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load sample image
img_path = "/content/sample20230919/images/train/chunk_7043_12.jpg"
raw_image = Image.open(img_path).convert("RGB")

checkpoint = "/content/checkpoint_best.pth"
model, vis_processors, _ = load_model_and_preprocess_my(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device, checkpoint=checkpoint)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# generate caption
caption = model.generate({"image": image}, max_length=200)
print("caption: ", caption)