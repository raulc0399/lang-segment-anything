import os
from PIL import Image
import numpy as np
from models import LangSAMModel
import cv2
import torch

os.makedirs("imgs/output", exist_ok=True)

langsam_model = LangSAMModel()

for img_name in ["1.png", "2.png", "3.png"]:
    langsam_model.process_image(img_name, return_mask=False)
