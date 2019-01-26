import os
import sys
import random
import math
import numpy as np
import skimage.io
import cv2

# import matplotlib
# import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def h_forpng1(img_src, img_mask):
    # img_mask = np.zeros_like(img_back)
    # img_mask = img_back[:, :, :3]
    # print(img_src.shape)
    img_src = np.concatenate((img_src, 255 * np.ones((img_src.shape[0], img_src.shape[1], 1))), axis=2)
    # print(img_src.shape)
    print(img_mask.shape)
    img_mask = img_mask.reshape((img_mask.shape[0], img_mask.shape[1], 1))
    img_mask_stack = np.concatenate((img_mask, img_mask, img_mask, img_mask), axis=2)
    img_back_zero = img_src.copy()
    img_back_zero[:, :, 3] = 0
    img_ret = np.where(img_mask_stack == 1, img_src, img_back_zero)

    return img_ret


# TODO: 顺序
path = "./img1/"
for idx, file in enumerate(os.listdir(path)):
    # Load a random image from the images folder
    # file_names = next(os.walk(IMAGE_DIR))[2]
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(path + file)
    image = image[:, :, :3]

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]

    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                             class_names, r['scores'])

    # BGR RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_ret = h_forpng1(image, r['masks'])
    # img_ret = img_ret[:, :, ::-1]
    # img_ret = cv2.cvtColor(img_ret, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./out1/%04d.png" % idx, img_ret)
    # skimage.io.imsave("./1.png", img_ret)
