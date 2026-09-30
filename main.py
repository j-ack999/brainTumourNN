#           imports             #
from torchvision import datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import torch
import os 
from collections import Counter
from PIL import Image
import numpy as np
#   #   #   #   #   #   #   #   #

# access data about the training and test sets 
ROOT = "cleaned"

for split in ("Training", "Testing"): # access the split (training and testing) and then go deeper to the files. so we have split - tumour class - file 
    print(split)
    for class_name in os.listdir(os.path.join(ROOT, split)):
        class_dir = os.path.join(ROOT, split, class_name)
        sizes, modes = Counter(), Counter()

        for fname in os.listdir(class_dir):
            img = Image.open(os.path.join(class_dir, fname))
            sizes[img.size] += 1 
            modes[img.mode] += 1 

        print(class_name, sum(sizes.values()), dict(sizes), dict(modes))
        # image sizes are 256 x 256, with 1400 of each type in the training set and 400 in the test set
        # note also that we are using RGB 