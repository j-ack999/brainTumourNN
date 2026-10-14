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
import matplotlib.pyplot as mp
#   #   #   #   #   #   #   #   #
SEED = 1
# access data about the training and test sets 
ROOT = "cleaned"
def data_inspection():
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
            print(img.mode) # ^ validated by this line!



# more processing of the data
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # mean and std = 0.5. 

    # xnormalised = (x - 0.5) / 0.5 . 
    # imagine black, its original (x) intensity is 0.0 - we normalise it,  0 - 0.5 / 0.5 = -1. 
    # imaging white, its original (x) intensity is 1.0 - we normalise it, 1 - 0.5 / 0.5 = 1
    # therefore now our pixel intensities have been transformed from [0,1], to [-1,1]

])

def main():
    # data_inspection() **UNCOMMENT AND RUN AT EACH SESSION**
    full_train = datasets.ImageFolder("cleaned/Training", transform=transform)
    indices = list(range(len(full_train)))
    labels = full_train.targets # classification of each image 

    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=labels) # validation set sized at 20% 
    # stratisfy spmply ensures each class has the same proportion in train as it does in the validation set
    
    # print(len(train_idx))
    # print(len(val_idx))


























if __name__ == "__main__":
    main()


