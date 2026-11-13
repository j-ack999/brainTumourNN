#           imports             #
import torchvision
from torch.utils.data import Subset
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import torch
import os 
from collections import Counter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model_architecture import *


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

            print(class_name, sum(sizes.values()), dict(sizes), dict(modes)) # this tells us each set for training has 1400 images and that of testing batch is 400 images
            # image sizes are 256 x 256, with 1400 of each type in the training set and 400 in the test set
            # note also that we are using RGB 
            # print(img.mode) # ^ validated by this line!



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
class BrainTumourDataset(Dataset): # inherit from the dataset class which was imported, then override the methods
        def __init__(self, root, split, transform=None): # prepares something that can be indexed over containing (path,label) for each image
            self.transform = transform
            split_dir = os.path.join(root,split) # e.g pass in 'cleaned','training' will access the cleaned folder, and then the training folder
            class_names = sorted(os.listdir(split_dir)) # access sorted contents of ROOT -> SPLIT (it accesses the names of folders containing the images)
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)} # dictionary comprehension

            self.samples = []
            for class_name, idx in self.class_to_idx.items():
                class_dir = os.path.join(split_dir ,class_name)
                for fname in os.listdir(class_dir):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, idx))

                    # samples is now a list of tuples which contains [path, label]
                    # e.g dataset.samples[0] = 'cleaned\\Training\\Glioma\\Tr-gl_1.jpg', 0
        

        def __len__(self):
             return len(self.samples)

        def __getitem__(self, idx):
             path, label = self.samples[idx]
             image = Image.open(path) # get the image at the current path, from classdir and fname 
             if self.transform is not None:
                  image = self.transform(image) # THIS is us tensor-ising our images. needed for gradients later on 

             return image, label
        
                 

dataset = BrainTumourDataset("cleaned", "Training")
print(dataset.class_to_idx)


def main():
    # data_inspection() # **UNCOMMENT AND RUN AT EACH SESSION**
    full_train = BrainTumourDataset("cleaned", "Training", transform=transform)
    indices = list(range(len(full_train)))
    # list of labels
    labels = [label for path, label in full_train.samples] # for path, label in fulltrainsamples: labels.append(label)
    # print(labels[1])
    # print(dataset.samples[1])
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=labels) # validation set sized at 20% 
    # stratisfy simply ensures each class has the same proportion in train as it does in the validation set
    
    # print(len(train_idx))
    # print(len(val_idx))
    # print(len(dataset.samples)) # how many samples do we have in our cleaned training data? 
    # print(dataset.samples[4444]) # what is the label of x sample? format of path, label, where path is the path in memory 
    

    batch_size = 32

    train_set = Subset(full_train, train_idx) # use full_train but only in the positions specified by train_idx 
    val_set = Subset(full_train, val_idx)

    # use of a validation set and a training set here is not trivial 
    # this is the standard of NN's since we want to test the network to tweak the architecture on a completely different set so the model does not learn from this data


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # no need to shuffle the validation set 
    # note that running trainset[i] gives us (image_tensor, label)

    # list of every class we have 
    classes = []
    for i in dataset.class_to_idx:
        classes.append(i)

        

    
    # Code from pytorch cifar tutorial to visualise the batches 

    def imshow(img):
        img = img / 2 + 0.5     
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(train_loader)
    images, batch_labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels for the batch
    print(' '.join(f'{classes[batch_labels[j]]:5s}' for j in range(batch_size)))


    # pull in the defined network from the other file 

    net = NeuralNetwork()


if __name__ == "__main__":
    main()


