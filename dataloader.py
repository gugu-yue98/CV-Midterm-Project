import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.utils.data
from torch.utils.data import Subset, DataLoader
import os
import numpy as np
import random


class PennnFudanDataset(Dataset):
    def __init__(self,root, transform=None):

        self.transform = transform
        self.root = root

        # root paths
        self.rootImages = os.path.join(self.root, "PNGImages")
        self.rootMasks = os.path.join(self.root, "PedMasks")
        # self.rootAnnotation = os.path.join(self.root, "Annotation")

        # list of data paths
        self.imagesPaths = sorted( os.listdir(self.rootImages) )
        self.masksPaths  = sorted( os.listdir(self.rootMasks) )
        # self.annotationPaths  = sorted( os.listdir(self.rootAnnotation))

        self.imagesPaths = [ os.path.join(self.rootImages, image) for image in self.imagesPaths ]
        self.masksPaths = [ os.path.join(self.rootMasks, mask) for mask in self.masksPaths ]


    def __getitem__(self, index):

        # load image & mask
        image = Image.open(self.imagesPaths[index])
        mask  = Image.open(self.masksPaths[index])

        image = image.convert("RGB")

        # We get the boxes from the masks instead of reading it from a CSV file

        # get list of object IDs (Pedestrians in the mask)
        # ex: if mask has 3 people in it, IDs = [0, 1, 2, 3] ... 0 for background and 1,2,3 for each pedestrian
        IDs = np.unique(np.array(mask))
        # remove the background ID
        IDs = IDs[1:]

        # transpose it to (N,1,1) to be similar to a column vector
        IDs = IDs.reshape(-1,1,1)

        # extract each mask from the IDs 
        masks = np.array(mask) == IDs

        # N Boxes
        N = len(IDs)

        boxes = []
        # area for each box
        area = []

        for i in range(N):
            # where gets the pixels where the mask = True (mask is a 2D Array of true and false , 
            # true at the pixels that is 1 as an indication of the mask & 0 for background)
            mask_pixels = np.where(masks[i])

            # extract the box from the min & max of these points
            # first dim is y , second dim is x
            xmin = np.min(mask_pixels[1])
            xmax = np.max(mask_pixels[1])
            ymin = np.min(mask_pixels[0])
            ymax = np.max(mask_pixels[0])

            boxes.append([xmin, ymin, xmax, ymax])
            area.append((ymax-ymin) * (xmax-xmin))

        # convert 2D List to 2D Tensor (this is not numpy array)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)

        # labels for each box
        # there is only 1 class (pedestrian) so it will always be 1 (if multiple classes, so we will assign 1,2,3 ... etc to each one)
        labels = torch.ones((N,), dtype=torch.int64)

        # image_id requirement for model, index is unique for every image
        image_id = torch.tensor([index], dtype=torch.int64)

        # instances with iscrowd=True will be ignored during evaluation.
        # set all = False (zeros)
        iscrowd = torch.zeros((N,), dtype=torch.uint8)

        # convert masks to tensor (model requirement)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # print("image size=", image.size)
        # print("mask size=", mask.size)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["area"] = area
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.imagesPaths)



# convert the mask into a colored one

def get_coloured_mask(mask):
    colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
              [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    colors = [[255, 0, 0]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colors[random.randrange(0, len(colors))]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def intersection_over_union(box1, box2):
    # Assign variable names to coordinates for clarity
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area

    return iou


# define our collate to allow data and target with different sizes
# as default collate (which collect the images,targets of the patch) dosn't allow diffirent sizes

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


# get the PennFudanDataset train & test loaders

def get_dataset_loaders(transform, batch_size, test_batch_size, root, split_perecentage):
    # Load Dataset
    dataset = PennnFudanDataset(root, transform)

    # Split dataset into train and test
    n = len(dataset)
    factor_subset = int(split_perecentage * n)

    train_dataset = Subset(dataset, list(range(0, factor_subset)))
    test_dataset = Subset(dataset, list(range(factor_subset, n)))

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=my_collate)

    return train_loader, test_loader, dataset