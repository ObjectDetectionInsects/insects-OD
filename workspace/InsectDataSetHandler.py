import torch
import os
import cv2
import numpy as np
from projUtils.utils import *
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from random import randrange

class InsectDataSetHandler(torch.utils.data.Dataset):
    def __init__(self, files_dir, width, height, onlyDetection = True,  transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width
        self.onlyDetection = onlyDetection
        self.imgs = [image for image in sorted(os.listdir(files_dir)) if image[-4:] == '.jpg']

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        # reading the images and converting them to correct size and color
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0

        # annotation file
        annot_filename = img_name[:-4] + '.csv'
        annot_file_path = os.path.join(self.files_dir, annot_filename)

        boxes = []
        labels = []

        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]

        # box coordinates for xml files are extracted and corrected for image size given
        with open(annot_file_path) as f:
            for line in f:
                parsed = [float(x) for x in line.split(',')]
                if self.onlyDetection:
                    labels.append(DEFAULT_LABEL)
                else:
                    labels.append([parsed[0]])


                x_min = parsed[1]
                y_min = parsed[2]
                box_wt = parsed[3]
                box_ht = parsed[4]

                x_max = x_min + box_wt
                y_max = y_min + box_ht

                x_min_corr = int(x_min)
                x_max_corr = int(x_max)
                y_min_corr = int(y_min)
                y_max_corr = int(y_max)

                # TODO: to verify reason
                # xmin_corr = int(xmin * self.width)
                # xmax_corr = int(xmax * self.width)
                # ymin_corr = int(ymin * self.height)
                # ymax_corr = int(ymax * self.height)

                boxes.append([x_min_corr, y_min_corr, x_max_corr, y_max_corr])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        is_crowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = is_crowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image=img_res,
                                     bboxes=target['boxes'],
                                     labels=labels)
            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return img_res, target

    def __len__(self):
        return len(self.imgs)


# Send train=True for training transforms and False for val/test transforms
def get_transform(train):
  if train:
    return A.Compose(
      [
        # A.HorizontalFlip(0.5),
        # ToTensorV2 converts image to pytorch tensor without div by 255
        ToTensorV2(p=1.0)
      ],
      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )
  else:
    return A.Compose(
      [ToTensorV2(p=1.0)],
      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )

if __name__ == '__main__':
    dataSetDir = SPLITTED_DATA_SET_PATH
    print(dataSetDir)
    dataSetClass = InsectDataSetHandler(dataSetDir, width=2000, height=2000, onlyDetection = True)
    for imageNumber in range(20):
        imageNumbRandom = randrange(1000)
        image, target = dataSetClass[imageNumbRandom]
        plotImage(image, target)
    print("plotted images for verification")
