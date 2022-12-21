import torch
import os
import cv2
import numpy as np
from utils.utils import DATA_SET_PATH
from matplotlib import pyplot as plt
from matplotlib import patches

class InsectDataSetHandler(torch.utils.data.Dataset):
    def __init__(self, files_dir, width, height, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width
        self.imgs = [image for image in sorted(os.listdir(files_dir)) if image[-4:] == '.jpg']

        # classes: 0 index is reserved for background
        # self.classes = [_, 'cone']

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
                labels.append(1)

                parsed = [float(x) for x in line.split(',')]
                x_center = parsed[1]
                y_center = parsed[2]
                box_wt = parsed[3]
                box_ht = parsed[4]

                xmin = x_center - box_wt / 2
                xmax = x_center + box_wt / 2
                ymin = y_center - box_ht / 2
                ymax = y_center + box_ht / 2

                xmin_corr = int(xmin * self.width)
                xmax_corr = int(xmax * self.width)
                ymin_corr = int(ymin * self.height)
                ymax_corr = int(ymax * self.height)

                boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
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


def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)
    for box in (target['boxes']):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle(
            (x, y),
            width, height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()


if __name__ == '__main__':
    dataSetDir = os.path.join(DATA_SET_PATH, "CircadianActivityNetivot-001")
    dataSetClass = InsectDataSetHandler(dataSetDir, width=7944, height=11256)
    image, target = dataSetClass[0]
    plot_img_bbox(image, target)