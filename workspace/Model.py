from InsectDataSetHandler import InsectDataSetHandler, get_transform
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from projUtils.utils import plot_img_bbox, SPECIMEN_FAMILIES_STR
import utils, engine
import warnings
warnings.filterwarnings('ignore')

class Model:
    def __init__(self, detetionOnly = True):
        self.dataSet = None
        self.dataSet_Test = None
        self.dataLoader = None
        self.dataLoader_Test = None
        self.model = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if detetionOnly:
            self.numOfClasses = 2
        else:
            self.numOfClasses = len(SPECIMEN_FAMILIES_STR) + 1

    def createDataSets(self, dataDir, imageDimensionX, imageDimensionY):
        self.dataSet = InsectDataSetHandler(dataDir, imageDimensionX, imageDimensionY, transforms=get_transform(train=False))
        self.dataSet_Test = InsectDataSetHandler(dataDir, imageDimensionX, imageDimensionY, transforms=get_transform(train=False))

    def splitAndCreateDataLoaders(self):
        # split the dataset in train and test set
        torch.manual_seed(1)
        indices = torch.randperm(len(self.dataSet)).tolist()

        # train test split
        test_split = 0.2
        tsize = int(len(self.dataSet) * test_split)
        dataset = torch.utils.data.Subset(self.dataSet, indices[:-tsize])
        dataset_test = torch.utils.data.Subset(self.dataSet, indices[-tsize:])

        # define training and validation data loaders
        self.dataLoader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            num_workers=4,
            collate_fn=utils.collate_fn,
        )

        self.dataLoader_Test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=10,
            shuffle=False,
            num_workers=4,
            collate_fn=utils.collate_fn,
        )

    def getPreTrainedObject(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.numOfClasses)
        self.model = model

    def train(self, numberOfEpochs):
        self.model.to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

        for epoch in range(numberOfEpochs):
            engine.train_one_epoch(self.model, optimizer, self.dataLoader, self.device, epoch, print_freq=10)
            lr_scheduler.step()
            engine.evaluate(self.model, self.dataLoader_Test, device=self.device)

    def filterOutPuts(self, orig_prediction, iou_threshold = 0.3):
        keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_threshold)

        final_prediction = orig_prediction
        final_prediction['boxes'] = final_prediction['boxes'][keep]
        final_prediction['scores'] = final_prediction['scores'][keep]
        final_prediction['labels'] = final_prediction['labels'][keep]

        return final_prediction

    def covnvertToPil(self, image):
        return torchtrans.ToPILImage()(image).convert('RGB')

    def testOurModel(self, imageNumber, iou_threshold):
        img, target = self.dataSet[imageNumber]
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([img.to(self.device)])[0]

        print('MODEL OUTPUT\n')
        nms_prediction = self.filterOutPuts(prediction, iou_threshold=iou_threshold)

        plot_img_bbox(self.covnvertToPil(img), nms_prediction)
