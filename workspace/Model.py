from InsectDataSetHandler import InsectDataSetHandler, get_transform
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from projUtils.utils import *
import utils, engine
import warnings
from random import randrange
from projUtils.configHandler import ConfigHandler, CONFIGPATH
import pickle
warnings.filterwarnings('ignore')

class Model:
    def __init__(self):
        self.configHandler = ConfigHandler(CONFIGPATH)
        self.dataSet = None
        self.dataSet_Test = None
        self.dataSet_Validation = None
        self.dataLoader = None
        self.dataLoader_Test = None
        self.model = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.configHandler.getIsOnlyDetect():
            self.numOfClasses = 2
        else:
            self.numOfClasses = len(SPECIMEN_FAMILIES_STR) + 1 # We add one for the background

    def createDataSets(self):
        imageDimensionX = self.configHandler.getImageWidth()
        imageDimensionY = self.configHandler.getImageHeight()
        self.dataSet = InsectDataSetHandler(TRAIN_DATA_SET_PATH, imageDimensionX, imageDimensionY, transforms=get_transform(train=False))
        self.dataSet_Test = InsectDataSetHandler(TEST_DATA_SET_PATH, imageDimensionX, imageDimensionY, transforms=get_transform(train=False))
        self.dataSet_Validation = InsectDataSetHandler(VALIDATION_DATA_SET_PATH, imageDimensionX, imageDimensionY, transforms=get_transform(train=False))

    def splitAndCreateDataLoaders(self):
        # split the dataset in train and test set
        torch.manual_seed(1)
        batch_size = self.configHandler.getImageBatchSize()
        # define training and validation data loaders
        self.dataLoader = torch.utils.data.DataLoader(
            self.dataSet,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=utils.collate_fn,
        )

        self.dataLoader_Test = torch.utils.data.DataLoader(
            self.dataSet_Test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=utils.collate_fn,
        )

    def getPreTrainedObject(self):
        if self.configHandler.isNewModelCreationEnabled():
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.numOfClasses)
        else:
            configFilePath = self.configHandler.getExistingModelPath()
            if os.path.isfile(configFilePath):
                model = pickle.load(open(configFilePath, 'rb'))
            else:
                raise FileNotFoundError
        self.model = model

    def train(self):
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
        numberOfEpochs = self.configHandler.getEpochAmount()
        doEvluate = self.configHandler.getDoEpochEvaluation()
        for epoch in range(numberOfEpochs):
            engine.train_one_epoch(self.model, optimizer, self.dataLoader, self.device, epoch, print_freq=10)
            lr_scheduler.step()
            if doEvluate:
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

    def testOurModel(self, iou_threshold):
        imageAmount = self.configHandler.getTestImagesAmount()
        validationImages = getValidationImagesAmount()
        for imageNum in range(imageAmount):
            imageNumberToEval = randrange(validationImages)
            img, target = self.dataSet_Validation[imageNumberToEval]
            self.model.eval()
            with torch.no_grad():
                prediction = self.model([img.to(self.device)])[0]

            print('MODEL OUTPUT\n')
            nms_prediction = self.filterOutPuts(prediction, iou_threshold=iou_threshold)

            plotImageModelOutput(self.covnvertToPil(img), nms_prediction,
                                 self.configHandler.getScoreLimitGreen(), self.configHandler.getScoreLimitBlue(),
                                 self.configHandler.getSaveImagesEnabled(), "{}.png".format(imageNum),
                                 self.configHandler.getImageDPI())
        print("finished evaluation")


    def calculate_precision_recall(self):
        # Obtain model predictions for test images
        thresh_hold = self.configHandler.getRetangaleOverlap()

        test_images = [self.dataSet_Validation[i][0] for i in range(len(self.dataSet_Validation))]
        test_boxes = [self.dataSet_Validation[i][1]["boxes"] for i in range(len(self.dataSet_Validation))]
        with torch.no_grad():
            predictions = [self.filterOutPuts(self.model([img.to(self.device)])[0]) for img in test_images]
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        preds = []
        tests = []
        for pred, test in zip(predictions, test_boxes):
            for test_box in test:
                preds.append((test_box[0].to(torch.int32), test_box[1].to(torch.int32), (test_box[2] - test_box[0]).to(torch.int32), (test_box[3] - test_box[1]).to(torch.int32)))
            for pred_box in pred["boxes"]:
                tests.append((pred_box[0].to(torch.int32), pred_box[1].to(torch.int32), (pred_box[2] - pred_box[0]).to(torch.int32), (pred_box[3] - pred_box[1]).to(torch.int32)))

        for pred_1 in preds:
            a = True
            for tests_1 in tests:
                if a & (abs(pred_1[0] - tests_1[0]) <= thresh_hold) & (abs(pred_1[1] - tests_1[1]) <= thresh_hold) & (abs(pred_1[2] - tests_1[2]) <= thresh_hold) & (abs(pred_1[3] - tests_1[3]) <= thresh_hold):
                    true_positives+=1
                    a = False
            if a:
                false_positives +=1
        for tests_1 in tests:
            a = True
            for pred_1 in preds:
                if a & (abs(pred_1[0] - tests_1[0]) <= thresh_hold) & (abs(pred_1[1] - tests_1[1]) <= thresh_hold) & (abs(pred_1[2] - tests_1[2]) <= thresh_hold) & (abs(pred_1[3] - tests_1[3]) <= thresh_hold):
                    a = False
            if a:
                false_negatives +=1
        print("True Positives:", true_positives)
        print("False Positives:", false_positives)
        print("True Negatives:", false_negatives)


    def export(self):
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        modelFileName = "{}.pkl".format(self.configHandler.getExportModelName())
        modelPath = os.path.join(OUTPUT_DIR, modelFileName)
        pickle.dump(self.model, open(modelPath, 'wb'))
