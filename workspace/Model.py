from InsectDataSetHandler import InsectDataSetHandler, get_transform
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from projUtils.utils import plot_img_bbox, SPECIMEN_FAMILIES_STR
import utils, engine
import warnings
from projUtils import utils as U
import numpy as np
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


class Model:
    def __init__(self, detetionOnly = True):
        self.dataSet = None
        self.dataSet_Test = None
        self.dataSet_Validation = None
        self.dataLoader = None
        self.dataLoader_Test = None
        self.model = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if detetionOnly:
            self.numOfClasses = 2
        else:
            self.numOfClasses = len(SPECIMEN_FAMILIES_STR) + 1

    def createDataSets(self, dataDir, imageDimensionX, imageDimensionY):
        # TRAIN_DATA_SET_PATH = os
        # TEST_DATA_SET_PATH = os.
        # VALIDATION_DATA_SET_PATH
        self.dataSet = InsectDataSetHandler(U.TRAIN_DATA_SET_PATH, imageDimensionX, imageDimensionY, transforms=get_transform(train=False))
        self.dataSet_Test = InsectDataSetHandler(U.TEST_DATA_SET_PATH, imageDimensionX, imageDimensionY, transforms=get_transform(train=False))
        self.dataSet_Validation = InsectDataSetHandler(U.VALIDATION_DATA_SET_PATH, imageDimensionX, imageDimensionY, transforms=get_transform(train=False))
        # print(self.dataSet)

    def splitAndCreateDataLoaders(self):
        # split the dataset in train and test set
        torch.manual_seed(1)
        # indices = torch.randperm(len(self.dataSet)).tolist()

        # train test split
        # test_split = 0.2
        # tsize = int(len(self.dataSet) * test_split)
        # dataset = torch.utils.data.Subset(self.dataSet, indices[:-tsize])
        # dataset_test = torch.utils.data.Subset(self.dataSet, indices[-tsize:])

        # define training and validation data loaders
        self.dataLoader = torch.utils.data.DataLoader(
            self.dataSet,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            collate_fn=utils.collate_fn,
        )

        self.dataLoader_Test = torch.utils.data.DataLoader(
            self.dataSet_Test,
            batch_size=1,
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
        img, target = self.dataSet_Validation[imageNumber]
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([img.to(self.device)])[0]

        print('MODEL OUTPUT\n')
        nms_prediction = self.filterOutPuts(prediction, iou_threshold=iou_threshold)

        plot_img_bbox(self.covnvertToPil(img), nms_prediction)

    def calculate_precision_recall(self):
        # Obtain model predictions for test images
        test_images = [self.dataSet_Validation[i][0] for i in range(len(self.dataSet_Validation))]
        test_boxes = [self.dataSet_Validation[i][1]["boxes"] for i in range(len(self.dataSet_Validation))]

        # img, target = self.dataSet_Validation[1]
        # test_images = images
        # test_labels = target["labels"]
        with torch.no_grad():
            predictions = [self.filterOutPuts(self.model([img.to(self.device)])[0]) for img in test_images]
        # predictions = self.model.predict(test_images)
        # nms_prediction = self.filterOutPuts(self.model([img.to(self.device)])[0], iou_threshold=iou_threshold)

        # Calculate precision and recall at different threshold levels
        # thresholds = np.linspace(0, 1, 30)  # Adjust the number of thresholds as desired
        # precision_values = []
        # recall_values = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        # x, y, width, height = box[0].cpu().numpy(), box[1].cpu().numpy(), (box[2] - box[0]).cpu().numpy(), (box[3] - box[1]).cpu().numpy()

        a = True
        preds = []
        tests = []
        for pred, test in zip(predictions, test_boxes):
            for test_box in test:
                preds.append((test_box[0].to(torch.int32), test_box[1].to(torch.int32), (test_box[2] - test_box[0]).to(torch.int32), (test_box[3] - test_box[1]).to(torch.int32)))
            for pred_box in pred["boxes"]:
                tests.append((pred_box[0].to(torch.int32), pred_box[1].to(torch.int32), (pred_box[2] - pred_box[0]).to(torch.int32), (pred_box[3] - pred_box[1]).to(torch.int32)))
        thresh_hold = 150
        for pred_1 in preds:
            a = True
            for tests_1 in tests:
                # print(f"pred[0] - tests[0] {pred[0]} - {tests_1[0]} = {abs(pred[0] - tests_1[0])}")
                # print(f"pred[1] - tests[1] {pred[1]} - {tests_1[1]} = {abs(pred[1] - tests_1[1])}")
                # print(f"pred[2] - tests[2] {pred[2]} - {tests_1[2]} = {abs(pred[2] - tests_1[2])}")
                # print(f"pred[3] - tests[3] {pred[3]} - {tests_1[3]} = {abs(pred[3] - tests_1[3])}")
                if a & (abs(pred_1[0] - tests_1[0]) <= thresh_hold) & (abs(pred_1[1] - tests_1[1]) <= thresh_hold) & (abs(pred_1[2] - tests_1[2]) <= thresh_hold) & (abs(pred_1[3] - tests_1[3]) <= thresh_hold):
                    true_positives+=1
                    a = False
            if a:
                false_positives +=1

        for tests_1 in tests:
            a = True
            for pred_1 in preds:
                # print(f"pred[0] - tests[0] {pred[0]} - {tests_1[0]} = {abs(pred[0] - tests_1[0])}")
                # print(f"pred[1] - tests[1] {pred[1]} - {tests_1[1]} = {abs(pred[1] - tests_1[1])}")
                # print(f"pred[2] - tests[2] {pred[2]} - {tests_1[2]} = {abs(pred[2] - tests_1[2])}")
                # print(f"pred[3] - tests[3] {pred[3]} - {tests_1[3]} = {abs(pred[3] - tests_1[3])}")
                if a & (abs(pred_1[0] - tests_1[0]) <= thresh_hold) & (abs(pred_1[1] - tests_1[1]) <= thresh_hold) & (abs(pred_1[2] - tests_1[2]) <= thresh_hold) & (abs(pred_1[3] - tests_1[3]) <= thresh_hold):
                    a = False
            if a:
                false_negatives +=1
        print("True Positives:", true_positives)
        print("False Positives:", false_positives)
        print("True Negatives:", false_negatives)

        # x_test, y_test, width_test, height_test = test_box[0].cpu().numpy(), test_box[1].cpu().numpy(), (
        #             test_box[2] - test_box[0]).cpu().numpy(), (test_box[3] - test_box[1]).cpu().numpy()
        # x_pred, y_pred, width_pred, height_pred = pred_box[0].cpu().numpy(), pred_box[1].cpu().numpy(), (
        #             pred_box[2] - pred_box[0]).cpu().numpy(), (pred_box[3] - pred_box[1]).cpu().numpy()

        # print(true_positives)       a = True
        # for pred,test in zip(predictions,test_boxes):
        #     for test_box in test:
        #         for pred_box in pred["boxes"]:
        #             for i in range(4):
        #                 a = a & (abs(pred_box[i] - test_box[i]) <= 70)
        #                 if a:
        #                     true_positives+=1
        #

        # for threshold in thresholds:
        #     true_positives = 0
        #     false_positives = 0
        #     false_negatives = 0
        #
        #     for i in range(len(predictions)):
        #         predicted_label = 1 if predictions[i] >= threshold else 0
        #         true_label = test_labels[i]
        #
        #         if predicted_label == 1 and true_label == 1:
        #             true_positives += 1
        #         elif predicted_label == 1 and true_label == 0:
        #             false_positives += 1
        #         elif predicted_label == 0 and true_label == 1:
        #             false_negatives += 1
        #
        #     precision = true_positives / (true_positives + false_positives)
        #     recall = true_positives / (true_positives + false_negatives)
        #
        #     precision_values.append(precision)
        #     recall_values.append(recall)

        # Plotting the precision-recall graph
        # plt.plot(recall_values, precision_values, marker='o')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Graph for RCNN')
        # plt.ylim([0, 1])
        # plt.xlim([0, 1])
        # plt.show()
        # return precision_values, recall_values