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
import matplotlib.pyplot as plt
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
            self.dataSet_Test,
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
                if self.device == torch.device('cpu'):
                    model = CPU_Unpickler(open(configFilePath, 'rb')).load()
                else:
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
        doLossPerEpoch = self.configHandler.getdoLossPerEpoch()
        loss_rate = []
        loss_rate2 = []
        for epoch in range(numberOfEpochs):
            loss_rate.append(engine.train_one_epoch(self.model, optimizer, self.dataLoader, self.device, epoch, print_freq=10))
            lr_scheduler.step()
            if doEvluate:
                engine.evaluate(self.model, self.dataLoader_Test, device=self.device)
        if doLossPerEpoch:
            # epochs = [i for i in range(len(loss_rate[0]*numberOfEpochs))]
            # print(loss_rate)
            # print(len(loss_rate[0]))
            # # loss_rate2 = [[loss_rate[0][i][0]] for i in len(loss_rate[0])]
            # print(len(loss_rate))
            loss_rate = loss_rate[0]
            print(loss_rate)
            epochs = [i for i in range(len(loss_rate))]
            print(f"!!!!!!!!!!!!!!{epochs}")
            # a=[[[1.1358028650283813, 0.5318865776062012, 0.3059080243110657, 0.015454649925231934, 0.2825535535812378], [1.0578477382659912, 0.40596503019332886, 0.4108307361602783, 0.020273687317967415, 0.22077827155590057]]]


            # max_loss = max(loss_rate[0])[0] * 1.1
            max_loss = max(loss_rate) * 1.1
            plt.plot(epochs, loss_rate, marker='o')
            plt.ylim(0,max_loss)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss per Epoch')
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "lossPerEpoch.png"))

    def filterOutPuts(self, orig_prediction, iou_threshold = 0.3):
        keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_threshold)

        final_prediction = orig_prediction
        final_prediction['boxes'] = final_prediction['boxes'][keep]
        final_prediction['scores'] = final_prediction['scores'][keep]
        final_prediction['labels'] = final_prediction['labels'][keep]

        return final_prediction

    def covnvertToPil(self, image):
        return torchtrans.ToPILImage()(image).convert('RGB')

    def testOurModel(self):
        imageAmount = self.configHandler.getTestImagesAmount()
        iou_threshold = self.configHandler.getIouThreshold()
        validationImages = getValidationImagesAmount()
        boxThreshold = self.configHandler.getBoxScoreLimit()

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
                                 self.configHandler.getImageDPI(), boxThreshold)
        print("finished evaluation")

    def calculate_precision_recall(self):
        minVal = self.configHandler.getPrecisionRecallMinIOU()
        maxVal = self.configHandler.getPrecisionRecallMaxIOU()
        step = self.configHandler.getPrecisionRecallIouSteps()
        thresh_hold = self.configHandler.getRetangaleOverlap()
        iou_threshold_arr = [minVal,maxVal]

        print("iou values tested are: {}".format(iou_threshold_arr))
        print("calculate_precision_recall loading")

        test_images = [self.dataSet_Validation[i][0] for i in range(0,len(self.dataSet_Validation))]
        test_boxes = [self.dataSet_Validation[i][1]["boxes"] for i in range(0,len(self.dataSet_Validation))]
        t=0
        precision=[]
        recall=[]
        actual_boxex = 0
        iou_threshold_arr = [0.01]
        for thresh in iou_threshold_arr:
            print(f"check {t} out of {len(iou_threshold_arr)}")
            with torch.no_grad():
                predictions = [self.filterOutPuts(self.model([img.to(self.device)])[0],iou_threshold=thresh) for img in test_images]
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            preds=([[(pred_box[0].to(torch.int32), pred_box[1].to(torch.int32), (pred_box[2] - pred_box[0]).to(torch.int32), (pred_box[3] - pred_box[1]).to(torch.int32)) for pred_box in  pred["boxes"]] for pred in predictions])
            tests=([[(test_box[0].to(torch.int32), test_box[1].to(torch.int32), (test_box[2] - test_box[0]).to(torch.int32), (test_box[3] - test_box[1]).to(torch.int32)) for test_box in test] for test in test_boxes])
            for pred_1,test_1 in zip(preds,tests):
                for pred_2 in pred_1:
                    a = True
                    for tests_2 in test_1:
                        if a & (abs(pred_2[0] - tests_2[0]) <= thresh_hold) & (
                                abs(pred_2[1] - tests_2[1]) <= thresh_hold):
                            true_positives += 1
                            a = False
                    if a:
                        false_positives += 1
                for tests_2 in test_1:
                    if t ==0:
                        actual_boxex+=1
                    a = True
                    for pred_2 in pred_1:
                        if a & (abs(pred_2[0] - tests_2[0]) <= thresh_hold) & (
                                abs(pred_2[1] - tests_2[1]) <= thresh_hold):
                            a = False
                    if a:
                        false_negatives +=1
            t+=1
            precision.append((true_positives/(true_positives + false_positives)))
            recall.append((true_positives/(true_positives + false_negatives)))
            print("True Positives:", true_positives)
            print("False Positives:", false_positives)
            print("False Negatives:", false_negatives)

        print("actual boxex", actual_boxex)
        print("calculate_precision_recall finished")
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('Precision-Recall Curve')
        print("Precision recall values are {} and {}".format(precision, recall))
        plt.savefig(os.path.join(OUTPUT_DIR, "precisionRecall.png"))

    def precisionRecall(self):
        minConfidence = self.configHandler.getPrecisionRecallMinConfidence()
        maxConfidence = self.configHandler.getPrecisionRecallMaxConfidence()
        step = self.configHandler.getPrecisionRecallConfidenceSteps()
        confidencesArray = getConfidenceArray(minConfidence, maxConfidence, step)
        iou = self.configHandler.getIouThreshold()

        precisions = []
        recalls = []

        validationImages = [self.dataSet_Validation[imageNum][0] for imageNum in range(0,len(self.dataSet_Validation))]
        actualBoxes = [self.dataSet_Validation[imageNum][1]["boxes"] for imageNum in range(0,len(self.dataSet_Validation))]
        with torch.no_grad():
            predictions = [self.model([img.to(self.device)])[0] for img in
                           validationImages]

        for confidence in confidencesArray:
            filteredPredictions = filterLowGradeBoxes(predictions, confidence)
            tp, fp, fn = 0, 0, 0
            for prediction, actual in zip(filteredPredictions, actualBoxes):
                tpImage, fpImage, fnImage = getOverlapResults(prediction, actual, iou)
                tp += tpImage
                fp += fpImage
                fn += fnImage
            print("tp:{}, fp:{}, fn:{}".format(tp, fp, fn))
            try:
                precision = tp/(tp+fp)
            except ZeroDivisionError:
                precision = 0.0
            try:
                recall = tp/(tp+fn)
            except ZeroDivisionError:
                recall = 0.0

            precisions.append(precision)
            recalls.append(recall)

        plt.plot(recalls, precisions)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('Precision-Recall Curve')
        print("Precision recall values are {} and {}".format(precisions, recalls))
        plt.savefig(os.path.join(OUTPUT_DIR, "precisionRecall.png"))
    def export(self):
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        modelFileName = "{}.pkl".format(self.configHandler.getExportModelName())
        modelPath = os.path.join(OUTPUT_DIR, modelFileName)
        pickle.dump(self.model, open(modelPath, 'wb'))

    def exportSingleInsect(self):
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        inputImagesPath = self.configHandler.getInputImages()
        imagesPath = os.listdir(inputImagesPath)
        minBoxScore = self.configHandler.getBoxScoreLimit()
        for image in imagesPath:
            self.model.eval()
            image_full_path = os.path.join(inputImagesPath, image)
            tensor_img = image_to_tensor(image_full_path)
            with torch.no_grad():
                prediction = self.model([tensor_img.to(self.device)])[0]
            for box,score in zip(prediction['boxes'],prediction['scores']):
                x, y, width, height = box[0].cpu().numpy(), box[1].cpu().numpy(), (box[2] - box[0]).cpu().numpy(), (
                            box[3] - box[1]).cpu().numpy()
                x = int(round(x.min(), 0))
                y = int(round(y.min(), 0))
                width = int(round(width.min(), 0))
                height = int(round(height.min(), 0))
                if score >= minBoxScore:
                    get_single_insect_image(image_full_path, x, y, width, height)
