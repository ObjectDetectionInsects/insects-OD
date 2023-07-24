import os
import glob
import torch
import pickle
import io
import numpy as np
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches
from workspace.projUtils.configHandler import ConfigHandler, CONFIGPATH


class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        return AttributeError


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def enum(**enums):
    return type('Enum', (), enums)

#please note - labels 1 and 0 are saved for default insect and background
SPECIMEN_FAMILIES = enum(Curculionidae=2, Gelechiidae=3,
                        GeneralBeetles=4, GeneralMoth=5,
                        GeneralParasitoidWasp=6, UnknownFamily=7)

SPECIMEN_FAMILIES_STR = Enum(['Curculionidae',
                              'Gelechiidae',
                              'Generalbeetles',
                              'Generalmoth',
                              'Generalparasitoidwasp',
                              'Unknownfamily'])
DEFAULT_LABEL = 1
CSV_X_POS = 1
CSV_Y_POS = 2
CSV_W_POS = 3
CSV_H_POS = 4
JPG_EXTENSION = 'jpg'
CSV_EXTENSION = 'csv'
PNG_EXTENSION = 'png'
CSV_DELIMETER = ','
TABLE_HEADER = "parent_image_file_name"
DATA_SET_PATH = os.path.join(os.path.abspath(__file__ + "/../../../"), "DataSets")
SPLITTED_DATA_SET_PATH = os.path.join(os.path.abspath(__file__ + "/../../../"), "SplittedDataSets")
TRAIN_DATA_SET_PATH = os.path.join(os.path.abspath(__file__ + "/../../../SplittedDataSets"), "TrainSet")
TEST_DATA_SET_PATH = os.path.join(os.path.abspath(__file__ + "/../../../SplittedDataSets"), "TestSet")
VALIDATION_DATA_SET_PATH = os.path.join(os.path.abspath(__file__ + "/../../../SplittedDataSets"), "ValidationSet")
OUTPUT_DIR = os.path.join(os.path.abspath(__file__ + "/../../../"), "modelOutPuts")
SPACE = " "
IMAGE_EXTENSION = [JPG_EXTENSION, PNG_EXTENSION]


def getSpecimenFamily(specimenSting):
    if specimenSting == SPECIMEN_FAMILIES_STR.Curculionidae:
        return SPECIMEN_FAMILIES.Curculionidae
    elif specimenSting == SPECIMEN_FAMILIES_STR.Gelechiidae:
        return SPECIMEN_FAMILIES.Gelechiidae
    elif specimenSting == SPECIMEN_FAMILIES_STR.Generalbeetles:
        return SPECIMEN_FAMILIES.GeneralBeetles
    elif specimenSting == SPECIMEN_FAMILIES_STR.Generalmoth:
        return SPECIMEN_FAMILIES.GeneralMoth
    elif specimenSting == SPECIMEN_FAMILIES_STR.Generalparasitoidwasp:
        return SPECIMEN_FAMILIES.GeneralParasitoidWasp
    else:# specimenSting == SPECIMEN_FAMILIES_STR.Unknownfamily
        return SPECIMEN_FAMILIES.UnknownFamily


def dataIsFull(splittedLine, onlyDetction):
    if not onlyDetction:
        return not splittedLine[18] == ""
    if splittedLine[1] == "" or splittedLine[2] == "" or splittedLine[3] == "" or splittedLine[4] == "":
        return False
    return True


def generateDataSetFromSingleCsv(csvFilePath, saveLocation ,onlyDetection = True):
    newCsvFileNames = []
    if os.path.isfile(csvFilePath):
        with open(csvFilePath, 'r') as file:
            for line in file.readlines():
                splittedLine = line.split(CSV_DELIMETER)
                associatedImage = splittedLine[6]
                if not associatedImage == TABLE_HEADER:
                    for extension in IMAGE_EXTENSION:
                        if extension in associatedImage:
                            newCsvName = associatedImage.split(extension)[0] + CSV_EXTENSION
                    if dataIsFull(splittedLine, onlyDetection):
                        if onlyDetection:
                            specimenFamily = DEFAULT_LABEL
                        else:
                            specimenFamily = getSpecimenFamily(splittedLine[18].replace(SPACE, ""))
                        lineForCsv = "{},{},{},{},{}\n".format(specimenFamily, splittedLine[1], splittedLine[2],
                                                               splittedLine[3], splittedLine[4])
                        if newCsvName in newCsvFileNames:
                            with open(os.path.join(saveLocation, newCsvName), "a") as newCsv:
                                try:
                                    newCsv.write(lineForCsv)
                                except:
                                    print("Failed on writing to csv file")
                        else:
                            newCsvFileNames.append(newCsvName)
                            with open(os.path.join(saveLocation, newCsvName),"w") as newCsv:
                                try:
                                    newCsv.write(lineForCsv)
                                except:
                                    print("Failed on writing to csv file")
        print("Succesfully divided csv {} into {}".format(csvFilePath, newCsvFileNames))
    else:
        print("csv path given is not a file")
    return


def generateAllDataSets(dataSetsPath, onlyDetection = True):
    if os.path.isdir(dataSetsPath):
        for folder in os.listdir(dataSetsPath):
            currentDir = os.path.join(dataSetsPath, folder)
            csvFiles = glob.glob(os.path.join(currentDir, '*.{}'.format(CSV_EXTENSION)))
            for csvfile in csvFiles:
                generateDataSetFromSingleCsv(os.path.join(currentDir, csvfile), currentDir, onlyDetection)
    else:
        print("Path given does not exist")


def get_filename(path):
    return '.'.join(os.path.basename(path).split('.')[:-1])


def split_image(image_path):
    configParser = ConfigHandler(CONFIGPATH)
    height = configParser.getImageHeight()
    width = configParser.getImageWidth()
    image_filename = get_filename(image_path)
    image_extension = os.path.basename(image_path).split('.')[-1]
    width_count = 0
    height_count = 0
    img = Image.open(image_path)
    img_width, img_height = img.size
    for i in range(0, img_height, height):
        for j in range(0, img_width, width):
            box = (j, i, j + width, i + height)
            a = img.crop(box)
            new_file_name = image_filename + "-{}-{}.".format(width_count, height_count) + image_extension
            a.save(os.path.join(SPLITTED_DATA_SET_PATH, new_file_name))
            width_count += 1
        width_count = 0
        height_count += 1


def split_csv(csv_path):
    configParser = ConfigHandler(CONFIGPATH)
    height = configParser.getImageHeight()
    width = configParser.getImageWidth()
    csv_filename = get_filename(csv_path)
    with open(csv_path, 'r') as file:
        data = file.readlines()
        for line in data:
            split_line = line.strip('\n').split(',')
            x_pos = int(split_line[CSV_X_POS])
            y_pos = int(split_line[CSV_Y_POS])
            rational_x_pos = x_pos % width
            rational_y_pos = y_pos % height
            w_pos = int(split_line[CSV_W_POS])
            h_pos = int(split_line[CSV_H_POS])

            width_count = int(x_pos / width)
            height_count = int(y_pos / height)

            # Fix the frames of a sliced bugs
            if rational_x_pos + w_pos > width:
                split_line[CSV_W_POS] = width - rational_x_pos
            if rational_y_pos + h_pos > height:
                split_line[CSV_H_POS] = height - rational_y_pos

            update_csv_filename = csv_filename + "-{}-{}.".format(width_count, height_count) + CSV_EXTENSION
            update_csv_path = os.path.join(SPLITTED_DATA_SET_PATH, update_csv_filename)
            split_line[CSV_X_POS] = rational_x_pos
            split_line[CSV_Y_POS] = rational_y_pos

            updated_line = [str(i) for i in split_line]
            updated_line = ",".join(updated_line) + '\n'

            with open(update_csv_path, 'a') as file_to_update:
                file_to_update.write(updated_line)


def split_images():
    if os.path.isdir(DATA_SET_PATH):
        for folder in os.listdir(DATA_SET_PATH):
            current_dir = os.path.join(DATA_SET_PATH, folder)
            csv_files = glob.glob(os.path.join(current_dir, '*.{}'.format(CSV_EXTENSION)))
            jpg_files = glob.glob(os.path.join(current_dir, '*{}'.format(JPG_EXTENSION)))
            print("\nFound the following files to split: ", ", ".join(csv_files))
            print("Inside folder: " + current_dir)
            for csv_file in csv_files:
                for jpg_file in jpg_files:
                    if get_filename(csv_file) == get_filename(jpg_file):
                        print("\nSplitting the following files: ", ", ".join([csv_file, jpg_file]))
                        split_csv(csv_file)
                        split_image(jpg_file)
            if len(jpg_files) > 0:
                split_image(jpg_files[0])


def get_single_insect_image(image_path, x, y, w, h):
    image_filename = get_filename(image_path)
    image_extension = os.path.basename(image_path).split('.')[-1]
    img = Image.open(image_path)
    box = (x, y, x + w, y + h)
    a = img.crop(box)
    new_file_name = image_filename + "-insect-{}-{}.".format(x, y) + image_extension
    a.save(os.path.join(OUTPUT_DIR, new_file_name))

def countBoxes(listOfPredictions, predictionResults = False):
    count = 0
    for prediction in listOfPredictions:
        if predictionResults:
            prediction = prediction["boxes"]
        for box in prediction:
            count += 1
    return count

def filterEdgePredictions(predictions, imageWidth, imageHeight):
    filtered_predictions = []
    for pred in predictions:
        filtered_boxes = []
        for pred_box in pred["boxes"]:
            x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = pred_box.data.cpu().numpy()
            if not ((imageHeight - y_topleft_p < 2) or (imageHeight - y_bottomright_p < 2)
                or (imageWidth - x_topleft_p < 2) or (imageWidth - x_bottomright_p < 2)):
                filtered_boxes.append(pred_box)
        if filtered_boxes:
            pred["boxes"] = filtered_boxes
            filtered_predictions.append(pred)
    return filtered_predictions

def filterLowGradeBoxes(predictions, boxThreshold):
    #the input is a list of predictions (on multiple images)
    filtered_predictions = []

    for pred in predictions:
        filtered_boxes = []
        for pred_box, pred_score in zip(pred["boxes"], pred["scores"]):
            if pred_score >= boxThreshold:
                filtered_boxes.append(pred_box)
        if filtered_boxes:
            pred["boxes"] = filtered_boxes
            filtered_predictions.append(pred)
    return filtered_predictions


def plotImageModelOutput(img, target, greenScore, blueScore, savePlot, imageName, imageDPI, minScore):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)
    for box,score in zip(target['boxes'],target['scores']):
        if score >= minScore:
            x, y, width, height = box[0].cpu().numpy(), box[1].cpu().numpy(), (box[2] - box[0]).cpu().numpy(), (box[3] - box[1]).cpu().numpy()
            score = score.item()
            color = (score,0,0)
            if score > blueScore:
                color = (0,0,score)
            if score > greenScore:
                color = (0,score,0)
            rect = patches.Rectangle(
                (x, y),
                width, height,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            # Draw the bounding box on top of the image
            a.add_patch(rect)
    if savePlot:
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        plt.savefig(os.path.join(OUTPUT_DIR, imageName), dpi= imageDPI)
    else:
        plt.show()


def plotImage(img, target, imageName):
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
    # plt.show()
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, imageName), dpi=500)


#TODO this is a temporary fix! for Ori testing. a better fix to be made.
def fixIncorrectSplittedCsv(splittedPath = SPLITTED_DATA_SET_PATH):
    csv_files = glob.glob(os.path.join(splittedPath, '*.{}'.format(CSV_EXTENSION)))
    jpg_files = glob.glob(os.path.join(splittedPath, '*{}'.format(JPG_EXTENSION)))
    for jpg_file in jpg_files:
        found = False
        for csv_file in csv_files:
            if get_filename(csv_file) == get_filename(jpg_file):
                found = True
        if not found:
            print("removing file {}".format(jpg_file))
            os.remove(jpg_file)


def split_train_test_validation(splittedPath = SPLITTED_DATA_SET_PATH):
    for path in (VALIDATION_DATA_SET_PATH, TEST_DATA_SET_PATH, TRAIN_DATA_SET_PATH):
        if not os.path.isdir(path):
            os.mkdir(path)
    count = 1
    for filename in os.listdir(splittedPath):
        # Check if the file is a JPG image
        if filename.endswith(".jpg"):
            # Get the base name of the file without the extension
            basename = os.path.splitext(filename)[0]
            # Find any CSV files with the same name
            csv_files = [f for f in os.listdir(splittedPath) if f.startswith(basename) and f.endswith(".csv")]
            # If at least one CSV file was found, rename it and number it from 1
            if csv_files:
                for csv_file in csv_files:
                    # Rename the CSV file
                    if(count%7==0):
                        if(count%2==0):
                            os.rename(os.path.join(splittedPath, csv_file),
                                      os.path.join(TEST_DATA_SET_PATH, f"{count}.csv"))
                            os.rename(os.path.join(splittedPath, filename),
                                      os.path.join(TEST_DATA_SET_PATH, f"{count}.jpg"))
                        else:
                            os.rename(os.path.join(splittedPath, csv_file),
                                      os.path.join(VALIDATION_DATA_SET_PATH, f"{count}.csv"))
                            os.rename(os.path.join(splittedPath, filename),
                                      os.path.join(VALIDATION_DATA_SET_PATH, f"{count}.jpg"))
                    if(count%7!=0):
                      os.rename(os.path.join(splittedPath, csv_file), os.path.join(TRAIN_DATA_SET_PATH, f"{count}.csv"))
                      os.rename(os.path.join(splittedPath, filename), os.path.join(TRAIN_DATA_SET_PATH, f"{count}.jpg"))
                    # Increment the counter variable
                    count += 1
            else:
                os.remove(os.path.join(splittedPath, filename))


def getImageAmountInDir(path):
    jpg_files = glob.glob(os.path.join(path, '*{}'.format(JPG_EXTENSION)))
    return len(jpg_files)


def image_to_tensor(img_path):
    tensor_img = Image.open(img_path)
    convert_tensor = transforms.ToTensor()
    return convert_tensor(tensor_img)


def getConfidenceArray(minVal, maxVal, step):
    arrayOfSteps = []
    currentVal = round(minVal, 2)

    while currentVal <= maxVal:
        arrayOfSteps.append(currentVal)
        currentVal += step
        currentVal = round(currentVal, 2)

    if arrayOfSteps[-1] < maxVal:
        arrayOfSteps.append(maxVal)

    return arrayOfSteps


def calculateIOU(predictionBox, actualBox):
    x1_1, y1_1, x2_1, y2_1 = actualBox.numpy()
    x1_2, y1_2, x2_2, y2_2 = predictionBox.data.cpu().numpy()

    # Calculate intersection coordinates
    intersect_x1 = max(x1_1, x1_2)
    intersect_y1 = max(y1_1, y1_2)
    intersect_x2 = min(x2_1, x2_2)
    intersect_y2 = min(y2_1, y2_2)

    # Calculate intersection area
    intersect_width = intersect_x2 - intersect_x1
    intersect_height = intersect_y2 - intersect_y1

    if intersect_width <= 0 or intersect_height <= 0:
        return 0.0

    intersection_area = intersect_width * intersect_height

    # Calculate union area
    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def getOverlapResults(prediction, actualBoxes, iouThresh):
    gt_idx_thr=[]
    pred_idx_thr=[]
    ious=[]

    for indexPredBox, predBox in enumerate(prediction['boxes']):
        for indexActBox, actBox in enumerate(actualBoxes):
            iou = calculateIOU(predBox, actBox)
            if iou >iouThresh:
                gt_idx_thr.append(indexActBox)
                pred_idx_thr.append(indexPredBox)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort) == 0:
        return 0, 0, 0
    else:
        gt_match_idx=[]
        pred_match_idx=[]
        for idx in iou_sort:
            gt_idx=gt_idx_thr[idx]
            pr_idx= pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(prediction['boxes']) - len(pred_match_idx)
        fn = len(actualBoxes) - len(gt_match_idx)
    return tp, fp, fn

def getOverlapResultsRefactor(prediction, actualBoxes, iouThresh):
    truePositives, falsePositives, falseNegative = 0, 0, 0
    for box in prediction['boxes']:
        bestIOU = 0
        for actBox in actualBoxes:
            currentIOU = calculateIOU(box, actBox)
            if currentIOU > bestIOU:
                bestIOU = currentIOU
        if bestIOU > iouThresh:
            truePositives += 1
            #remove Good prediction
        else:
            falsePositives += 1

    for actBox in actualBoxes:
        bestIOU = 0
        for box in prediction['boxes']:
            currentIOU = calculateIOU(box, actBox)
            if currentIOU > bestIOU:
                bestIOU = currentIOU
        if bestIOU < iouThresh:
            falseNegative += 1

    print("TP: {}, FP: {}, FN: {}".format(truePositives, falsePositives, falseNegative))
    return truePositives, falsePositives, falseNegative




if __name__ == '__main__':
    configParser = ConfigHandler(CONFIGPATH)
    generateAllDataSets(DATA_SET_PATH, configParser.getIsOnlyDetect())
    if not os.path.isdir(SPLITTED_DATA_SET_PATH):
        os.mkdir(SPLITTED_DATA_SET_PATH)
    split_images()
    fixIncorrectSplittedCsv()
    split_train_test_validation()



