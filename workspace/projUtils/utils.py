import os
import glob
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches

class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        return AttributeError

def enum(**enums):
    return type('Enum', (), enums)


SPECIMEN_FAMILIES = enum(Curculionidae=1, Gelechiidae=2,
                        GeneralBeetles=3, GeneralMoth=4,
                        GeneralParasitoidWasp=5, UnknownFamily=6)

SPECIMEN_FAMILIES_STR = Enum(['Curculionidae',
                              'Gelechiidae',
                              'Generalbeetles',
                              'Generalmoth',
                              'Generalparasitoidwasp',
                              'Unknownfamily'])
DEFAULT_LABEL = 0
WIDTH = 2000
HEIGHT = 2000
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
    elif specimenSting == SPECIMEN_FAMILIES_STR.Unknownfamily:
        return SPECIMEN_FAMILIES.UnknownFamily
    else:
        print("unsupported type {}".format(specimenSting))

def dataIsFull(splittedLine, onlyDetction):
    if not onlyDetction:
        return not splittedLine[18] == ""
    if splittedLine[1] == "" or splittedLine[2] == "" or splittedLine[3] == "" or splittedLine[4] == "":
        return False
    return True

def generateDataSetFromSingleCsv(csvFilePath, onlyDetection = True):
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
                            with open(newCsvName, "a") as newCsv:
                                try:
                                    newCsv.write(lineForCsv)
                                except:
                                    print("Failed on writing to csv file")
                        else:
                            newCsvFileNames.append(newCsvName)
                            with open(newCsvName,"w") as newCsv:
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
                generateDataSetFromSingleCsv(os.path.join(currentDir, csvfile), onlyDetection)
    else:
        print("Path given does not exist")


def get_filename(path):
    return '.'.join(os.path.basename(path).split('.')[:-1])


def split_image(image_path):
    image_filename = get_filename(image_path)
    image_extension = os.path.basename(image_path).split('.')[-1]
    width_count = 0
    height_count = 0
    img = Image.open(image_path)
    img_width, img_height = img.size
    for i in range(0, img_height, HEIGHT):
        for j in range(0, img_width, WIDTH):
            box = (j, i, j + WIDTH, i + HEIGHT)
            a = img.crop(box)
            new_file_name = image_filename + "-{}-{}.".format(width_count, height_count) + image_extension
            a.save(os.path.join(SPLITTED_DATA_SET_PATH, new_file_name))
            width_count += 1
        width_count = 0
        height_count += 1


def split_csv(csv_path):
    csv_filename = get_filename(csv_path)
    with open(csv_path, 'r') as file:
        data = file.readlines()
        for line in data:
            split_line = line.strip('\n').split(',')
            x_pos = int(split_line[CSV_X_POS])
            y_pos = int(split_line[CSV_Y_POS])
            rational_x_pos = x_pos % WIDTH
            rational_y_pos = y_pos % HEIGHT
            w_pos = int(split_line[CSV_W_POS])
            h_pos = int(split_line[CSV_H_POS])

            width_count = int(x_pos / WIDTH)
            height_count = int(y_pos / HEIGHT)

            # Fix the frames of a sliced bugs
            if rational_x_pos + w_pos > WIDTH:
                split_line[CSV_W_POS] = WIDTH - rational_x_pos
            if rational_y_pos + h_pos > HEIGHT:
                split_line[CSV_H_POS] = HEIGHT - rational_y_pos

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


def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)
    for box,score in zip(target['boxes'],target['scores']):
        x, y, width, height = box[0].cpu().numpy(), box[1].cpu().numpy(), (box[2] - box[0]).cpu().numpy(), (box[3] - box[1]).cpu().numpy()
        score = score.item()
        color = (score,0,0)
        if score > 0.8:
            color = (0,0,score)
        if score > 0.9:
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
    plt.show()

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

if __name__ == '__main__':
    # pass
    # generateAllDataSets(DATA_SET_PATH, onlyDetection=True)
    # if not os.path.isdir(SPLITTED_DATA_SET_PATH):
    #     os.mkdir(SPLITTED_DATA_SET_PATH)
    # split_images()
    fixIncorrectSplittedCsv()


