import os
import glob
from PIL import Image

class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        return AttributeError

def enum(**enums):
    return type('Enum', (), enums)


SPECIMEN_FAMILIES = enum(Curculionidae=0, Gelechiidae=1,
                        GeneralBeetles=2, GeneralMoth=3,
                        GeneralParasitoidWasp=4, UnknownFamily=5)

SPECIMEN_FAMILIES_STR = Enum(['Curculionidae',
                              'Gelechiidae',
                              'Generalbeetles',
                              'Generalmoth',
                              'Generalparasitoidwasp',
                              'Unknownfamily'])

WIDTH = 2000
HEIGHT = 2000
CSV_WIDTH_POS = 1
CSV_HEIGHT_POS = 2
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

def generateDataSetFromSingleCsv(csvFilePath):
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
                    if not splittedLine[18] == "":
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


def generateAllDataSets(dataSetsPath):
    if os.path.isdir(dataSetsPath):
        for folder in os.listdir(dataSetsPath):
            currentDir = os.path.join(dataSetsPath, folder)
            csvFiles = glob.glob(os.path.join(currentDir, '*.{}'.format(CSV_EXTENSION)))
            for csvfile in csvFiles:
                generateDataSetFromSingleCsv(os.path.join(currentDir, csvfile))
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
    path = r'/Users/idoyacovhai/UniversityProject/insects-OD/DataSets/test' # to fix
    csv_filename = get_filename(csv_path)
    with open(csv_path, 'r') as file:
        data = file.readlines()
        for line in data:
            split_line = line.split(',')
            width_count = int(int(split_line[CSV_WIDTH_POS]) / WIDTH)
            height_count = int(int(split_line[CSV_HEIGHT_POS]) / HEIGHT)
            update_csv_filename = csv_filename + "-{}-{}.".format(width_count, height_count) + CSV_EXTENSION
            update_csv_path = os.path.join(SPLITTED_DATA_SET_PATH, update_csv_filename)
            update_line = split_line
            update_line[CSV_WIDTH_POS] = int(split_line[CSV_WIDTH_POS]) % WIDTH
            update_line[CSV_HEIGHT_POS] = int(split_line[CSV_HEIGHT_POS]) % HEIGHT
            str_update_line = [str(i) for i in update_line]
            with open(update_csv_path, 'a') as file_to_update:
                file_to_update.write(",".join(str_update_line))


def split_images():
    if os.path.isdir(DATA_SET_PATH):
        for folder in os.listdir(DATA_SET_PATH):
            current_dir = os.path.join(DATA_SET_PATH, folder)
            csv_files = glob.glob(os.path.join(current_dir, '*.{}'.format(CSV_EXTENSION)))
            jpg_files = glob.glob(os.path.join(current_dir, '*{}'.format(JPG_EXTENSION)))
            print("Found the following files to split: ", ", ".join(csv_files))
            for csv_file in csv_files:
                for jpg_file in jpg_files:
                    if get_filename(csv_file) == get_filename(jpg_file):
                        split_csv(csv_file)
                        split_image(jpg_file)
            if len(jpg_files) > 0:
                split_image(jpg_files[0])


if __name__ == '__main__':
    pass
    #generateAllDataSets(DATA_SET_PATH)
    #split_images()


