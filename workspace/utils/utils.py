import os
import glob

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

CSV_EXTENSION = 'csv'
CSV_DELIMETER = ','
JPG_EXTENSION = '.jpg'
TABLE_HEADER = "parent_image_file_name"
DATA_SET_PATH = os.path.join(os.path.abspath(__file__ + "/../../../"), "DataSets")
SPACE = " "


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
                    newCsvName = "{}.csv".format(associatedImage.split(JPG_EXTENSION)[0])
                    if not splittedLine[18] == "":
                        specimenFamily = getSpecimenFamily(splittedLine[18].replace(SPACE, ""))
                        lineForCsv = "{},{},{},{},{}\n".format(specimenFamily,splittedLine[1],splittedLine[2],splittedLine[3],splittedLine[4])
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
            csvFiles = glob.glob(currentDir + '\\*.{}'.format(CSV_EXTENSION))
            for csvfile in csvFiles:
                generateDataSetFromSingleCsv(os.path.join(currentDir, csvfile))
    else:
        print("Path given does not exist")


if __name__ == '__main__':
    generateAllDataSets(DATA_SET_PATH)


