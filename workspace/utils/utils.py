import os
import glob
CSV_EXTENSION = 'csv'
CSV_DELIMETER = ','
JPG_EXTENSION = '.jpg'
TABLE_HEADER = "parent_image_file_name"
def generateDataSetFromSingleCsv(csvFilePath):
    newCsvFileNames = []
    if os.path.isfile(csvFilePath):
        with open(csvFilePath, 'r') as file:
            for line in file.readlines():
                splittedLine = line.split(CSV_DELIMETER)
                associatedImage = splittedLine[6]
                if not associatedImage == TABLE_HEADER:
                    newCsvName = "{}.csv".format(associatedImage.split(JPG_EXTENSION)[0])
                    lineForCsv = "{},{},{},{},{}\n".format(splittedLine[18],splittedLine[1],splittedLine[2],splittedLine[3],splittedLine[4])
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
    dataSetsPath = os.path.join(os.path.abspath(__file__ + "/../../../"), "DataSets")
    generateAllDataSets(dataSetsPath)


