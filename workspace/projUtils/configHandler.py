import os
from workspace.projUtils.Singleton import Singleton
from configparser import ConfigParser

CONFIGPATH = os.path.join(os.path.abspath(__file__ + "/../../"), "config.dat")

class ConfigHandler(Singleton):
    def __init__(self, configFilePath):
        if os.path.exists(configFilePath):
            self.config = ConfigParser()
            self.config.read_file(open(configFilePath))
        else:
            print("config does not exist !")
            raise FileNotFoundError

    def isOnlyOutput(self):
        if self.config.has_option('General', 'onlyOutput'):
            return self.config.getboolean('General', 'onlyOutput')
        else:
            return False

    def isNewModelCreationEnabled(self):
        if self.config.has_option('General', 'makeModelFromScratch'):
            return self.config.getboolean('General', 'makeModelFromScratch')
        else:
            return True

    def getExistingModelPath(self):
        if self.config.has_option('General', 'pathToModelToLoad'):
            return self.config.get('General', 'pathToModelToLoad')
        else:
            return ""

    def getInputImages(self):
        if self.config.has_option('General', 'inputImages'):
            return self.config.get('General', 'inputImages')
        else:
            return os.path.join(os.path.abspath(__file__ + "/../../../"), "inputImages")

    def getIsOnlyDetect(self):
        if self.config.has_option('HyperParameters', 'onlyDetection'):
            return self.config.getboolean('HyperParameters', 'onlyDetection')
        else:
            return True

    def getEpochAmount(self):
        if self.config.has_option('HyperParameters', 'epochs'):
            return self.config.getint('HyperParameters', 'epochs')
        else:
            return 3

    def getIouThreshold(self):
        if self.config.has_option('HyperParameters', 'iouThreshold'):
            return self.config.getfloat('HyperParameters', 'iouThreshold')
        else:
            return 0.01


    def getPrecisionRecallIOUs(self):
        if self.config.has_option('HyperParameters', 'precisionRecallIouValues'):
            listStr = self.config.get('HyperParameters', 'precisionRecallIouValues')
        else:
            listStr = "0.01"
        return [float(val) for val in listStr.split(", ")]

    def getTestSplit(self):
        if self.config.has_option('HyperParameters', 'testSplit'):
            return self.config.getfloat('HyperParameters', 'testSplit')
        else:
            return 0.2

    def getImageBatchSize(self):
        if self.config.has_option('HyperParameters', 'imageBatchSize'):
            return self.config.getint('HyperParameters', 'imageBatchSize')
        else:
            return 10

    def getImageWidth(self):
        if self.config.has_option('HyperParameters', 'splitImageWidthInPixels'):
            return self.config.getint('HyperParameters', 'splitImageWidthInPixels')
        else:
            return 2000

    def getImageHeight(self):
        if self.config.has_option('HyperParameters', 'splitImageHeightInPixels'):
            return self.config.getint('HyperParameters', 'splitImageHeightInPixels')
        else:
            return 2000

    def getRetangaleOverlap(self):
        if self.config.has_option('HyperParameters', 'rectangleOverLapLimitInPixels'):
            return self.config.getint('HyperParameters', 'rectangleOverLapLimitInPixels')
        else:
            return 150

    def getDoPrecissionRecall(self):
        if self.config.has_option('HyperParameters', 'performPrecisionRecall'):
            return self.config.getboolean('HyperParameters', 'performPrecisionRecall')
        else:
            return False

    def getDoEpochEvaluation(self):
        if self.config.has_option('HyperParameters', 'performEvaluate'):
            return self.config.getboolean('HyperParameters', 'performEvaluate')
        else:
            return False

    def getScoreLimitGreen(self):
        if self.config.has_option('ScoreLimits', 'green'):
            return self.config.getfloat('ScoreLimits', 'green')
        else:
            return 0.9

    def getScoreLimitBlue(self):
        if self.config.has_option('ScoreLimits', 'blue'):
            return self.config.getfloat('ScoreLimits', 'blue')
        else:
            return 0.8

    def getSaveImagesEnabled(self):
        if self.config.has_option('OutPutBehavior', 'saveImages'):
            return self.config.getboolean('OutPutBehavior', 'saveImages')
        else:
            return False

    def getTestImagesAmount(self):
        if self.config.has_option('OutPutBehavior', 'imagesCountToTest'):
            return self.config.getint('OutPutBehavior', 'imagesCountToTest')
        else:
            return 10

    def getImageDPI(self):
        if self.config.has_option('OutPutBehavior', 'imageDPI'):
            return self.config.getint('OutPutBehavior', 'imageDPI')
        else:
            return 500

    def isExportEnabled(self):
        if self.config.has_option('OutPutBehavior', 'exportModelAsPickle'):
            return self.config.getboolean('OutPutBehavior', 'exportModelAsPickle')
        else:
            return False


    def getExportModelName(self):
        if self.config.has_option('OutPutBehavior', 'ModelExportName'):
            return self.config.get('OutPutBehavior', 'ModelExportName')
        else:
            return "model"

if __name__ == '__main__':
    #for testing purposes of configHandler object
    configHandler = ConfigHandler(CONFIGPATH)
    print("verifying parameters")
    # print(configHandler.getIsOnlyDetect())
    # print(configHandler.getSaveImagesAmount())
    # print(configHandler.getTestSplit())
    # print(configHandler.getScoreLimitBlue())
    test2 = configHandler.getPrecisionRecallIOUs()
    test1 = configHandler.getIouThreshold()
    print(configHandler.getRetangaleOverlap())
