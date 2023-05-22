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
        if self.config.has_option('OutPutBehavor', 'saveImages'):
            return self.config.getboolean('OutPutBehavor', 'saveImages')
        else:
            return False

    def getTestImagesAmount(self):
        if self.config.has_option('OutPutBehavor', 'imagesCountToTest'):
            return self.config.getint('OutPutBehavor', 'imagesCountToTest')
        else:
            return 10

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

    def getImageDPI(self):
        if self.config.has_option('OutPutBehavor', 'imageDPI'):
            return self.config.getint('OutPutBehavor', 'imageDPI')
        else:
            return 500

if __name__ == '__main__':
    #for testing purposes of configHandler object
    configHandler = ConfigHandler(CONFIGPATH)
    print("verifying parameters")
    # print(configHandler.getIsOnlyDetect())
    # print(configHandler.getSaveImagesAmount())
    # print(configHandler.getTestSplit())
    # print(configHandler.getScoreLimitBlue())
    print(configHandler.getRetangaleOverlap())