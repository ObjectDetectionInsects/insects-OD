import os.path
from workspace.projUtils.Singleton import Singleton
from configparser import ConfigParser

CONFIGPATH = os.path.join(os.getcwd(), "config.dat")

class ConfigHandler(Singleton):
    def __init__(self, configFilePath):
        if os.path.exists(configFilePath):
            self.config = ConfigParser()
            self.config.read_file(open(configFilePath))
        else:
            print("config does not exist !")
            raise FileNotFoundError


    def getIsOnlyDetect(self):
        if self.config.has_option('Hyper_parameters', 'onlyDetection'):
            return self.config.getboolean('Hyper_parameters', 'onlyDetection')
        else:
            return True

    def getEpochAmount(self):
        if self.config.has_option('Hyper_parameters', 'epochs'):
            return self.config.getint('Hyper_parameters', 'epochs')
        else:
            return 3

    def getTestSplit(self):
        if self.config.has_option('Hyper_parameters', 'testSplit'):
            return self.config.getfloat('Hyper_parameters', 'testSplit')
        else:
            return 0.2

    def getImageBatchSize(self):
        if self.config.has_option('Hyper_parameters', 'imageBatchSize'):
            return self.config.getint('Hyper_parameters', 'imageBatchSize')
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

if __name__ == '__main__':
    #for testing purposes of configHandler object
    configHandler = ConfigHandler(CONFIGPATH)
    print("verifying parameters")
    print(configHandler.getIsOnlyDetect())
    print(configHandler.getSaveImagesAmount())
    print(configHandler.getTestSplit())
    print(configHandler.getScoreLimitBlue())