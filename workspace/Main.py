from Model import Model
from projUtils import utils
from projUtils.configHandler import ConfigHandler, CONFIGPATH

if __name__ == '__main__':
    configObject = ConfigHandler(CONFIGPATH) #initial creation of config object - it is a singleton!
    Model = Model()
    Model.createDataSets(utils.SPLITTED_DATA_SET_PATH, 2000, 2000)
    Model.splitAndCreateDataLoaders()
    Model.getPreTrainedObject()
    Model.train()
    Model.testOurModel(0.01)
    Model.calculate_precision_recall()