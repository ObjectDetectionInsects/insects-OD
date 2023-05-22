from Model import Model
from projUtils.configHandler import ConfigHandler, CONFIGPATH

if __name__ == '__main__':
    configObject = ConfigHandler(CONFIGPATH) #initial creation of config object - it is a singleton!
    Model = Model()
    Model.createDataSets()
    Model.splitAndCreateDataLoaders()
    Model.getPreTrainedObject()
    Model.train()
    Model.testOurModel(0.01)
    if configObject.getDoPrecissionRecall():
        print("Generating confusion matrix - this might take a while!")
        Model.calculate_precision_recall()