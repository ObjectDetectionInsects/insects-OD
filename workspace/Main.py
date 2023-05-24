from Model import Model
from projUtils.configHandler import ConfigHandler, CONFIGPATH

if __name__ == '__main__':
    configObject = ConfigHandler(CONFIGPATH) #initial creation of config object - it is a singleton!
    Model = Model()
    Model.getPreTrainedObject()
    if not configObject.isOnlyOutput():
        Model.createDataSets()
        Model.splitAndCreateDataLoaders()
        Model.train()
        Model.testOurModel()
        if configObject.isExportEnabled():
            Model.export()
        if configObject.getDoPrecissionRecall():
            print("Generating confusion matrix - this might take a while!")
            Model.calculate_precision_recall()
    else:
        Model.exportSingleInsect()
