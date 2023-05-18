from Model import Model
from projUtils import utils

if __name__ == '__main__':
    Model = Model()
    Model.createDataSets(utils.SPLITTED_DATA_SET_PATH, 2000, 2000)
    Model.splitAndCreateDataLoaders()
    #Temporary behavior - using existing model that is modified for our needs
    Model.getPreTrainedObject(2)
    Model.train(1)
    Model.testOurModel(3, 0.01)
