from Model import Model
from projUtils import utils

if __name__ == '__main__':
    Model = Model(detetionOnly=True)
    Model.createDataSets(utils.SPLITTED_DATA_SET_PATH, 2000, 2000)
    Model.splitAndCreateDataLoaders()
    Model.getPreTrainedObject()
    Model.train(3)
    Model.testOurModel(5, 0.01)
