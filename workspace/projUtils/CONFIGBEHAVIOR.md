# Config parameters explanations

The config file has multiple sections to separate different parameters based on their intended use

Sections:
1. General - holds parameters related to the basic behavior of the project during run 
2. HyperParameters - holds parameters in charge of changing the way in which we create and train the model
3. PrecisionRecall - holds parameters related only for the creation of the confusion matrixs
4. ScoreLimits - parameters for images notation and output score filtering
5. OutPutBehavior - control the output of the project


### Section 1: General

1. <u>onlyOutput</u> - [Type: Boolean, default: False]
   - if True - will overpass entire model creation and testing phase -
    will only generate output based on new input images supplied.
    intended to be used on "production"

2. <u>InputImages</u> - [Type: Path String, default: rootdir + inputimages]
   - if <i>onlyOutput</i> is True - from this given path the images will be loaded to generate prediction on
3. <u>makeModelFromScratch</u> - [Type: Boolean, default: True]
   - if True - the new model will be generated based on the basic fasterRCNN model
4. <u>pathToModelToLoad</u> - [Type: Path String, default: """]
   - if <i>makeModelFromScratch</i> is False - will load a pretrained model located on the computer for retraining or evaluation


### Section 2: HyperParameters

1. <u>onlyDetection</u> - [Type: Boolean, default: True] 
   - if True - both the data handler and the model object will work with only 2 classes - background and an insect. 
2. <u>performEvaluate</u> - [Type: Boolean, default: False] 
   - if True - on each cycle of training will perform evaluation on the testing data set
3. <u>performLossPerEpoch</u> - [Type: Boolean, default: False] 
   - if True - will collect the loss values collected from each epoch and generate a graph
4. <u>epochs</u> - [Type: integer, default: 3] 
   - for how many cycles to train our model 
5. <u>imageBatchSize</u> - [Type: integer, default: 10] 
   - defines the amount of images to be loaded at once during each epoch(keep low for slow computers)
6. <u>splitImageWidthInPixels</u> - [Type: integer, default: 2000] 
   - width in pixels of images in dataSet - both for preliminary creation of data and for dataloaders
7. <u>splitImageHeightInPixels</u> - [Type: integer, default: 2000]
   - height in pixels of images in dataSet - both for preliminary creation of data and for dataloaders
8. <u>iouThresholdForOverlap</u> - [Type: float, default: 0.5]
   - during testing of our model will delete overlapping boxes with this overlap 


### Section 3: PrecisionRecall

1. <u>performPrecisionRecall</u> - [Type: Boolean, default: 10]  
   - if true - will perform precision recall (long process) 
2. <u>iouThreshold</u> - [Type: float, default: 0.5] 
   - overlap threshold between found (prediction) box and actual box for TP 
3. <u>precisionRecallMinConfidence</u> - [Type: float, default: 0.01]
   - minimum confidence for precision recall generation
4. <u>precisionRecallMaxConfidence</u> - [Type: float, default: 1.0] 
   - maximum confidence for precision recall generation
5. <u>getPrecisionRecallConfidenceSteps</u> - [Type: float, default: 0.1] 
   - steps between minimum and maximum to take - for example:
   
   minimum = 0 maximum = 1.0 steps = 0.25 will generate [0, 0.25, 0.5, 0.75, 1.0] confidence list for run

### Section 4: ScoreLimits

1. <u>green</u> - [Type: float, default: 0.9] 
   - the score for model's output from which the bounding box will be colored greed 
2. <u>blue</u> - [Type: float, default: 0.8] 
   - the score for model's output from which the bounding box will be colored blue 
3. <u>minConfidence</u> - [Type: float, default: 0.5] 
   - the score limit for boxes to keep - boxes with score lower than minConfidence will be deleted from prediction

### Section 5: OutPutBehavior

1. <u>saveImages</u> - [Type: Boolean, default: False] - 
   - if True - during testing phase of model creation the tested images will be saved to the disk
2. <u>imagesCountToTest</u> - [Type: integer, default: 10]  - 
   - amount of images to test our model on from verification image set 
3. <u>imageDPI</u> - [Type: integer, default: 500]  - 
   - the dot per inch or image quality of the output 
4. <u>exportModelAsPickle</u> - [Type: Boolean, default: False] -
   - if True - the model created during the project execution will be saved to disk and can be used for future predictions or retraining
5. <u>ModelExportName</u> - [Type: String, default: "model"] -
   - the name for the export model pickle file