return to home readme [here](../../README.md)
## Data preperation

This read me will go over steps of data preparation for the expected format of the project
general steps:
1. splitting full csv into a csv for each image and transforming the data based on detection or classification 
2. splitting each csv to sub images based on dimension of width and height from config
3. split the now ready data into train-test-validation split

### please notice:

- all steps are for data manipulation under workspace\projUtils\utils.py 
- data manipulation is mandatory for the correct flow of the project 
- expected DataSet pet is one above "workspace" - under the folder DataSets



### steps 


 <b><u>1: Splitting initial dataset csv file</u></b>

calling the function generateAllDataSets.\
this function will go over each sub-folder under DataSet dir and then under each csv file in that folder  
the function will split the csv based on the column :<i>"parent_image_file_name"</i> to multiple CSVs with corresponding names

<u>Preliminary expectations and parameters:</u>\
the function also changes the csv file and removes unnecessary columns.\
the expected input csv stracture is :
- columns 1 - x location
- column 2 - y location
- column 3 - width
- column 4 - height
- column 6 - associated image
- column 18 - specimen family 

if the "onlyDetection" parameter in config is True all specimen family will be marked as 1
read more about config parameters [here](./CONFIGBEHAVIOR.md)

Specimen family is taken from predefined enum in the code if the family is not predefined there - the family will be marked as unknown

* after this step - move preliminary data set as it is no longer needed

\
\
 <b><u>2: Splitting csv-per-image to sub images</u></b>
 
calling the function "split_images".
this function will go over each csv - image pair in DataSets dir and split them into sub images based on predefined size\
all sub images will be located in SplittedDataSet dir - which will be the main directory used for the project

<u>Preliminary expectations and parameters:</u>\
the function expects the naming of each csv-image pair to be X.csv and X.jpg this should already be handled be step 1\
don't skip step 1 directly to this step as it won't work 

the size of the images as determined by two parameters in the config file:
1. "splitImageWidthInPixels"
2. "splitImageHeightInPixels"
read more about config parameters [here](./CONFIGBEHAVIOR.md)

* notice - after the execution of this function call "fixIncorrectSplittedCsv" which will verify and delete any incorrect splitting done

\
\
 <b><u>3: Splitting the ready data set into Train/Test/Validation</u></b>


calling the function "split_train_test_validation".
This function will the data set as such:
- 60% - train dir
- 20% - test dir
- 20% - validation dir\
the split is hardcoded! -  the ability to modify this behavior can be implemented using the configHandler

the function will loop on each image-csv pair and will place them in one of three dirs:\
TrainSet/ TestSet/ ValidationSet\
**~important~** - the project requires the data to be located in these folders for each dataHandler object - do not skip this step 

the way in which the split is performed is :
```
    loop until finished all files:
    |
    - place 5 images in train
    - place 1 image in test
    - place 1 image in validation
```
this behavior was implemented inorder to mix the given dataSet 


thank you! \
return to home readme [here](../../README.md)