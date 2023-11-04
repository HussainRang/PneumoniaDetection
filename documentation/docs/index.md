# PROJECT STRUCTURE

## STRUCTURE OVERVIEW
The project has 4 main directories: <br/>

📦Pneumonia Detection Capstone Project <br/>
 ┣ 📂analysis <br/>
 ┣ 📂data <br/>
 ┣ 📂documentation <br/>
 ┣ 📂src <br/>

We will look about what each directory contains.

<br/>

---


## DATA

The `data` folder contains the data for training the models. It has structure as

📦data <br/>
 ┣ 📂images

The `images` folder contains the classes of the dataset as directories

📦data <br/>
 ┣ 📂images <br/>
 ┃ ┣ 📂BAC_PNEUMONIA <br/>
 ┃ ┣ 📂NORMAL <br/>
 ┃ ┗ 📂VIR_PNEUMONIA <br/>

For example , my dataset has 3 categories: <br/>
- **BAC_PNEUMONIA** : bacterial pneumonia <br/>
- **NORMAL** : no such disease <br/>
- **VIR_PNEUMONIA** : viral pneumonia <br/>

So each category is represented as a directory in the `images` directory and each of these directory then contains the images of their respective class.

<br/>

---


## ANALYSIS
The `analysis` directory contains the `analysis.ipynb` file which contains the code for analysis which is done using matplotlib , seaborn and plotly. The purporse of this analysis was : <br/>
- Getting familiar with the dataset <br/>
- To see if some patterns could be extracted which could help the models to easily predict on the dataset<br/>
- To know about number of images of each class to check if our dataset is imbalanced 

<br/>

---



## DOCUMENTATION
The `documentation` contains the documentation for the project.

<br/>

---


## SRC
This is the main folder of the project which contains all the code for training the models and logging it.The structure of this directory is <br/>
📦src <br/>
 ┣ 📂logs <br/>
 ┣ 📂mlruns <br/>
 ┣ 📂pipelines <br/>
 ┣ 📂steps <br/>
 ┣ 📂utils <br/>
 ┣ 📜config.py <br/>
 ┗ 📜main.py <br/>

- **logs** : The first `logs` directory stores the logs for the run. It is recreated for every run. You don't have to create this directory manually, it will be created automatically when you run the code. 

- **mlruns** : It stores the metrics , models and artifacts for the run and displays all the data in a website when MLFlow is executed using the command `mlflow ui`.This folder would be created on its own when pipeline is run and `LOG_TO_MLFLOW=True` in the `config.py` file. It would store all the previous data for all the run when we have logged to MLFLow and It would not recreate the folder again.

- **pipelines** : This directory stores the all pipelines which are made to train the model. All the essential steps like fetching the data, fetching the model would called from here.

- **steps** : This folder consists of code files where steps are performed like loading images , fetching the model etc. These steps are implemented only in the `pipelines`. The files in this directory are <br/>
    📂steps <br/>
    ┣ 📜dimensionality_reduction.py <br/>
    ┣ 📜get_metrics.py <br/>
    ┣ 📜get_model.py <br/>
    ┣ 📜image_loader.py <br/>
    ┣ 📜label_encoding.py <br/>
    ┣ 📜mlflow_logging.py <br/>
    ┣ 📜randomized_search_cv.py <br/>
    ┣ 📜scaling.py <br/>
    ┗ 📜split_data.py <br/>
Here is a brief description of all the files: <br/>
    - `dimensionality_reduction.py` : This file contains a dictionary of all the dimensionality reduction algorithms with the key as algorithm name. Specified dimensionality reduction algorithm is performed on the `X_train` and `y_train` and the data is compressed and then passed to the further steps.
    - `get_metrics.py` : This file instantiates `MetricsCalculator.py` class in `utils` and it collects all the specified metrics and puts them into a dictionary and the returns them.
    - `get_model.py` : This file contains a dictionary where the algorithm names are specified as keys and the models machine learning models are specified as values. It fetches the specified model from the dictionary returns it to the `pipeline`.
    - `image_loader.py` : This file instantiates `ImageLoader.py` class in `utils`. This file takes in the image names and passes it to the class and loads the images and returns them to the `pipeline`.
    - `label_encoding.py` : This file encodes the labels and converts them to numbers and then returns them to the `pipeline`.
    - `mlflow_logging.py` : This file contains the logging setup for MLFlow and it logs the metrics, parameters ,models and confusion matrix as an artifact to MLFlow.
    - `randomized_search_cv.py` : This file takes in the model and parameters and then returns the best_parameters and the best model to the `pipeline`.
    - `scaling.py` : This file contains the Scalers present in the `DataScaler.py` in utils. It scales or standardizes the training and testing data and then passes them back to the `pipeline`.
    - `split_data.py` : This file takes in the complete data and creates the training and testing set and returns them to the `pipeline`. It uses the `DataSplitter.py` class for splitting the dataset.  

- **utils** : This folder contains all the functions and classes which are just extra services or helping in the completion of the `steps` . The files present in this directory are: <br/>
📂utils <br/>
┣ 📜create_metadata_file.py <br/>
┣ 📜DataScaler.py <br/>
┣ 📜DataSplitter.py <br/>
┣ 📜DimensionalityReduction.py <br/>
┣ 📜get_class_weights.py <br/>
┣ 📜ImageLoader.py <br/>
┣ 📜linearize_images.py <br/>
┣ 📜MetricsCalculator.py <br/>
┣ 📜plot_confusion_matrix.py <br/>
┗ 📜setup_logs.py <br/>
Here is a brief description of all the files in the project: <br/>
    - `create_metadata_file.py` : This function creates a `.csv` file in the `data` directory which contains all the names of the images    and its corresponding category.
    - `DataScaler.py` : This file contains a template for the scaler class which is inherited by Data Scaling alogrithm classes present in the file.
    - `DataSplitter.py` : This file contains the `DataSplitter` class which implements `train_test_split` of sklearn to split the dataset into training and testing data.
    - `DimensionalityReduction.py` : This file contains the `DimensionalityReductionTemplate` which is inherited by the classes implementing the dimensionality reduction algorithm. These classes takes the data and perform their respective data dimensionality reduction algorithm and then return the data back.
    - `get_class_weights.py` : This file contains the algorithms implemented for calculating the class weights according to the parameter given to them.
    - `ImageLoader.py` : This file contains `ImageLoader` class uses the opencv module to load the images with their paths given and the return them back.
    - `linearize_images.py` : This file contains function to convert the 2D image to a 1D image by flattening it.
    - `MetricsCalculator.py` : This file contains `MetricsCalultor` class which inherits `MetricsCalculatorTemplate` class and implements methods which return their corresponding metric when called.
    - `plot_confusion_matrix.py`: This file takes in the confusion matrix and saves the plot which is logged to the MLFlow.
    - `setup_logs.py` : This file creates the `logs` directory if it does not exists.

- **config.py** : This is the most important file which contains the complete configuration used by the project to to run and initialize the models,scalers etc. Complete components of this file would be explained [here](http://127.0.0.1:8000/ConfigFile/).

- **main.py** : This file runs the complete `pipeline` of the project. 