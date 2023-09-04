# deep-learning-challenge

#### The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

#### From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

#### Using Pandas and scikit-learn's StandardScaler, the datase was processed by compiling, training, and evaluating the neural network model. The data targets and features for my model were identified and unnecessary columns were dropped. The number of data points for each unique value were determined using any columns that had more than 10 unique values. Using the number of data points for each unique value, a cutoff point was identified to bin "rare" categorical variables toghether in a new value and then checked to ensure the binning was successful. The categorical variables were encoded using pd.get_dummies to create binary data. The data was split into a features array (X), and a target array (y), which were used with the train_test_split function to split the data into training and testing datasets. The training and testing features datasets were scaled using the StandardScaler instance, fit to the training data, and then using the transform function.

#### A neural network was designed using TensorFlow to create a binary classification model to predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. Inputs to the model were determined for the number of neurons and layers for the model. Once the model was developed the data was compiled, trained, and evaluated to calculated the model's loss and accuracy. A callback was created to save teh model's weights for every five epochs. The model was evaluated using the test data to determine the loss and accuracy with the results being save and exported to a HDF5 file.

#### The initial model returned an accuracy lower that 75%. Therefore, using the Keras Tuner, the model was optimized to return a higher accuracy rate. A new notebook was created to preprocess the data, create the Keras Tuner model and compile, Train, and Evaluate the model. The results were exported to an HDF5 file.

## Alphabet Soup Charity Neural Network Model Report

### Overview of the analysis:

#### The primary purpose of this analysis was to assist the Alphabet Soup Charity in determining the potential success of applicants requesting funding. By leveraging machine learning and neural network techniques, I set out to create a binary classifier that would predict if an organization would effectively use the funds granted by Alphabet Soup based on the features provided in the dataset.

### Results:

#### Data Preprocessing

- Target Variable(s) for the model:
  - IS_SUCCESSFUL
- Features for the model:
  - APPLICATION_TYPE
  * AFFILIATION
  * CLASSIFICATION
  * USE_CASE
  * ORGANIZATION
  * STATUS
  * INCOME_AMT
  * SPECIAL_CONSIDERATIONS
  * ASK_AMT

* Variables to be removed from the input data:
* EIN and NAME (As these are merely identification columns and do not contribute to predicting the success of funding.)

#### Compiling, Training, and Evaluating the Model

- Neurons, Layers, and Activation Functions:
- Activation Function: Relu
- Number of Layers: 4
- First Layer Neurons: 76
- Second Layer Neurons: 21
- Third Layer Neurons: 11
- Fourth Layer Neurons: 21
- Fifth Layer Neurons: 6

The architecture was suggested by the KerasTuner, aiming to optimize model accuracy by tweaking various hyperparameters.

- Achieving Target Model Performance:
- The best model achieved an accuracy of approximately 73.08%, which falls short of the 75% target.

### Summary:

#### The deep learning model built to predict the successful use of Alphabet Soup's funds by applicants achieved an accuracy of 73.08%. While this model offers a foundation and certain predictive power, there is potential for improvement.

#### Recommendation:

#### A potential approach to further enhance this classification problem would be to:

- Further feature engineering: Dive deeper into the features to create new relevant features.
- Collect additional data: Sometimes, the features present may not be sufficient to achieve the desired accuracy, and collecting additional relevant data could prove beneficial.
- Iteratively refine and validate the model using techniques such as cross-validation, which can offer a more robust measure of the model's performance on unseen data.

In conclusion, while the current deep learning model serves as a good starting point, exploring other machine learning techniques and refining the current approach could yield even better results.
