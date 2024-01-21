# Disaster Response Machine Learning Pipeline

### Table of Contents

1. [Project Overview](#project-overview)
2. [File Description](#file-descriptions)
3. [Instructions](#instructions)
4. [ETL Pipeline](#etl-pipeline)
5. [ML Pipeline](#ml-pipeline)
6. [Flask Web App](#flask-web-app)
7. [Author and Acknowledgements](#authors-and-acknowledgements)

## Project Overview<a name="project-overview"></a>

In this project, we have built a machine learning pipeline to categorize emergency messages based on the needs communicated by the sender. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

The project is divided into the following key sections:

ETL Pipeline: An Extract, Transform, and Load process that cleans data and stores it in a SQLite database.
Machine Learning Pipeline: A machine learning pipeline that trains a model to classify text message into categories.
Flask Web App: A web application that visualizes the results of the models and allows input of new messages for classification.

## File Descriptions<a name="file-descriptions"></a>

- app
  | - template
  | |- master.html # main page of web app
  | |- go.html # classification result page of web app
  |- run.py # Flask file that runs app

- data
  |- disaster_categories.csv # data to process
  |- disaster_messages.csv # data to process
  |- disaster_response.db # database to save clean data to

- models
  |- train_classifier.py # script for ML pipeline
  |- classifier.pkl # saved model

- README.md
- process_data.py # script for ETL pipeline

## Instructions<a name="instructions"></a>

### Running the Web App from the Project Workspace IDE:

Open a new terminal window. You should already be in the workspace folder, but if not, then use terminal commands to navigate inside the folder with the run.py file.

Type in the command line:
python run.py

### Running the ETL pipeline:

To run the ETL pipeline that cleans and stores the data in a database, navigate to the data folder and run the following command:
python process_data.py disaster_messages.csv disaster_categories.csv disaster_response.db

### Running the ML pipeline:

To run the ML pipeline that trains the classifier and saves the model, navigate to the models folder and run the following command:
python train_classifier.py ../data/DisasterResponse.db classifier.pkl

## ETL Pipeline<a name="etl-pipeline"></a>

The ETL pipeline is implemented in the process_data.py file. It performs the following steps:

Loads the messages and categories datasets.
Merges the two datasets.
Cleans the data.
Stores it in a SQLite database.

## ML Pipeline<a name="ml-pipeline"></a>

The Machine Learning pipeline is implemented in the train_classifier.py file. It performs the following steps:

Loads data from the SQLite database.
Splits the dataset into training and test sets.
Builds a text processing and machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV.
Trains and tunes a model using GridSearchCV.
Outputs results on the test set.
Exports the final model as a pickle file.

## Flask Web App<a name="flask-web-app"></a>

The Flask web app provides an interface for an emergency worker to input a new message and get classification results in several categories. It also displays visualizations of the dataset.

1. It shows the mean amount of tokens per category
2. It shows a scatter plot with the tfidf_score per token for any category. It is initialized for the category 'tools' but can be shown for any other category by changing the variable 'category' in run.py

The web app is modular and allows for easy integration of the machine learning model and the dataset.

## Authors and Acknowledgements<a name="authors-and-acknowledgements"></a>

Please acknowledge Figure Eight for providing the relevant dataset for this project.
Author: Valentin Rahts
