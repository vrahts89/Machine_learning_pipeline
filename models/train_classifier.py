import sys
# Import libraries
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# Download necessary NLTK data
nltk.download(['stopwords', 'wordnet', 'punkt'])

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM disaster_response", engine)
    X = df["message"]
    Y = df.drop(columns=["message", "genre", "id"], axis=1)
    return X, Y, Y.columns

def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    #instatiate lemmatizer  
    lemmatizer = WordNetLemmatizer()
    
    #remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # Reduce words to their stems
    tokens = [PorterStemmer().stem(w) for w in tokens]
    
    # lemmatize words to their root form
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


def build_model():
    #create a machine learning pipeline
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(estimator=None))
        ])
    
    #create a set of parameters that can be used for the gridsearch
    parameters = {
        'clf__estimator': [RandomForestClassifier(), GradientBoostingClassifier(), AdaBoostClassifier()],
        'clf__estimator__n_estimators': [50, 100],
        'vect__max_df': [0.5, 1.0],
        'tfidf__use_idf': [True, False]
    }

    # Create GridSearchCV
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2)

    #return the model
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # Print best model
    print("Best parameters:", model.best_params_)
    print("Best score:", model.best_score_)
    
    # Predict on test data with best estimator
    y_pred = model.predict(X_test)

    # Test model for every column
    for i in range(len(Y_test.columns)):
        #print(f"Classification Report for {Y_test.columns[i]}:")
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    # Export model as a pickle file

    # Save best model as a variable
    best_model = model.best_estimator_

    # Save the model to a file
    with open(model_filepath, 'wb') as file:
        pickle.dump(best_model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    
