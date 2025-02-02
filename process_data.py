# import libraries
import sys
import requests
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on="id")
    
    #return the merge dataframe
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat=";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # lambda function is applied that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split("-")[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # change datatype of columns to string
    categories = categories.astype(str)

    for column in categories:
        # set each value to be the last character of the string
        # convert column from string to int
        categories[column] = categories[column].str[-1].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns="categories", axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # check number of duplicates
    print(df.duplicated().sum())
    # drop duplicates
    df = df.drop_duplicates()

    df = df.drop(columns=["original"])

    # check for null values
    print(df.isna().sum())
    #remove null values
    df = df.dropna()

    #check if the values in the columns are only 1s and 0s
    for column in df.columns:
        print(df[column].value_counts())

    #filter out the rows with value 2
    df = df[df['related'] != 2]
    
    #return the cleaned dataframe
    return df


def save_data(df, database_filename):
    #save clean dataset into sqlite database
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql("disaster_response", engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()