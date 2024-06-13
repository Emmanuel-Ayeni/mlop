#!/usr/bin/env python
# coding: utf-8

'''# Converts the categorical features in the DataFrame to a list of dictionaries, 
and transforms these dictionaries into a feature matrix using a DictVectorizer, 
and the uses a trained ML model to make predictions based on the transformed feature.'''

import os
import sys

import pickle
import pandas as pd
from datetime import date



# Load the model
def load_model(model_file):
    with open ('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


#  Trip Duration Features
categorical = ['PULocationID', 'DOLocationID']

# Read in the data for March 2023
def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


# Adds Ride ID column to the data frame.
def save_results(df, y_pred, output_file, year, month):
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    df_result['ride_id'] = df['ride_id']    

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )



def apply_model(input_file, model_file, output_file, year, month):

    print(f"reading the data from {input_file}...")
    df = read_data(input_file)

    ## Isolating stated catgeories, call dict vectorizer and predict
    dicts = df[categorical].to_dict(orient='records')

    print(f'loading the model with RUN_ID={model_file}...')
    retriv_model = load_model(model_file)
    dv = retriv_model[0]
    model = retriv_model[1]

    X_val = dv.transform(dicts)

    print(f'the model application in progress********...')
    y_pred = model.predict(X_val)
    
    ## Predication mean 
    print(f"Q5. Mean of predictions: {round(y_pred.mean(), 3)}")

    print(f'save result to {output_file}...')
    save_results(df, y_pred, output_file, year, month)

    return output_file

def run():
    # Input parameters definition

    taxi_type = sys.argv[1] # yellow
    year =  int(sys.argv[2])# 2023
    month = int(sys.argv[3]) # 3

    # Input date: year and month to get the Taxi data
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet" 
    output_file = './output_file'
    model = model.bin

    apply_model(input_file=input_file, model=model, output_file = output_file, year = year, month = month)




if __name__ == '__main__':
    run()