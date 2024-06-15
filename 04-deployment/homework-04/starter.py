#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In this homework, we'll deploy the ride duration model in batch mode. Like in homework 1, we'll use the Yellow Taxi Trip Records dataset. 
# 
# You'll find the starter code in the [homework](homework) directory.
# 
# 
# ## Q1. Notebook
# 
# We'll start with the same notebook we ended up with in homework 1.
# We cleaned it a little bit and kept only the scoring part. You can find the initial notebook [here](homework/starter.ipynb).
# 
# Run this notebook for the March 2023 data.
# 
# What's the standard deviation of the predicted duration for this dataset?
# 
# * 1.24
# * 6.24
# * 12.28
# * 18.28

# In[2]:


get_ipython().system('python -V')


# In[3]:


import pickle
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# In[4]:


print(pd.__version__)


# In[5]:


get_ipython().system('pip list')


# In[6]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[7]:


# Compute Trip Duration 
categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[8]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet') # Read in the data for March 2023


# In[ ]:


# Converts the categorical features in the DataFrame to a list of dictionaries, 
# and transforms these dictionaries into a feature matrix using a DictVectorizer, 
# and the uses a trained ML model to make predictions based on the transformed feature.

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)
std_dev = y_pred.std()


# In[ ]:


print(f"Homework-04 Question 1 the Standard Deviation: {std_dev:.2f}")


# ## Q2. Preparing the output
# 
# Like in the course videos, we want to prepare the dataframe with the output. 
# 
# First, let's create an artificial `ride_id` column:
# 
# ```python
# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
# ```
# 
# Next, write the ride id and the predictions to a dataframe with results. 
# 
# Save it as parquet:
# 
# ```python
# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )
# ```
# 
# What's the size of the output file?
# 
# * 36M
# * 46M
# * 56M
# * 66M
# 
# __Note:__ Make sure you use the snippet above for saving the file. It should contain only these two columns. For this question, don't change the
# dtypes of the columns and use `pyarrow`, not `fastparquet`. 

# In[ ]:


df.head() # verify and confirm the target data with the date as requested    


# In[ ]:


year = 2023
month = 3


# In[ ]:


# Adds Ride ID column to the data frame.
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[ ]:


df.dtypes # ride type


# In[ ]:


df.head()


# In[ ]:


# write the ride ID and the predictions to a data frame with the results
df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

# Define the output file path
output_file = 'output_file.parquet'

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

# -rw-r--r-- 1 Acer 197121 68641880 Jun 12 17:52 output_file.parquet


# In[ ]:


get_ipython().system("ls -l output_file.parquet | awk '{print $5, $9}'")


# ## Q3. Creating the scoring script
# 
# Now let's turn the notebook into a script. 
# 
# Which command you need to execute for that?

# In[ ]:


get_ipython().system('jupyter nbconvert --to script starter.ipynb')


# ## Q4. Virtual environment
# 
# Now let's put everything into a virtual environment. We'll use pipenv for that.
# 
# Install all the required libraries. Pay attention to the Scikit-Learn version: it should be the same as in the starter
# notebook.
# 
# After installing the libraries, pipenv creates two files: `Pipfile`
# and `Pipfile.lock`. The `Pipfile.lock` file keeps the hashes of the
# dependencies we use for the virtual env.
# 
# What's the first hash for the Scikit-Learn dependency?
# 

# In[ ]:


import sklearn
print(sklearn.__version__)


# ## Q5. Parametrize the script
# 
# Let's now make the script configurable via CLI. We'll create two 
# parameters: year and month.
# 
# Run the script for April 2023. 
# 
# What's the mean predicted duration? 
# 
# * 7.29
# * 14.29
# * 21.29
# * 28.29
# 
# Hint: just add a print statement to your script.

# ## Q6. Docker container 
# 
# Finally, we'll package the script in the docker container. 
# For that, you'll need to use a base image that we prepared. 
# 
# This is what the content of this image is:
# ```
# FROM python:3.10.13-slim
# 
# WORKDIR /app
# COPY [ "model2.bin", "model.bin" ]
# ```
# 
# Note: you don't need to run it. We have already done it.
# 
# It is pushed it to [`agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim`](https://hub.docker.com/layers/agrigorev/zoomcamp-model/mlops-2024-3.10.13-slim/images/sha256-f54535b73a8c3ef91967d5588de57d4e251b22addcbbfb6e71304a91c1c7027f?context=repo),
# which you need to use as your base image.
# 
# That is, your Dockerfile should start with:
# 
# ```docker
# FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
# 
# # do stuff here
# ```
# 
# This image already has a pickle file with a dictionary vectorizer
# and a model. You will need to use them.
# 
# Important: don't copy the model to the docker image. You will need
# to use the pickle file already in the image. 
# 
# Now run the script with docker. What's the mean predicted duration
# for May 2023? 
# 
# * 0.19
# * 7.24
# * 14.24
# * 21.19
# 
# 
# ## Bonus: upload the result to the cloud (Not graded)
# 
# Just printing the mean duration inside the docker image 
# doesn't seem very practical. Typically, after creating the output 
# file, we upload it to the cloud storage.
# 
# Modify your code to upload the parquet file to S3/GCS/etc.

# ## Bonus: Use Mage for batch inference
# 
# Here we didn't use any orchestration. In practice we usually do.
# 
# * Split the code into logical code blocks
# * Use Mage to orchestrate the execution
# 
# ## Publishing the image to dockerhub
# 
# This is how we published the image to Docker hub:
# 
# ```bash
# docker build -t mlops-zoomcamp-model:2024-3.10.13-slim .
# docker tag mlops-zoomcamp-model:2024-3.10.13-slim agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
# 
# docker login --username USERNAME
# docker push agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
# ```
# 
# This is just for your reference, you don't need to do it.
# 
# 
# ## Submit the results
# 
# * Submit your results here: https://courses.datatalks.club/mlops-zoomcamp-2024/homework/hw4
# * It's possible that your answers won't match exactly. If it's the case, select the closest one.
# 
