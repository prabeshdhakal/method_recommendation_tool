# Use this file to train the model and save it.
import os
os.chdir(".")
import pickle
from mrt_utils import train_model

data_file = "data_210128.xlsx" # data file name here
model_file_name = "model_210128.pkl" # model output file name here


train_model(data_file=data_file, n_neighbors=6, metric="cosine", save_file=model_file_name)