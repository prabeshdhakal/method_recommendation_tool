# Use this file to train the model and save it.

from mrt_utils import train_model

data_file = "data/data_210114.xlsx" # data file name here
model_file_name = "model/model_210114.pkl" # model output file name here


train_model(data=data_file, n_neighbors=6, metric="cosine", save_file=model_file_name)