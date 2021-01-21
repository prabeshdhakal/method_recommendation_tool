# Methods Recommendation Tool

A k-Nearest Neighbor and Dash/Plotly based tool that creates profile of methods selected by the user and recommends other methods that they can learn based on their preferences.

Main method used: [k-Nearest Neighbors Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)  
Main libraries used: `Dash`, `Plotly`, `scikit-learn`, and `pandas`

This project was created for the lab of [Prof. Dr. Henrik von Wehrden](https://henrikvonwehrden.web.leuphana.de/henrik-von-wehrden/) where he and his team made the tool available to students in a methods based course.

## Contents

1. [Environment](#environment)
1. [Running the App](#running-the-app)
1. [Training a new kNN model](#training-a-new-knn-model)
1. [Updating the App](#updating-the-app)
1. [TODOs](#todos)

## Environment

Developed and tested on a **Python 3.7** conda environment with the installations listed in `requirements.txt`. Creating a new conda environment to load/test this project using `requirements.txt` is strongly advised.

## Running the App

After ensuring that all the packages listed in `requirements.txt` are installed, run the following in the terminal:

`python app.py`

On your web browser, visit `localhost:8050`.

## Training a new kNN model

1. Ensure that the data that you use matches `data/data_210114.xlsx` exactly.
1. Open `get_trained_model.py` and update the path to the data file and output model file name.
1. On the terminal, run `python get_trained_model.py`.
1. Ensure that the saved model is placed in the `models` directory.

## Updating the App

To run the app with the newly trained model based on the data used to train the model, open `app.py` and update `data_path` and `model_path` in lines 21 and 22. Then save the file and run the following on your terminal:

`python app.py`

On your web browser, visit `localhost:8050`.

## TODOs

1. [x] Put the project on GitHub
1. [ ] Make the font is consistent between MRT and Sustainability Methods Wiki (manual tweaking of the bootstrap theme)
1. [x] For the bottom left plot: reverse the color
1. [x] Change legend of the top plot to: "Profile of {}".format(selected_method)
1. [x] Center the text in the table
1. [ ] Ensure that the selected_method != recommended_method
