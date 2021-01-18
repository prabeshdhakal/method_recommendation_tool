# Load saved model
import pickle

# Work with arrays/dataframes
import numpy as np
import pandas as pd

# Plotly visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Train KNN Model
from sklearn.neighbors import NearestNeighbors

# Scale the scores
from sklearn.preprocessing import MinMaxScaler

def get_methods_profile(methods_df):

    """
    Takes dataframe of methods as input and returns 
    a min-max scaled profile of methods.

    Parameters:
        methods_df (pandas df): the dataframe of methods whose profile is to be created.

    Returns:
        scores (dataframe): profile of the input methods
    """

    category = methods_df.columns
    
    scores_sum = np.sum(np.array(methods_df), axis=0)
    scaled_sum = MinMaxScaler().fit_transform(scores_sum.reshape(-1, 1))
    
    scores = pd.DataFrame(scaled_sum, index=category)
    scores.columns = ["score"]
    scores.index.name = "category"

    return scores

def clean_methods_names(methods_array):

    """
    Remove underscores from, capitalize the 1st letters of, and
    remove duplicates from a list of methods' names.

    Parameters:
        methods_array (list): the array of methods names that need to be cleaned.

    Returns:
        cleaned_titles (list): list of titles that are cleaned
    """
    
    # 1. remove underscore & capitalize 1st letter
    processed_titles = []
    for item in methods_array:
        item_ = " ".join(item.split("_")).title()
        processed_titles.append(item_)

    # 2. remove duplicates
    cleaned_titles = list(dict.fromkeys(processed_titles))
    
    return cleaned_titles

def predict_methods(X_pred, no_preds=1, data="data.xlsx", saved_model="model_v1.pkl"):
    
    """
    Returns the predictions made based on the inputs given a saved kNN model
    and the data that the model is based on are provided.

    ONLY 1 OR 3 PREDICTIONS CAN BE MADE.

    Parameters:
        X_pred (list):
        no_preds (int ;1 or 3): the no. of predictions to be made; 1 or 3
        data (str): path to the data file to be used
        saved_model (str): path to the kNN model to be used

    Returns:
        preds_dict (dict): a dictionary that contains the recommended methods 
                            and the profile of recommended methods
    """

    # 1. Read data file
    df = pd.read_excel(data, index_col="Method").fillna(0)
    #METHODS_LIST = df.index # names of the methods
    #CATEGORY_LIST = df.columns # names of the features (column names)

    # 2. Read saved model
    with open(saved_model, "rb") as f:
        model = pickle.load(f)

    # 3. Make prediction with the loaded model
    preds = model.kneighbors(X_pred, return_distance=True)

    # 4. Process data and return results

    # 4.1 only one method selection (for the top section)
    if no_preds == 1:

        y_preds = preds[1][0] # indices of rec. methods

        rec_methods_df = df.iloc[y_preds[1:]]
        rec_methods_list = rec_methods_df.index # list of rec. methods

        rec_methods_profile = get_methods_profile(rec_methods_df) # overall profile of rec. methods

        preds_dict = {
            "recommended_methods": clean_methods_names(rec_methods_list),
            "recommended_methods_profile": rec_methods_profile
        }

        return preds_dict

    # 4.2 three methods selected (for the bottom section)
    elif no_preds == 3:

        scores = preds[0]
        y_preds = preds[1]

        # 4.2.1 get method names sorted by values
        rec_methods_names = [df.iloc[y_preds[0]].index, df.iloc[y_preds[1]].index, df.iloc[y_preds[2]].index]

        methods_scores = dict()
        for i in range(len(scores)):
            names_ = rec_methods_names[i]
            scores_ = scores[i]

            for j in range(1, len(names_)):
                methods_scores[names_[j]] = scores_[j]
        
        methods_score_sorted = dict(sorted(methods_scores.items(), key=lambda item: item[1]))
        methods_names_sorted = list(methods_score_sorted.keys())

        # 4.2.2 get the profile of recommended methods
        
        # first item of every group of pred. is ignored
        # because they are the methods selected by the user
        y_preds_flattened = [i for lst in y_preds for i in lst[1:]]

        rec_methods_df = df.iloc[y_preds_flattened]

        #rec_methods_list = rec_methods_df.index # list of rec. methods
        rec_methods_profile = get_methods_profile(rec_methods_df) # overall profile of rec. methods

        preds_dict = {
            "recommended_methods": clean_methods_names(methods_names_sorted),
            "recommended_methods_profile": rec_methods_profile
        }
        
        return preds_dict

def rec_plot_table(input_method, input_profile, output_profile, rec_methods):

    """
    Returns plotly chart with a table stacked on top of a polar chart
    showing the profile of the methods selected by the user and the profile
    of the methods recommended by the algorithms.

    Parameters:
        input_profile (pandas df): profile of methods selected by user
        output_profile (pandas df): profile of recommended methods
        rec_methods (array): recommended methods

    Returns:
        fig (Plotly figure): a figure with a table and polar chart
    """

    # column names for the original dataframe
    DF_COLNAMES = ['Quantitative', 'Qualitative', 'Inductive', 'Deductive', 
                   'Spatial_Global', 'Spatial_System', 'Spatial_Individual', 
                   'Temporal_Past','Temporal_Present', 'Temporal_Future']
    
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.15,
        subplot_titles=["Selection: {}".format(input_method), ""],
        specs=[
            [{"type":"table"}],
            [{"type":"polar"}]
        ]
    )
    
    # Table: Recommended Methods
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Recommended Methods"],
                fill_color="lavender",
                font=dict(family="Gravitas One", size=16)
            ),
            cells=dict(
                values=[rec_methods],
                fill_color="snow",
                font=dict(family="Arial", size=14)
            ),
        ),
        row=1, col=1
    )
    
    # 2.1. Polar Plot: Input Profile (Selected)
    fig.add_trace(
        go.Scatterpolar(
            r=input_profile.score,
            theta=DF_COLNAMES,
            mode="lines+markers",
            line=dict(smoothing=1.3, shape="spline", color="rgb(220, 5, 12)"),
            fillcolor="rgb(220, 5, 12)",
            fill='toself',
            hoverinfo=None,
            opacity=0.5,
            name='Profile of Selected Methods',
            ),
            row=2, col=1
    )
    
    # 2.2. Polar Plot: Output Profile (Recommended)
    fig.add_trace(
        go.Scatterpolar(
            r=output_profile.score,
            theta=DF_COLNAMES,
            mode="lines+markers",
            line=dict(smoothing=1.3, shape="spline", color="rgb(54, 75, 154)"),
            fillcolor="rgb(54, 75, 154)",
            fill='toself',
            hoverinfo=None,
            opacity=0.5,
            name='Profile of Recommended Methods'
            ),
            row=2, col=1
    )
    
    # 3. Update Figure Layout
    fig.update_layout(
        polar=dict(
            #bgcolor="rgb(242, 242, 242)",
            bgcolor="whitesmoke",
            radialaxis=dict(
                visible=False,
                range=[0, 1],
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            #xanchor="center",
            x=0.25
        ),
        margin=dict(l=0, r=0, t=25, b=0)
    )

    return fig

def plot_profile_selected(input_profile):

    """
    Plot the polar chart that contains the profile of the selected methods
    and the profile of the recommended methods.

    Parameters:
        input_profile (pandas df): the profile of methods out of 
                                    which a polar chart is to be made

    Returns:
        fig (Plotly figure): a single polar chart of the profile of methods
                                based on the input profile
    """

    # column names for the original dataframe
    DF_COLNAMES = ['Quantitative', 'Qualitative', 'Inductive', 'Deductive', 
                   'Spatial_Global', 'Spatial_System', 'Spatial_Individual', 
                   'Temporal_Past','Temporal_Present', 'Temporal_Future']

    fig = go.Figure()

    # 1. Polar Chart: Profile of selected methods
    fig.add_trace(
    go.Scatterpolar(
        r=input_profile.score,
        theta=DF_COLNAMES,
        mode="lines",
        line=dict(smoothing=1.3, shape="spline", color="rgb(54, 75, 154)"),
        fillcolor="rgb(54, 75, 154)",
        fill='toself',
        opacity=0.8,
        name='Profile of Selected Methods',
        ),
    )
       
    # 2. Update Figure Layout
    fig.update_layout(
        polar=dict(
            #bgcolor="rgb(242, 242, 242)",
            bgcolor="whitesmoke",
            radialaxis=dict(
                visible=False,
                range=[0, 1],
                )
            ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            #xanchor="center",
            x=0.25
            ),
        margin=dict(l=85, r=80, t=0, b=0)
        )

    return fig

def plot_profile_both(input_profile, output_profile):

    """
    Returns a Polar chart of the profiles of both selected and recommended methods.

    Parameters:
        input_profile (pandas df): profile of selected methods
        output_profile (pandas df): profile of recommended methods
    
    Returns:
        fig (Plotly figure): polar chart based on profile of selected and 
                                recommended methods.
    """

    # column names for the original dataframe
    DF_COLNAMES = ['Quantitative', 'Qualitative', 'Inductive', 'Deductive', 
                   'Spatial_Global', 'Spatial_System', 'Spatial_Individual', 
                   'Temporal_Past','Temporal_Present', 'Temporal_Future']

    fig = go.Figure()

    fig.add_trace(
    go.Scatterpolar(
        r=input_profile.score,
        theta=DF_COLNAMES,
        mode="lines",
        line=dict(smoothing=1.3, shape="spline", color="rgb(220, 5, 12)"),
        fillcolor="rgb(220, 5, 12)",
        fill='toself',
        opacity=0.5,
        name='Profile of Selected Methods',
        ),
    )
    
    # 2.2. Polar Plot: Output Profile
    fig.add_trace(
        go.Scatterpolar(
            r=output_profile.score,
            theta=DF_COLNAMES,
            mode="lines",
            line=dict(smoothing=1.3, shape="spline", color="rgb(54, 75, 154)"),
            fillcolor="rgb(54, 75, 154)",
            fill='toself',
            opacity=0.5,
            name='Profile of Recommended Methods'
            ),
        )
    
    # 3. Update Figure Layout
    fig.update_layout(
        polar=dict(
            #bgcolor="rgb(242, 242, 242)",
            bgcolor="whitesmoke",
            radialaxis=dict(
                visible=False,
                range=[0, 1],
                )
            ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0,
            x=0.25
            ),
        margin=dict(l=85, r=80, t=0, b=30)
        )

    return fig

def plot_table(selected_methods, rec_methods):

    """
    Returns a plotly table that lists the methods that are selected
    and the mtehods that have been recommended.

    Parameters:
        selected_methods (list): methods that have been selected
        rec_methods (list): methods that were recommended

    Returns:
        fig (Plotly figure): a Plotly table with selected and recommended methods.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Selected Methods", "Recommended Methods"],
                fill_color="lavender",
                font=dict(family="Gravitas One", size=16)
            ),
            cells=dict(
                values=[selected_methods, rec_methods],
                fill_color="snow",
                font=dict(family="Arial", size=14)
            ),
        ),
    )

    # 2. Update Figure Layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=25, b=20)
    )

    return fig

def train_model(data, n_neighbors=6, metric="cosine", save_file=None):
    """
    Train a kNN model based on the supplied data. Optionally, save
    the model into a pickled file.

    Parameters:
        data (str): the path of the data file used to train the model.
        n_neighbors (int, optional): No. of neighbors. Defaults to 6.
        metric (str, optional): The similarity metric . Defaults to "cosine".
        save_file (str, optional): file name for the pickle file (eg. model.pkl). Defaults to None.
    
    """
    # load the data
    df = pd.read_excel(data, index_col="Method").fillna(0)

    # instantiate and fit the model
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(df)

    # save the model
    if save_file:
        with open(save_file, "wb") as f:
            pickle.dump(knn, f)
