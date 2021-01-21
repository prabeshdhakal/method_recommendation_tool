import pickle
import datetime
import numpy as np
import pandas as pd

# Plotly, Dash, and Dash Bootstrap Components
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mrt_utils import get_methods_profile, clean_methods_names
from mrt_utils import predict_methods, rec_plot_table
from mrt_utils import plot_table, plot_profile_selected, plot_profile_both

# 1. SET MODEL & DATA FILE PATH
data_path = "data/data_210114.xlsx"
model_path = "models/model_210114.pkl"

# 2. LOAD DATASET
raw_df = pd.read_excel(data_path)

# NAs are imputed with 0
df = raw_df.copy().set_index("Method").fillna(0)

# select list of names of methods
methods_names = raw_df.Method

# 3. READ TEXT FROM FILES
with open("text/what_now.md", "r") as f:
    what_now = f.read()

with open("text/tool_explanation.md", "r") as f:
    explanation = f.read()

# 4. DEFINE DROPDOWN OPTIONS

# 4.1 example of how dropdown_options looks like: 
# dropdown_options = [
#    {"label": "Action Research", "value":"action_research"},
#    {"label": "ANOVA", "value":"anova"},
#    {"label": "Bayesian Inference", "value":"bayesian_inference"}
#]

# 4.2 Generate cleaned method names
dropdown_options = []
for i in range(methods_names.shape[0]):
    method = methods_names[i]
    dropdown_options.append(
        dict(
            label=clean_methods_names([method])[0],
            value=method
        )
    )

# 4.3 Define the three dropdown options
dropdown_0 = dcc.Dropdown(
        options=dropdown_options,
        value="agent_based_model",
        multi=False,
        placeholder="Select 1st method",
        id="method_0"
    )
dropdown_1 = dcc.Dropdown(
        options=dropdown_options,
        value="anova",
        multi=False,
        placeholder="Select 2nd method",
        id="method_1"
    )
dropdown_2 = dcc.Dropdown(
        options=dropdown_options,
        value="bayesian_inference",
        multi=False,
        placeholder="Select 3rd method",
        id="method_2"
    )

# 4.4 create the HTML Div object for the dropdown tools

dropdown_menus = [dropdown_0, dropdown_1, dropdown_2]

dropdown_row = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Div(dropdown_0)),
                dbc.Col(html.Div(dropdown_1)),
                dbc.Col(html.Div(dropdown_2)),
            ]
        ),
    ]
)


# 5. DEFINE the TOP SECTION RECOMMENDATION 
# (the 3 plots below the 3 dropdown)
rec_plots_row = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Div(dcc.Graph(id="plot_0"))),
                dbc.Col(html.Div(dcc.Graph(id="plot_1"))),
                dbc.Col(html.Div(dcc.Graph(id="plot_2"))),
            ]
        ),
    ]
)


# 6. DEFINE the PLOTS AT THE BOTTOM SECTION
final_plots_row = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Div(dcc.Graph(id="plot_selected_methods_profile"))),
                #dbc.Col(html.Div(dcc.Graph(id="plot_table"))),
                dbc.Col(html.Div(dcc.Graph(id="plot_both_profiles"))),
            ]
        ),
    ]
)

final_table_bottom = dcc.Graph(id="plot_table")

# 7. THE MAIN APP

app = dash.Dash(external_stylesheets=[dbc.themes.JOURNAL])

app.layout = dbc.Container(
    children=[
        # 1. Page Heading
        html.H1(children="Method Recommendation Tool",style={"textAlign":"center"}),
        html.P("A project of the Faculty of Sustainability at Leuphana University", style={"textAlign":"center"}),
        html.Br(),

        # 1.1 Introduction to the tool
        html.H3("Welcome to the Method Recommendation Tool!", style={"textAlign":"center"}),
        html.Br(),
        html.P("Choose three scientific methods that you are familiar with. Then, the tool will recommend you five new methods each to explore.", style={"textAlign":"center"}),

        # 2. Method Selection Dropdown Tools
        html.H3(children="Select 3 Methods", style={"textAlign":"center"}),
        html.P("Choose 3 methods that you found interesting during the semester.", style={"textAlign":"center"}),
        dropdown_row, ### DROPDOWN
        #html.Br(),
        html.Br(),
        
        # 3. Plotly Plot (Recommended Methods List + Profiles)x3
        html.P("Here are the results based on your selection:", style={"textAlign":"center"}),
        rec_plots_row,
        html.Br(),
        html.Br(),

        # 4. What now section
        html.H3("What now?", style={"textAlign":"center"}),
        dcc.Markdown(what_now, style={"textAlign":"center"}),
        html.Br(),
        # 5. Further explanation section
        html.H3("How does the tool work, you ask?", style={"textAlign":"center"}),
        dcc.Markdown(explanation, style={"textAlign":"center"}),

        # 6. Final chart with Overall Profile and Profile Overlay + Recs.
        html.Br(),
        final_plots_row,

        # 7. Plot the final Selection Profiles and Recommendation Profiles
        html.Br(),
        html.P("Here are all the methods that you selected and that were recommended by our tool:", style={"textAlign":"center"}),
        final_table_bottom

    ]
)

# 8. CALLBACKs (look up dash documentation to learn more)
# short story short: callbacks enable dynamic updates of dropdowns and plotly charts

# 8.1 top left dropdown/table/plot
@app.callback(
    Output("plot_0", "figure"),
    [Input("method_0", "value")]
)
def update_rec_plot_0(method_0):
    
    selected_method = df.loc[[method_0]]
    selected_method_clean = clean_methods_names([method_0])[0]
    input_profile = get_methods_profile(selected_method)

    preds = predict_methods(selected_method, no_preds=1, data=data_path, saved_model=model_path)

    rec_methods = preds["recommended_methods"]    
    output_profile = preds["recommended_methods_profile"]
    
    return rec_plot_table(selected_method_clean, input_profile, output_profile, rec_methods)

# 8.2 top middle dropdown/table/plot
@app.callback(
    Output("plot_1", "figure"),
    [Input("method_1", "value")]
)
def update_rec_plot_1(method_1):
    
    selected_method = df.loc[[method_1]]
    selected_method_clean = clean_methods_names([method_1])[0]
    input_profile = get_methods_profile(selected_method)

    preds = predict_methods(selected_method, no_preds=1, data=data_path, saved_model=model_path)

    rec_methods = preds["recommended_methods"]    
    output_profile = preds["recommended_methods_profile"]
    
    return rec_plot_table(selected_method_clean, input_profile, output_profile, rec_methods)

# 8.3 top right dropdown/table/plot
@app.callback(
    Output("plot_2", "figure"),
    [Input("method_2", "value")]
)
def update_rec_plot_2(method_2):
    
    selected_method = df.loc[[method_2]]
    selected_method_clean = clean_methods_names([method_2])[0]
    input_profile = get_methods_profile(selected_method)

    preds = predict_methods(selected_method, no_preds=1, data=data_path, saved_model=model_path)

    rec_methods = preds["recommended_methods"]    
    output_profile = preds["recommended_methods_profile"]
    
    return rec_plot_table(selected_method_clean, input_profile, output_profile, rec_methods)

# 8.4 bottom left plot
@app.callback(
    Output("plot_selected_methods_profile", "figure"),
    [
        Input("method_0", "value"),
        Input("method_1", "value"),
        Input("method_2", "value"),
    ]
)
def update_selected_method_profile(method_0, method_1, method_2):
    input_methods = [method_0, method_1, method_2]

    selected_methods = df.loc[input_methods]
    
    input_profile = get_methods_profile(selected_methods)

    return plot_profile_selected(input_profile)

# 8.5 bottom middle table
@app.callback(
    Output("plot_table", "figure"),
    [
        Input("method_0", "value"),
        Input("method_1", "value"),
        Input("method_2", "value"),
    ]
)
def update_table(method_0, method_1, method_2):
    input_methods = [method_0, method_1, method_2]

    selected_methods = df.loc[input_methods]
    selected_methods_clean = clean_methods_names(input_methods)

    preds = predict_methods(selected_methods, no_preds=3, data=data_path, saved_model=model_path)

    rec_methods = preds["recommended_methods"]

    return plot_table(selected_methods_clean, rec_methods)

# 8.6 bottom right plot
@app.callback(
    Output("plot_both_profiles", "figure"),
    [
        Input("method_0", "value"),
        Input("method_1", "value"),
        Input("method_2", "value"),
    ]
)
def update_both_profiles(method_0, method_1, method_2):
    input_methods = [method_0, method_1, method_2]

    selected_methods = df.loc[input_methods]

    preds = predict_methods(selected_methods, no_preds=3, data=data_path, saved_model=model_path)

    input_profile = get_methods_profile(selected_methods)
    output_profile = preds["recommended_methods_profile"]

    return plot_profile_both(input_profile, output_profile)



if __name__ == "__main__":
    app.run_server(debug=True)