#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import xgboost as xgb
import numpy as np
import plotly.express as px
from pickle import load
from pipeline_functions import DataFrameFeatureUnion, SelectColumnsTransfomer
from pipeline_functions import DataFrameFunctionTransformer, backToDf
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
import pickle

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

np.random.seed(42)

# Loading data and pipeline:
application_test = pd.read_csv('../input/application_test.csv')
application_test = application_test.set_index('SK_ID_CURR')

pipeline = load(open('../preprocessing_features.pkl', 'rb'))

# Loading the model:
model = pickle.load(open('../regLog', 'rb'))

# Taking only the good columns - I'll see if I can put that on the pipeline later
col = ['CODE_GENDER', 'FLAG_OWN_CAR', 'AMT_CREDIT', 'AMT_ANNUITY',
       'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
       'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
       'EXT_SOURCE_3', 'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_YEAR',
       'TERM', 'OVER_EXPECT_CREDIT', 'AMT_REQ_CREDIT_BUREAU_TOTAL',
       'BIRTH_EMPLOTED_INTERVEL', 'BIRTH_REGISTRATION_INTERVEL',
       'SEASON_REMAINING']

# Features engineering - I'll see if I can put that on the pipeline later
application_test['TERM'] = application_test.AMT_CREDIT / application_test.AMT_ANNUITY
application_test['OVER_EXPECT_CREDIT'] = (application_test.AMT_CREDIT > application_test.AMT_GOODS_PRICE).map({False:0, True:1})
application_test['MEAN_BUILDING_SCORE_AVG'] = application_test.iloc[:, 44:58].mean(skipna=True, axis=1)
application_test['TOTAL_BUILDING_SCORE_AVG'] = application_test.iloc[:, 44:58].sum(skipna=True, axis=1)
application_test['FLAG_DOCUMENT_TOTAL'] = application_test.iloc[:, 96:116].sum(axis=1)
application_test['AMT_REQ_CREDIT_BUREAU_TOTAL'] = application_test.iloc[:, 116:122].sum(axis=1)
application_test['BIRTH_EMPLOTED_INTERVEL'] = application_test.DAYS_EMPLOYED - application_test.DAYS_BIRTH
application_test['BIRTH_REGISTRATION_INTERVEL'] = application_test.DAYS_REGISTRATION - application_test.DAYS_BIRTH
application_test['MEAN_BUILDING_SCORE_AVG'] = application_test.iloc[:, 44:58].mean(skipna=True, axis=1)
application_test['TOTAL_BUILDING_SCORE_AVG'] = application_test.iloc[:, 44:58].sum(skipna=True, axis=1)
application_test['INCOME_PER_FAMILY_MEMBER'] = application_test.AMT_INCOME_TOTAL / application_test.CNT_FAM_MEMBERS
application_test['SEASON_REMAINING'] = application_test.AMT_INCOME_TOTAL/4 -  application_test.AMT_ANNUITY
application_test['RATIO_INCOME_GOODS'] = application_test.AMT_INCOME_TOTAL -  application_test.AMT_GOODS_PRICE
application_test['CHILDREN_RATIO'] = application_test.CNT_CHILDREN / application_test.CNT_FAM_MEMBERS

X_test = application_test.copy()

# Transforming data from the pipeline
X_test = pipeline.transform(X_test)
X_test = X_test.loc[:, col]
X_test.index = application_test.index
application_test = application_test.loc[:, col]

# Making the prediction on the X_test:
predictions = model.predict_proba(X_test)

pred_df = pd.DataFrame(predictions[:,0], index=X_test.index, columns=['PREDICT'])
pred_df['PREDICT_BOOL'] = np.where(pred_df['PREDICT']>0.50, 'YES', 'NO')
df = pd.merge(application_test, pred_df, left_index=True, right_index=True, how='inner')
df_ = pd.merge(application_test, pred_df, left_index=True, right_index=True, how='inner')

# Saving to change in the graph
available_indicators = df.drop(columns=['PREDICT_BOOL', 'PREDICT']).columns

# Init the coef of the model
importance = model.coef_[0]
importance = pd.DataFrame(importance, index=X_test.columns, columns=['Importance'])

app.layout = html.Div([
    html.Div([

# Input user for x axis and customer id
        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='SEASON_REMAINING'
            )
        ],
        style={'width': '49%', 'display': 'inline-block'})
    ,
       html.Div([
            dcc.Dropdown(
                id='customer',
                options=[{'label': i, 'value': i} for i in df.index],
                value=100005
            )
        ],
        style={'width': '49%', 'display': 'inline-block'})
    ], 
        style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    
# Graph display
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
        ),
        dcc.Graph(
            id='allcust',
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'})
    ,
    html.Div([
        dcc.Graph(id='prob'),
        dcc.Graph(id='explain'),
    ], style={'display': 'inline-block', 'width': '49%', 'padding': '0 20'})
    ])


# Callback from input user
@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value')])

def update_graph(xaxis_column_name):
    dff = df.copy()
    
    # If columns is numeric we get a figure, if not we get nothing
    if (dff[xaxis_column_name].dtypes == int) or (dff[xaxis_column_name].dtypes == float):    
        df_mean = dff.drop(columns=['PREDICT'])

        df_mean = df_mean.groupby(['PREDICT_BOOL']).mean().reset_index()

        fig = px.histogram(df_mean, x=df_mean['PREDICT_BOOL'],
                           y=xaxis_column_name, hover_data=df_mean.columns,
                           color=df_mean['PREDICT_BOOL'],
                           title="Mean Value of the Feature by Answer")

        fig.update_xaxes(title='IS LOAN GRANTED')
        fig.update_layout(margin={'l': 40, 'b': 40, 't': 30, 'r': 0}, hovermode='closest')
        
    else:
        df_mean = dff.drop(columns=['PREDICT'])
        df_mean['COUNT'] = 1
        df_mean = df_mean.groupby(['PREDICT_BOOL', xaxis_column_name]).sum().reset_index()
        df_mean = df_mean.set_index(['PREDICT_BOOL', xaxis_column_name])['COUNT']

        fig = px.imshow(df_mean.unstack(), 
                        title="Heatmap")
        fig.update_layout(margin={'l': 40, 'b': 40, 't': 30, 'r': 0}, hovermode='closest')

    return fig

@app.callback(
    dash.dependencies.Output('allcust', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
    dash.dependencies.Input('customer', 'value')])

def all_graph(xaxis_column_name, customer):
    dff = df.copy()
    dff['is_cust'] = 0
    dff.loc[customer, 'is_cust'] = 1

    fig = px.histogram(dff, x=xaxis_column_name, hover_data=df.columns,
                       color=dff['is_cust'],
                       title="Descriptive Analysis by Customer")


    fig.update_xaxes(title=xaxis_column_name)

    fig.update_yaxes(title='COUNT', type='linear')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear')

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})

    return fig

# Callback from input user
@app.callback(
    dash.dependencies.Output('explain', 'figure'),
    [dash.dependencies.Input('customer', 'value')])
    
def create_explain(customer):
    
    dff = X_test.loc[customer,:]
    
    # Calculate the weight of eich feature:
    weights = (-importance['Importance']) * dff
    weights = pd.DataFrame(weights, columns=['Importance']).reset_index()

    # Plot the results
    fig = px.histogram(weights, x=weights['index'], y=weights['Importance'], hover_data=weights.columns,
                       title="Prediction Explanation")
    fig.update_yaxes(title='Values')

    return fig



@app.callback(
    dash.dependencies.Output('prob', 'figure'),
    [dash.dependencies.Input('customer', 'value')])

def update_prediction(customer):
    
    pred = pd.DataFrame(predictions, index=X_test.index, columns=['YES', 'NO'])
    dff_cust = pd.merge(X_test, pred, left_index=True, right_index=True, how='inner')
    answer = dff_cust.stack(level=0)
    answer = pd.DataFrame(answer, columns=['Proba'])
    answer = answer.reset_index(level=1).rename(columns={'level_1':'Answer'})
    answer = answer[(answer['Answer'] == 'YES') | (answer['Answer'] == 'NO')]
    answer = answer.loc[customer,:]
                                     
    fig = px.histogram(answer, x=answer['Answer'], y=answer['Proba'],
                       title="Is your Loan Granted")
    fig.update_yaxes(title='Probability')
    fig.update_layout(height=250, margin={'l': 20, 'b': 30, 'r': 10, 't': 40})
    
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
# http://127.0.0.1:8050/