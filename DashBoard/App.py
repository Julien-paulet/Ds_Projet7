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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Loading data and pipeline:
application_test = pd.read_csv('../input/application_test.csv')
application_test = application_test.set_index('SK_ID_CURR')

pipeline = load(open('../preprocessing_features.pkl', 'rb'))

# Loading the model:
model = xgb.Booster({'nthread': 4})  # init model
model.load_model('../0001.model')  # load data

# Taking only the good columns - I'll see if I can put that on the pipeline later
col = [False,  True,  True, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
        True,  True,  True,  True,  True, False,  True,  True, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False,  True,  True,
        True, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True,  True,  True, False, False, False,  True,
        True,  True, False,  True, False, False]

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
X_test = X_test.iloc[:, col]
X_test.index = application_test.index
application_test = application_test.iloc[:, col]

# Transforming data into DMatrix - Not in pipeline yet as I don't know if XGBoost will be the final model:
dtest = xgb.DMatrix(X_test)

# Making the prediction on the X_test:
predictions = model.predict(dtest)

pred_df = pd.DataFrame(predictions, index=X_test.index, columns=['PREDICT'])
pred_df['PREDICT'] = np.where(pred_df['PREDICT']>0.50, 'NO', 'YES')
df = pd.merge(application_test, pred_df, left_index=True, right_index=True, how='inner')

# Saving to change in the graph
available_indicators = X_test.columns


### This was from the example code ###
#df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')
#available_indicators = df['Indicator Name'].unique()

#### End of example code ####

app.layout = html.Div([
    html.Div([

# Input user for x and y axis
        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Fertility rate, total (births per woman)'
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Life expectancy at birth, total (years)'
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    
# Graph display
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'})])


# Callback from input user
@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value')])

def update_graph(xaxis_column_name):
    
    dff = df.copy()

    fig = px.histogram(dff, x=xaxis_column_name, hover_data=df.columns)


    fig.update_xaxes(title=xaxis_column_name, type='linear')

    fig.update_yaxes(title='COUNT', type='linear')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear')

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


def create_time_series(dff, title):

    fig = px.scatter(dff, x='Year', y='Value')

    fig.update_traces(mode='lines+markers')

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)', text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value')])

def update_y_timeseries(hoverData, xaxis_column_name):
    country_name = hoverData['points'][0]['customdata']
    dff = df[df['Country Name'] == country_name]
    dff = dff[dff['Indicator Name'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
    return create_time_series(dff, title)


@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value')])

def update_x_timeseries(hoverData, yaxis_column_name):
    dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
    dff = dff[dff['Indicator Name'] == yaxis_column_name]
    return create_time_series(dff, yaxis_column_name)


if __name__ == '__main__':
    app.run_server(debug=True)
# http://127.0.0.1:8050/