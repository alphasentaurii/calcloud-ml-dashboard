import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import cmx, scoring, load_data, features


app = dash.Dash(__name__, title="CalcloudML")

# LOAD MODEL DATA
timestamps = [1620351000, 1620740441, 1620929899, 1621096666]
versions = ["v0", "v1", "v2", "v3"]
meta = load_data.make_meta(timestamps, versions)
results = load_data.make_res(meta, versions)
df_meta = load_data.import_csv(src='file', key='./data/training_metadata.csv')

# LOAD DATA AND FIGURES
df_scores = load_data.get_scores(results)
acc_fig, loss_fig = scoring.acc_loss_bars(df_scores)
# history = load_data.load_res_file(meta, "history", "v0", "mem_bin")
# keras_acc, keras_loss = scoring.keras_plots(history)
y_true = results['v0']['mem_bin']['y_true']
y_pred = results['v0']['mem_bin']['y_pred']

cmx_fig = cmx.make_cmx_figure()

# LOAD TRAINING DATASET (for scatterplot)
df, instruments = load_data.get_single_dataset(meta, "v0")
feature_list = ['x_files', 'x_size', 'drizcorr', 'pctecorr', 'crsplit', 
        'subarray', 'detector', 'dtype', 'instr', 'n_files', 
        'total_mb', 'mem_bin', 'memory', 'wallclock']
#training_data, instruments = load_data.get_training_data(meta)

app.layout = html.Div(children=[
    html.H1('CALCLOUD Machine Learning', style={'padding': 5}),
    html.H2('Analytics Dashboard', style={'padding': 5}),
    html.Div(children=[
        html.Div('Model Performance + Statistical Analysis'), 
        html.Div('for the Hubble Space Telescope\'s data reprocessing pipeline.')], style={'display': 'inline-block'}),
    html.Div(children=[
        html.H3('Model Performance'),

        # MEMORY CLASSIFIER CHARTS
        html.Div(children=[
            html.H4(children='Memory Bin Classifier', style={'padding': 10}),
            
            # ACCURACY vs LOSS (BARPLOTS)
            'Accuracy vs Loss',
            html.Div(children=[
                dcc.Graph(
                    id='acc-bars',
                    figure=acc_fig,
                    style={
                        'display': 'inline-block',
                        'float': 'center',
                        'padding': 25}
                        ),
                dcc.Graph(
                    id='loss-bars',
                    figure=loss_fig,
                    style={
                        'display': 'inline-block',
                        'float': 'center',
                        'padding': 25
                        }
                    )
                ]),
            
            # KERAS HISTORY
            html.P('Keras History', style={'margin': 25}),
            html.Div(children=[
                html.Div(children=[
                    html.Div([
                    dcc.Dropdown(
                        id='version-picker',
                        options=[{'label':str(v),'value':v} for v in versions],
                        value="v0"
                        )], style={
                            'color': 'black',
                            'display': 'inline-block',
                            'float': 'center',
                            'width': 150,
                            })
                    ]),
                dcc.Graph(
                    id='keras-acc',
                    style={
                        'display': 'inline-block',
                        'float': 'center',
                        'padding': 25}
                    ),
                dcc.Graph(
                    id='keras-loss',
                    style={
                        'display': 'inline-block',
                        'float': 'center',
                        'padding': 25}
                    )
                ]
            ),
        # CONFUSION MATRIX
        html.Div(children=[
            dcc.Graph(
                id='confusion-matrix',
                figure=cmx_fig
                )], 
                style={
                    'color':'white',
                    'padding': 50,
                    'display': 'inline-block',
                    'width': '80%'
                    }
                )
            ], 
        style={
            'color':'white',
            'border':'2px #333 solid', 
            'borderRadius':5,
            'margin': 25,
            'padding': 10
            }),
    ]),
    html.Div(children=[
        html.H2(children='Exploratory Data Analysis'),
        html.Div([
            # FEATURE COMPARISON SCATTERPLOTS
            html.Div(children=[
                html.Div(children=[
                html.Div([
                    dcc.Dropdown(
                        id='xaxis-features',
                        options=[{'label': f, 'value': f} for f in feature_list],
                        value='n_files'
                    )], style={'width': '20%', 'display': 'inline-block', 'padding': 5}),
                html.Div([
                    dcc.Dropdown(
                        id='yaxis-features',
                        options=[{'label': f, 'value': f} for f in feature_list],
                        value='memory'
                    )], style={'width': '20%', 'display': 'inline-block', 'padding': 5})]),
                dcc.Graph(
                    id='acs-scatter',
                    style={
                        'display': 'inline-block',
                        'float': 'center'
                        }),
                dcc.Graph(
                    id='wfc3-scatter',
                    style={
                        'display': 'inline-block',
                        'float': 'center'
                        }),
                dcc.Graph(
                    id='cos-scatter',
                    style={
                        'display': 'inline-block',
                        'float': 'center'}),
                dcc.Graph(
                    id='stis-scatter',
                    style={
                        'display': 'inline-block',
                        'float': 'center'
                        }),
            ],
            style={
                'color':'white',
                'width': '100%'}
                )
            ],
        style={
            'backgroundColor': '#242a44', 
            'color':'white', 
            'padding':20, 
            'display': 'inline-block', 
            'width': '85%'
            }
        )
        ]),
    html.Div(children=[
        html.H2(children='Prediction Testing'),
        html.Div(children=[
            html.Div(children=[
            html.P("Predict Resource Allocation requirements for memory (GB) and max execution `kill time` or `wallclock` (seconds) using three pre-trained neural networks."),
            html.P("MEMORY BIN: classifier outputs probabilities for each of the four bins (`target classes`). The class with the highest probability score is considered the final predicted outcome (y). This prediction variable represents which of the 4 possible memory bins is most likely to meet the minimum required needs for processing an HST dataset (ipppssoot) successfully according to the given inputs (x)."),
            html.Div(children=[
                "Memory Bin Sizes - target class `y`: 0: < 2GB; 1: 2-8GB; 2: 8-16GB; 3: >16GB"
            ]),
            html.P("WALLCLOCK REGRESSION: regression generates estimate for specific number of seconds needed to process the dataset using the same input data. This number is then tripled in Calcloud for the sake of creating an extra buffer of overhead in order to prevent larger jobs from being killed unnecessarily."),
            html.P("MEMORY REGRESSION: A third regression model is used to estimate the actual value of memory needed for the job. This is mainly for the purpose of logging/future analysis and is not currently being used for allocating memory in calcloud jobs."),    
        ])
    ]),

        html.Div([
            html.Div(children=[
                html.Div(children=[
                    html.Div('inputs (x)'),
                    dcc.Dropdown(
                        id='instr',
                        options=[{'label': i, 'value': i} for i in instruments],
                        value='acs',
                        style={'color': 'black', 'display': 'inline-block', 'float': 'left', 'width': 150,
                        'margin': 5, 'padding': 5}
                    ),
                    dcc.Dropdown(
                        id='detector',
                        options=[{'label': 'UVIS', 'value': 'UVIS'},
                                {'label': 'IR', 'value': 'IR'}],
                        value='UVIS',
                        style={'color': 'black', 'display': 'inline-block', 'float': 'left', 'width': 150, 'margin': 5, 'padding': 5} 
                    ),
                    dcc.Dropdown(
                        id='drizcorr',
                        options=[{'label': 'perform', 'value': 'perform'},
                                {'label': 'omit', 'value': 'omit'}],
                        value='perform',
                        style={'color': 'black', 'display': 'inline-block', 'float': 'left', 'width': 150, 'margin': 5, 'padding': 5}
                    ),
                    dcc.Dropdown(
                        id='pctecorr',
                        options=[{'label': 'perform', 'value': 'perform'},
                                {'label': 'omit', 'value': 'omit'}],
                        value='perform',
                        style={'color': 'black', 'display': 'inline-block', 'float': 'left', 'width': 150, 'margin': 5, 'padding': 5}
                    )

                ], style={'width': '20%', 'border':'2px #fff solid', 'display': 'inline-block', 'float': 'left', 'padding': 5}),

                html.Div(children=[
                    html.Div('hidden layers'),
                    html.Div(children=[
                        html.Div('18'),
                        html.Div('32'),
                        html.Div('64'),
                        html.Div('32'),
                        html.Div('18'),
                        html.Div('9'),
                    ])

                ], style={'width': '40%', 'border':'2px #fff solid', 'display': 'inline-block', 'float': 'left', 'padding': 5}),

                html.Div(children=[
                    html.Div('outputs (y)'),
                    html.Div(children=[
                        html.Div('BIN'),
                        html.Div('GB'),
                        html.Div('SEC'),        
                    # dcc.Graph(
                    #     id='pred-bin'
                    # ),
                    # dcc.Graph(
                    #     id='pred-mem'
                    # ),
                    # dcc.Graph(
                    #     id='pred-wall'
                    # )
                ], style={'padding': 5})
                ], style={'width': '30%', 'border':'2px #fff solid', 'display': 'inline-block', 'float':'left', 'padding': 5})              
            ])
        ])
    ], style={'margin': 15})
    
    ],
    style={
        'backgroundColor':'#1b1f34', 
        'color':'white',
        'textAlign': 'center',
        'width': '100%',
        'display': 'inline-block',
        'float': 'center'
        })

# KERAS CALLBACK
@app.callback(
    [Output('keras-acc', 'figure'),
    Output('keras-loss', 'figure')],
    Input('version-picker', 'value')
)

def update_keras(selected_version):
    history = results[selected_version]['mem_bin']['history']
    keras_figs = scoring.keras_plots(history)
    return keras_figs

# SCATTER CALLBACK
@app.callback(
[Output('acs-scatter', 'figure'),
Output('wfc3-scatter', 'figure'),
Output('cos-scatter', 'figure'),
Output('stis-scatter', 'figure')],
[Input('xaxis-features', 'value'),
Input('yaxis-features', 'value')])


def update_scatter(xaxis_name, yaxis_name):
    instr_dict = features.df_by_instr(df)
    scatter_figs = []
    for instr, (data, color) in instr_dict.items():
        trace = go.Scatter(
                x=data[xaxis_name],
                y=data[yaxis_name],
                text=data['ipst'],
                mode='markers',
                opacity=0.7,
                marker={'size': 15, 'color': color},
                name=instr
            )
        layout = go.Layout(
            xaxis={'title': xaxis_name},
            yaxis={'title': yaxis_name},
            title=instr,
            #margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest',
            paper_bgcolor='#242a44',
            plot_bgcolor='#242a44',
            font={'color': '#ffffff'},
        )
        fig=go.Figure(data=trace, layout=layout)
        scatter_figs.append(fig)
    return scatter_figs


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=True)
