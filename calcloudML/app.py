import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
import dash_daq as daq


from makefigs import cmx, scoring, load_data, roc_auc, features, predictor, nodegraph

app = dash.Dash(__name__, title="CalcloudML", suppress_callback_exceptions=True)
# server = app.server

# LOAD MODEL DATA
timestamps = [1620740441, 1620929899, 1621096666]
versions = ["v0", "v1", "v2"]
meta = load_data.make_meta(timestamps, versions)
results = load_data.make_res(meta, versions)
# df_meta = load_data.import_csv(src='file', key='./data/training_metadata.csv')

# LOAD DATA AND FIGURES
df_scores = load_data.get_scores(results)
acc_fig, loss_fig = scoring.acc_loss_bars(df_scores)

# LOAD TRAINING DATASET (for scatterplot)
df = load_data.get_single_dataset('data/hst_data.csv')
instruments = list(df['instr_key'].unique())
feature_list = ['x_files', 'x_size', 'drizcorr', 'pctecorr', 'crsplit', 
        'subarray', 'detector', 'dtype', 'instr', 'n_files', 
        'total_mb', 'mem_bin', 'memory', 'wallclock']

acs = df.groupby('instr').get_group(0)
cos = df.groupby('instr').get_group(1)
stis = df.groupby('instr').get_group(2)
wfc3 = df.groupby('instr').get_group(3)

stylesheet=nodegraph.make_stylesheet()
styles=nodegraph.make_styles()
clf = predictor.get_model('./models/mem_clf')
edges, nodes = nodegraph.make_neural_graph()

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

layout_index = html.Div(children=[
    html.Br(),
    html.H1('CALCLOUD', 
        style={'padding': 15}),
    html.H2('Machine Learning Dashboard'),
    html.Div(children=[
        html.Div('Model Performance + Statistical Analysis'), 
        html.Div('for the Hubble Space Telescope\'s'),
        html.Div('data reprocessing pipeline.')], style={'display': 'inline-block'}),
        html.Div([
            html.Br(),
            dcc.Link('Model Performance Evaluation', href='/page-1'),
            html.Br(),
            dcc.Link('Exploratory Data Analysis', href='/page-2'),
            html.Br(),
            dcc.Link('Neural Network Graph', href='/page-3')
            ]
            )],
         style={
        'backgroundColor':'#1b1f34', 
        'color':'white',
        'textAlign': 'center',
        'width': '80%',
        'display': 'inline-block',
        'float': 'center',
        'padding': '10%',
        })


layout_page_1 = html.Div(children=[
    html.Div(children=[
        html.Br(),
        dcc.Link('Home', href='/'),
        html.Br(),
        dcc.Link('Exploratory Data Analysis', href='/page-2'),
        html.Br(),
        dcc.Link('Neural Network Graph', href='/page-3'),
        html.Br()
        ]),
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
        html.P('ROC AUC', style={'margin': 25}),
        html.P('Receiver Operator Characteristic', style={'margin': 25}),
        html.P('(Area Under the Curve)', style={'margin': 25}),
        html.Div(children=[
            html.Div(children=[
                html.Div([
                    dcc.Dropdown(
                        id='rocauc-picker',
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
                    id='roc-auc',
                    style={
                        'display': 'inline-block',
                        'float': 'center',
                        'padding': 25
                        }),
                dcc.Graph(
                    id='precision-recall-fig',
                    style={
                        'display': 'inline-block',
                        'float': 'center',
                        'padding': 25
                        })
            ]
        ),
        # CONFUSION MATRIX
        html.Div(children=[
            html.Div([
                dcc.Dropdown(
                    id='cmx-type',
                    options=[
                        {'label': 'counts', 'value': 'counts'},
                        {'label': 'normalized', 'value': 'normalized'}
                        ],
                    value="normalized"
                        )], style={
                            'color': 'black',
                            'display': 'inline-block',
                            'float': 'center',
                            'width': 150,
                            }),
                dcc.Graph(
                    id='confusion-matrix',
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
    ])],
    style={
        'backgroundColor':'#1b1f34', 
        'color':'white',
        'textAlign': 'center',
        'width': '100%',
        'display': 'inline-block',
        'float': 'center'
        })


layout_page_2 = html.Div(children=[
    html.Div(children=[
        html.Br(),
        dcc.Link('Home', href='/'),
        html.Br(),
        dcc.Link('Model Performance Evaluation', href='/page-1'),
        html.Br(),
        dcc.Link('Neural Network Graph', href='/page-3'),
        html.Br()
        ]),
        html.Div(children=[
        # FEATURE COMPARISON SCATTERPLOTS
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
                        })
            ],
            style={
                'color':'white',
                'width': '100%'}
                ),
        # BOX PLOTS: CONTINUOUS VARS (N_FILES + TOTAL_MB)
        html.Div(children=[
            html.Div(children=[
                html.Div(children=[
                html.Div([
                    dcc.Dropdown(
                        id='continuous-vars',
                        options=[{'label': 'Raw Data', 'value': 'raw'},
                        {'label': 'Normalized', 'value': 'norm'}],
                        value='raw'
                    )], 
                    style={
                        'width': '40%', 
                        'display': 'inline-block', 
                        'padding': 5, 
                        'float': 'center',
                        'color': 'black'})
                    ]),
                dcc.Graph(
                    id='n_files',
                    style={
                        'display': 'inline-block',
                        'float': 'center',
                        'width': '50%'
                        }),
                dcc.Graph(
                    id='total_mb',
                    style={
                        'display': 'inline-block',
                        'float': 'center',
                        'width': '50%'
                        })
            ], style={
                'color':'white',
                'width': '100%'})
        ], 
        style={
            'backgroundColor': '#242a44', 
            'color':'white', 
            'padding':15, 
            'display': 'inline-block', 
            'width': '85%'
            })

        ], style={
            'backgroundColor': '#242a44', 
            'color':'white', 
            'padding':20, 
            'display': 'inline-block', 
            'width': '100%',
            'textAlign': 'center'
            })



layout_page_3 = html.Div(children=[
# nav
html.Div(children=[
    html.Br(),
    dcc.Link('Home', href='/'),
    html.Br(),
    dcc.Link('Model Performance Evaluation', href='/page-1'),
    html.Br(),
    dcc.Link('Exploratory Data Analysis', href='/page-2'),
    html.Br()
    ]),

    # GRAPH
    html.Div(children=[
    cyto.Cytoscape(
        id='cytoscape-compound',
        layout={'name': 'preset'},
        style={'width': '99vw', 'height': '70vh', 'display': 'inline-block', 'float': 'center', 'background-color': '#1b1f34'},
        stylesheet=stylesheet,
        elements=edges+nodes
    ),
    html.Div(children=[
        html.P(id='cytoscape-tapNodeData-output', style=styles['pre']),
        html.P(id='cytoscape-tapEdgeData-output', style=styles['pre']),
        html.P(id='cytoscape-mouseoverNodeData-output', style=styles['pre']),
        html.P(id='cytoscape-mouseoverEdgeData-output', style=styles['pre']),
    ], style={'width': '100%', 'margin': 0, 'padding': 0, 'display': 'inline-block', 'float': 'left', 'background-color': '#1b1f34'})
    ]),
    # CONTROLS 
    html.Div(children=[
    # INPUT DROPDOWNS
    html.Div(id='Xi', children=[
        # INPUTS LEFT COL
        html.Div(children=[
            html.Label([
                html.Label("INSTR", style={'padding': 5, 'display': 'inline-block', 'float': 'left'}),
                    dcc.Dropdown(
                        id='instr-state',
                        options=[{'label': i, 'value': i} for i in ['ACS', 'COS', 'STIS', 'WFC3']],
                        value='ACS',
                        style={'color': 'black', 'width': 135, 'display': 'inline-block', 'float': 'right'}
                    )
                    ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 255}),
            html.Label([
                html.Label("DTYPE", style={'padding': 5, 'display': 'inline-block', 'float': 'left'}),
                    dcc.Dropdown(
                        id='dtype-state',
                        options=[{'label': i, 'value': i} for i in ['SINGLETON', 'ASSOCIATION']],
                        value='SINGLETON',
                        style={'color': 'black', 'width': 135, 'display': 'inline-block', 'float': 'right'}
                    ),
                    ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 255}),
            html.Label([
                html.Label("DETECTOR", style={'padding': 5, 'display': 'inline-block', 'float': 'left'}),
                    dcc.Dropdown(
                        id='detector-state',
                        options=[{'label': i, 'value': i} for i in ['UVIS', 'WFC', 'IR', 'HRC', 'SBC']],
                        value='UVIS',
                        style={'color': 'black', 'width': 135, 'display': 'inline-block', 'float': 'right'}
                    ),
                    ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 255}),
            html.Label([
                html.Label("SUBARRAY", style={'padding': 5, 'display': 'inline-block', 'float': 'left'}),
                    dcc.Dropdown(
                        id='subarray-state',
                        options=[{'label': i, 'value': i} for i in ['TRUE', 'FALSE']],
                        value='FALSE',
                        style={'color': 'black', 'width': 135, 'display': 'inline-block', 'float': 'right'}
                    ),
                    ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 255}),
            html.Label([
                html.Label("PCTECORR", style={'padding': 5, 'display': 'inline-block', 'float': 'left'}),
                    dcc.Dropdown(
                        id='pctecorr-state',
                        options=[{'label': i, 'value': i} for i in ['OMIT', 'PERFORM']],
                        value='PERFORM',
                        style={'color': 'black', 'width': 135, 'display': 'inline-block', 'float': 'right'}
                    ),
                    ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 255}),
            # END outputs Left Col
        ], style={'display': 'inline-block', 'float': 'left', 'width': 270, 'padding': 5}),
        # INPUTS RIGHT COL
        html.Div(children=[
            html.Label([
                html.Label("DRIZCORR", style={'padding': 5, 'display': 'inline-block', 'float': 'left'}),
                    dcc.Dropdown(
                        id='drizcorr-state',
                        options=[{'label': i, 'value': i} for i in ['OMIT', 'PERFORM']],
                        value='PERFORM',
                        style={'color': 'black', 'width': 135, 'display': 'inline-block', 'float': 'right'}
                    ),
                    ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 255}),
            html.Label([
                html.Label("CRSPLIT", style={'padding': 5, 'display': 'inline-block', 'float': 'left'}),
                    daq.NumericInput(
                        id='crsplit-state',
                        value=2,
                        min=0,
                        max=2,
                        style={'color': 'black', 'width': 135, 'display': 'inline-block', 'float': 'right'}
                    ),
                    ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 255}),
            html.Label([
                html.Label("TOTAL_MB", style={'padding': 5, 'display': 'inline-block', 'float': 'left'}),
                    daq.NumericInput(
                        id='totalmb-state',
                        value=4,
                        min=0,
                        max=900,
                        style={'color': 'black', 'width': 135, 'display': 'inline-block', 'float': 'right'}
                    ),
                    ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 255}),
            html.Label([
                html.Label("N_FILES", style={'padding': 5, 'display': 'inline-block', 'float': 'left'}),
                    daq.NumericInput(
                        id='nfiles-state',
                        value=2,
                        min=1,
                        max=200,
                        style={'color': 'black', 'width': 135, 'display': 'inline-block', 'float': 'right'}
                    )
                    ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 255}),
            
    # TODO SUBMIT BUTTON
            html.Button('Submit', id='submit-button-state', n_clicks=0, style={'margin': 10})

            # END Input Right COL
            ], style={'display': 'inline-block', 'float': 'left', 'width': 270, 'margin': 10, 'padding': 5}
        # END INPUTS (BOTH COLS)
        )], style={'width': 620, 'display': 'inline-block', 'float': 'left', 'padding': 5, 'background-color': '#242a44'}),
    # OUTPUTS
    html.Div(children=[
        # MEMORY PRED VS ACTUAL
        html.Div(children=[
            # Memory Bin Pred vs Actual LED Display Values
            daq.LEDDisplay(
                id='prediction-bin-output',
                label="Y PRED",
                labelPosition='bottom',
                #value='1',
                color="cyan",
                backgroundColor='#242a44',
                style={'padding': 5, 'width': 75, 'display': 'inline-block', 'float': 'left'}
                ),
            # daq.LEDDisplay(
            #     id='memory-bin-output',
            #     label="Y TRUE",
            #     labelPosition='bottom',
            #     value='2',
            #     color='lightgreen',
            #     backgroundColor='#242a44',
            #     style={'padding': 5, 'width': 75, 'display': 'inline-block', 'float': 'left'}
            #     )
            ], style={'width': 100, 'margin': 5, 'padding': 5, 'display': 'inline-block', 'float': 'left', 'background-color': '#242a44'}),
            # Probabilities
        html.Div(children=[
            daq.GraduatedBar(
                id='p0',
                label='P(0)',
                labelPosition='right',
                step=0.1,
                min=0,
                max=1,
                #value=0.42,
                showCurrentValue=True,
                vertical=False,
                color='cyan',
                style={'color': 'black', 'display': 'inline-block', 'float': 'left', 'padding': 5}
            ),
            daq.GraduatedBar(
                id='p1',
                label='P(1)',
                labelPosition='right',
                step=0.1,
                min=0,
                max=1,
                #value=0.99,
                showCurrentValue=True,
                vertical=False,
                color='cyan',
                style={'color': 'black', 'display': 'inline-block', 'float': 'left', 'padding': 5}
            ),
            daq.GraduatedBar(
                id='p2',
                label='P(2)',
                labelPosition='right',
                step=0.1,
                min=0,
                max=1,
                #value=0.23,
                showCurrentValue=True,
                vertical=False,
                color='cyan',
                style={'color': 'black', 'display': 'inline-block', 'float': 'left', 'padding': 5}
            ),
            daq.GraduatedBar(
                id='p3',
                label='P(3)',
                labelPosition='right',
                step=0.1,
                min=0,
                max=1,
                #value=0.09,
                showCurrentValue=True,
                vertical=False,
                color='cyan',
                style={'color': 'black', 'display': 'inline-block', 'float': 'left', 'padding': 5}
            )
            # END Probabilities
            ], style={'width': 360, 'padding': 10, 'display': 'inline-block', 'float': 'left', 'color': 'white', 'background-color': '#242a44'}),
        # Memory GAUGE Predicted vs Actual
        html.Div(children=[
            daq.Gauge(
                id='memory-gauge-predicted',
                color={"gradient":True,"ranges":{"yellow":[0,2],"orange":[2,8],"red":[8,16],"blue":[16,64]}},
                #value=4.2,
                label='Predicted',
                labelPosition='bottom',
                units='GB',
                showCurrentValue=True,
                max=64,
                min=0,
                size=150,
                style={'color': 'white', 'display': 'inline-block', 'float': 'left'}
                )
            # daq.Gauge(
            #     id='memory-gauge-actual',
            #     color={"gradient":True,"ranges":{"yellow":[0,2],"orange":[2,8],"red":[8,16],"blue":[16,64]}},
            #     value=16,
            #     label='Actual',
            #     labelPosition='bottom',
            #     units='GB',
            #     showCurrentValue=True,
            #     max=64,
            #     min=0,
            #     size=150,
            #     style={'color': 'white', 'display': 'inline-block', 'float': 'left'}
            #     )
            ],
            style={'width': 375, 'display': 'inline-block', 'float': 'left', 'color': 'white', 'background-color': '#242a44'}),
            ])
    # END Controls and Outputs
    ], style={'width': '100%',  'display': 'inline-block', 'float': 'left', 'background-color': '#242a44'}),
    # SUMMARY TEXT
    # html.Div(children=[
    #     html.Div(children=[
    #         html.P("Predict Resource Allocation requirements for memory (GB) and max execution `kill time` or `wallclock` (seconds) using three pre-trained neural networks."),
    #         html.Br(),
    #         html.P("MEMORY BIN: classifier outputs probabilities for each of the four bins (`target classes`). The class with the highest probability score is considered the final predicted outcome (y). This prediction variable represents which of the 4 possible memory bins is most likely to meet the minimum required needs for processing an HST dataset (ipppssoot) successfully according to the given inputs (x)."),
    #         html.Div(children=[
    #             html.P("Memory Bin Sizes - target class `y`:"),
    #             html.Li("0: < 2GB"),
    #             html.Li("1: 2-8GB"),
    #             html.Li("2: 8-16GB"),
    #             html.Li("3: >16GB"),
    #         ]),
    #         html.Br(),
    #         html.P("WALLCLOCK REGRESSION: regression generates estimate for specific number of seconds needed to process the dataset using the same input data. This number is then tripled in Calcloud for the sake of creating an extra buffer of overhead in order to prevent larger jobs from being killed unnecessarily."),
    #         html.Br(),
    #         html.P("MEMORY REGRESSION: A third regression model is used to estimate the actual value of memory needed for the job. This is mainly for the purpose of logging/future analysis and is not currently being used for allocating memory in calcloud jobs."),    
    #         ],
    #         style={
    #             'width': '50%',
    #             'display': 'inline-block',
    #             'float': 'center',
    #             'padding': 15
    #         })
    #     ], style={
    #     'backgroundColor':'#1b1f34', 
    #     'color':'white',
    #     'textAlign': 'center',
    #     'width': '100%',
    #     'display': 'inline-block',
    #     'float': 'center'
    #     }), # END SUMMARY TEXT
    
# PAGE LAYOUT
], style={'width': '100%', 'height': '100%','background-color': '#242a44', 'color':'white'})





# index layout
app.layout = url_bar_and_content_div

# "complete" layout
app.validation_layout = html.Div([
    url_bar_and_content_div,
    layout_index,
    layout_page_1,
    layout_page_2,
    layout_page_3
])

# Index callbacks
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == "/page-1":
        return layout_page_1
    elif pathname == "/page-2":
        return layout_page_2
    elif pathname == "/page-3":
        return layout_page_3
    else:
        return layout_index


# Page 1 callbacks
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


# ROC AUC CALLBACK
@app.callback(
    [Output('roc-auc', 'figure'),
    Output('precision-recall-fig', 'figure')],
    Input('rocauc-picker', 'value')
)

def update_roc_auc(selected_version):
    y = results[selected_version]['mem_bin']['y_true']
    y_scores = results[selected_version]['mem_bin']['proba']
    roc_figs = roc_auc.make_curves(y, y_scores)
    return roc_figs


@app.callback(
    Output('confusion-matrix', 'figure'),
    Input('cmx-type', 'value')
)

def update_cmx(cmx_type):
    cmx_fig = cmx.make_cmx_figure(results, cmx_type)
    return cmx_fig


# Page 2 callbacks
# SCATTER CALLBACK
@app.callback(
    [Output('acs-scatter', 'figure'),
    Output('wfc3-scatter', 'figure'),
    Output('cos-scatter', 'figure'),
    Output('stis-scatter', 'figure')],
    [Input('xaxis-features', 'value'),
    Input('yaxis-features', 'value')]
)


def update_scatter(xaxis_name, yaxis_name):
    instr_dict = features.df_by_instr(df)
    scatter_figs = features.make_scatter_figs(instr_dict, xaxis_name, yaxis_name)
    return scatter_figs


@app.callback(
    [Output('n_files', 'figure'),
    Output('total_mb', 'figure')],
    Input('continuous-vars', 'value')
)

def update_continuous(raw_norm):
    if raw_norm == 'raw':
        vars = ['n_files', 'total_mb']
    elif raw_norm == 'norm':
        vars = ['x_files', 'x_size']
    continuous_figs = features.make_continuous_figs(acs, cos, stis, wfc3, vars)
    return continuous_figs
# @app.callback(Output('page-2-display-value', 'children'),
#               Input('page-2-dropdown', 'value'))
# def display_value(value):
#     print('display_value')
#     return 'You have selected "{}"'.format(value)


# PAGE 3 CALLBACKS
@app.callback([
    Output('prediction-bin-output', 'value'),
    Output('memory-gauge-predicted', 'value'),
    Output('p0', 'value'),
    Output('p1', 'value'),
    Output('p2', 'value'),
    Output('p3', 'value')],
    Input('submit-button-state', 'n_clicks'),
    State('nfiles-state', 'value'),
    State('totalmb-state', 'value'),
    State('drizcorr-state', 'value'),
    State('pctecorr-state', 'value'),
    State('crsplit-state', 'value'),
    State('subarray-state', 'value'),
    State('detector-state', 'value'),
    State('dtype-state', 'value'),
    State('instr-state', 'value')
    )
def update_output(n_clicks, n_files, total_mb, drizcorr, pctecorr, crsplit, subarray, detector, dtype, instr):
    if n_clicks > 0:
        x_features = predictor.read_inputs(n_files, total_mb, drizcorr, pctecorr, crsplit, subarray, detector, dtype, instr)
        m_preds = predictor.make_preds(x_features) # [membin, memval, p0, p1, p2, p3]
        n_clicks = 0

        return m_preds
        # membin = output_preds['predictions']['memBin']
        # memval = output_preds['predictions']['memVal']
        # proba = output_preds['probabilities'][0]
        # print(proba)
        # p0, p1, p2, p3 = proba[0], proba[1], proba[2], proba[3]
        
        # n_clicks=0



@app.callback(
    Output('cytoscape-tapNodeData-output', 'children'),
    Input('cytoscape-compound', 'tapNodeData'))
def displayTapNodeData(data):
    if data:
        node = data['id']
        if node[0] not in ['x', 'i']:
            b = nodegraph.node_bias_clicks(node)
        else:
            b = None
        return f"bias: {node} = {str(b)}"



@app.callback(Output('cytoscape-tapEdgeData-output', 'children'),
                Input('cytoscape-compound', 'tapEdgeData'))
def displayTapEdgeData(data):
    if data:
        src = data['source'] # x1
        trg = data['target'] # h1-1
        w = nodegraph.edge_weight_clicks(src, trg)
        return f"weight: {src} and {trg} = {str(w)}"


@app.callback(Output('cytoscape-mouseoverNodeData-output', 'children'),
                Input('cytoscape-compound', 'mouseoverNodeData'))
def displayTapNodeData(data):
    if data:
        node = data['id']
        if node[0] not in ['x', 'i']:
            b = nodegraph.node_bias_clicks(node)
        else:
            b = None
        return f"bias: {node} = {str(b)}"


@app.callback(Output('cytoscape-mouseoverEdgeData-output', 'children'),
                Input('cytoscape-compound', 'mouseoverEdgeData'))
def displayTapEdgeData(data):
    if data:
        src = data['source'] # x1
        trg = data['target'] # h1-1
        w = nodegraph.edge_weight_clicks(src, trg)
        return f"weight: {src} and {trg} = {str(w)}"


if __name__ == '__main__':
    app.run_server(debug=True)