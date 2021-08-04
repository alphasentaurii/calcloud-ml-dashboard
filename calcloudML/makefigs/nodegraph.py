import dash
import dash_cytoscape as cyto
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_daq as daq
import numpy as np
import json
import tensorflow as tf
import itertools

from .predictor import get_model
# app = dash.Dash(__name__)

#TODO: REFACTOR (lots of repitition, won't work if model layers change)

# def get_model(model_path):
#     """Loads pretrained Keras functional model"""
#     model = tf.keras.models.load_model(model_path)
#     return model

# def classifier(model, data):
#     """Returns class prediction"""
#     pred_proba = model.predict(data)
#     pred = int(np.argmax(pred_proba, axis=-1))
#     return pred, pred_proba


def input_layer_weights():
    neurons = clf.layers[1].weights[0].shape[0]
    input_weights = {}
    for i in list(range(neurons)):
        key = f"x{i}"
        input_weights[key] = np.array(clf.layers[1].weights[0][i])
    return input_weights


def dense_layer_weights(lyr, src):
    if lyr == 'y':
        layer_num = 7
    else:
        layer_num = int(lyr[1:])
    src_key = src.split('-')[0]
    neurons = clf.layers[layer_num].weights[0].shape[0] #18
    dense_weights = {}
    for i in list(range(neurons)):
        key = f"{src_key}-{i}"
        dense_weights[key] = np.array(clf.layers[layer_num].weights[0][i])
    return dense_weights


def make_permutations(src, trg):
    if trg == 'y':
        layer_num = 7
    else:
        layer_num = int(trg[1])
    if src != 'x':
        src_key = f"{src}-"
    else:
        src_key = "x"
    n1 = clf.layers[layer_num].weights[0].shape[0]
    n2 = clf.layers[layer_num].weights[0].shape[1]
    src_neurons, trg_neurons = [], []
    permutations = []
    for s in list(range(n1)):
        src_neurons.append(f"{src_key}{s}")
    for t in list(range(n2)):
        trg_neurons.append(f"{trg}-{t}")
    for r in itertools.product(src_neurons, trg_neurons):
        permutations.append((r[0], r[1]))
    return permutations


def make_weights():
    x_weights = input_layer_weights()
    # h_src = ['h2', 'h3', 'h4', 'h5', 'h6', 'y']
    # h_trg = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    h1_weights = dense_layer_weights('h2', 'h1')
    h2_weights = dense_layer_weights('h3', 'h2')
    h3_weights = dense_layer_weights('h4', 'h3')
    h4_weights = dense_layer_weights('h5', 'h4')
    h5_weights = dense_layer_weights('h6', 'h5')
    h6_weights = dense_layer_weights('y', 'h6')
    weight_groups = [x_weights, h1_weights, h2_weights, h3_weights, h4_weights, h5_weights, h6_weights]
    weights = []
    for group in weight_groups:
        for arr in group.values():
            for w in arr:
                weights.append(w)
    return weights

def make_edges(weights):
    xh1 = make_permutations('x', 'h1')
    h1h2 = make_permutations('h1', 'h2')
    h2h3 = make_permutations('h2', 'h3')
    h3h4 = make_permutations('h3', 'h4')
    h4h5 = make_permutations('h4', 'h5')
    h5h6 = make_permutations('h5', 'h6')
    h6y = make_permutations('h6', 'y')
    permutations = [xh1, h1h2, h2h3, h3h4, h4h5, h5h6, h6y]
    edge_pairs = []
    for p in permutations:
        for index, (source, target) in enumerate(p):
                edge_pairs.append((source, target, weights[index]))
    return edge_pairs


def make_parent_nodes():
    ids = ['inputs', 'dense1', 'dense2', 'dense3', 'dense4', 'dense5', 'dense6', 'outputs']
    labels = ['InputLayer', 'Dense1', 'Dense2', 'Dense3', 'Dense4', 'Dense5', 'Dense6', 'Outputs']
    classes = ['inputLayer', 'hiddenLayer', 'hiddenLayer', 'hiddenLayer', 'hiddenLayer', 'hiddenLayer', 'hiddenLayer', 'outputLayer']
    parent_nodes = list(zip(ids, labels, classes))
    return parent_nodes


def set_origin_points(layer_name):
    xy_origin = {
        'x': (0, 300),
        'h1': (1000, 250),
        'h2': (2500, 150),
        'h3': (5000, 50),
        'h4': (7500, 150),
        'h5': (9000, 250),
        'h6': (10000, 300),
        'y': (11000, 400)
    }
    index = list(enumerate(xy_origin.keys()))
    for i in index:
        if layer_name in i:
            layer_idx = i[0]
    return xy_origin[layer_name], layer_idx
    

def get_coords(xy_origin, layer_idx):
    x0 = xy_origin[0]
    y0 = xy_origin[1]
    
    if layer_idx == 0:
        neurons = clf.layers[layer_idx].output_shape[0][1]
    else:
        neurons = clf.layers[layer_idx].units
    slope = int(3200/neurons)
    xy_coords = []
    for i in list(range(neurons)):
        x = x0
        y = y0
        xy_coords.append((x, y))
        y0 += slope
    return xy_coords


def make_nodes(layer, neurons, parent):
    ids = []
    parents = []
    for n in list(range(neurons)):
        if layer == 'x':
            i = f"{layer}{n}"
        else:
            i = f"{layer}-{n}"
        ids.append(i)
        parents.append(parent)
    if parent == 'inputs':
        labels = ['n_files','total_mb','drizcorr','pctecorr','crsplit','subarray','detector','dtype','instr']
    elif parent == 'outputs':
        labels = ['2G', '8G', '16G', '64G']
    else:
        labels = []
        for n in list(range(neurons)):
            labels.append(str(n + 1))

    xy_origin, layer_idx = set_origin_points(layer)
    xy_coords = get_coords(xy_origin, layer_idx)
    xs = [x for (x, y) in xy_coords]
    ys = [y for (x, y) in xy_coords]
    nodes = list(zip(ids, labels, parents, xs, ys))
    return nodes


def make_node_groups():
    input_nodes = make_nodes('x', 9, 'inputs')
    h1 = make_nodes('h1', 18, 'dense1')
    h2 = make_nodes('h2', 32, 'dense2')
    h3 = make_nodes('h3', 64, 'dense3')
    h4 = make_nodes('h4', 32, 'dense4')
    h5 = make_nodes('h5', 18, 'dense5')
    h6 = make_nodes('h6', 9, 'dense6')
    outputs = make_nodes('y', 4, 'outputs')
    node_list = [input_nodes, h1, h2, h3, h4, h5, h6, outputs]
    node_groups = []
    for n in node_list:
        for id, label, parent, x, y in n:
            node_groups.append((id, label, parent, x, y))
    return node_groups


def edge_weight_clicks(src, trg):
    input_weights = input_layer_weights() #delete
    lyr, idx = trg.split('-')
    if src[0] == 'x':
        w = input_weights[src][int(idx)]
    elif src[0] == 'h':
        weights = dense_layer_weights(lyr, src)
        w = weights[src][int(idx)]
    else:
        return None
    return w

def node_bias_clicks(node):
    node_layer = node.split('-')
    if len(node_layer) == 1:
        bias = None
    else:
        lyr, idx = node.split('-') # h1, 0
        if lyr == 'y':
            layer_num = 7
        else:
            layer_num = int(lyr[1:])
        bias = float(np.array(clf.layers[layer_num].bias[int(idx)]))
    return bias



def nodes_edges(parent_nodes, node_groups, edge_pairs):
    nodes = [
    {
        'data': {'id': id, 'label': label}, 'classes': layerclass
    }
    for id, label, layerclass in parent_nodes
    ]
    nodes.extend([
    {
        'data': {'id': id, 'label': label, 'parent': parent},
        #'classes': 'neurons',
        'position': {'x': x, 'y': y}
    }
    for id, label, parent, x, y in node_groups
    ])
    edges = [
        {'data': {'source': source, 'target': target, 'weight': weight}}
        #, 'classes': 'weights'}
        for source, target, weight in edge_pairs
    ]
    return nodes, edges

def make_styles():
    styles = {
        'pre': {
            #'border': 'thin lightgrey solid',
            'overflowX': 'scroll',
            'display': 'inline-block',
            'float': 'left',
            'width': 400,
            'background-color': '#1b1f34',
            'margin': 0,
            'padding': 0,
            'text-align': 'center'
        }
    }
    return styles

def make_stylesheet():
    stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)',
                    'text-halign': 'center',
                }
            },
            {
                'selector': '.layers',
                'style': {
                    'width': 10,
                    'color': 'white',
                    'margin': 5
                    }
            },
            {
                'selector': '.inputLayer',
                'style': {
                    'width': 100,
                    'background-color': 'cyan',
                    'color': 'white'
                }
            },
            {
                'selector': '.outputLayer',
                'style': {
                    'width': 100,
                    'background-color': 'lightgreen',
                    'color': 'white'
                }
            },
            {
                'selector': '.hiddenLayer',
                'style': {
                    'background-color': 'beige',
                    'color': 'white'
                }
            },
            {
                'selector': '.heavy',
                'style': {
                    'background-color': '#0074D9'

                }
            }
            # {
            #     'selector': 'edge',
            #     #'selector': '[weight > 0]',
            #     'style': {
            #         'source-endpoint': 'inside-to-node',
            #         'target-endpoint': 'inside-to-node',
            #         'line-style': 'dashed',
            #         #'line-color': '#eee' 
            #     }
            # },
        ]
    return stylesheet

def make_neural_graph(NN=None):
    if NN is None:
        global clf
        clf = get_model('./models/mem_clf')
    weights = make_weights()
    edge_pairs = make_edges(weights)
    parent_nodes = make_parent_nodes()
    node_groups = make_node_groups()
    nodes, edges = nodes_edges(parent_nodes, node_groups, edge_pairs)
    return nodes, edges

# app.layout = html.Div([
#     # GRAPH
#     html.Div(children=[
#     cyto.Cytoscape(
#         id='cytoscape-compound',
#         layout={'name': 'preset'},
#         style={'width': '99vw', 'height': '70vh', 'display': 'inline-block', 'float': 'center', 'background-color': '#1b1f34'},
#         stylesheet=stylesheet,
#         elements=edges+nodes
#     ),
#     html.Div(children=[
#         html.P(id='cytoscape-tapNodeData-output', style=styles['pre']),
#         html.P(id='cytoscape-tapEdgeData-output', style=styles['pre']),
#         html.P(id='cytoscape-mouseoverNodeData-output', style=styles['pre']),
#         html.P(id='cytoscape-mouseoverEdgeData-output', style=styles['pre']),
#     ], style={'width': '100%', 'margin': 0, 'padding': 0, 'display': 'inline-block', 'float': 'left', 'background-color': '#1b1f34'})
#     ]),
#     # CONTROLS 
#     html.Div(children=[
#     # INPUT DROPDOWNS
#     html.Div(id='Xi', children=[
#         # INPUTS LEFT COL
#         html.Div(children=[
#             html.Label([
#                 html.Label("INSTR", style={'padding': 5, 'text-valign': 'center'}),
#                     dcc.Dropdown(
#                         id='instr-state',
#                         options=[{'label': i, 'value': i} for i in ['ACS', 'COS', 'STIS', 'WFC3']],
#                         value='ACS',
#                         style={'color': 'black', 'width': 130, 'display': 'inline-block', 'float': 'right'}
#                     )
#                     ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 250}),
#             html.Label([
#                 html.Label("DTYPE", style={'padding': 5}),
#                     dcc.Dropdown(
#                         id='dtype-state',
#                         options=[{'label': i, 'value': i} for i in ['SINGLETON', 'ASSOCIATION']],
#                         value='SINGLETON',
#                         style={'color': 'black', 'width': 130, 'display': 'inline-block', 'float': 'right'}
#                     ),
#                     ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 250}),
#             html.Label([
#                 html.Label("DETECTOR", style={'padding': 5}),
#                     dcc.Dropdown(
#                         id='detector-state',
#                         options=[{'label': i, 'value': i} for i in ['UVIS', 'WFC', 'IR', 'HRC', 'SBC']],
#                         value='UVIS',
#                         style={'color': 'black', 'width': 130, 'display': 'inline-block', 'float': 'right'}
#                     ),
#                     ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 250}),
#             html.Label([
#                 html.Label("SUBARRAY", style={'padding': 5}),
#                     dcc.Dropdown(
#                         id='subarray-state',
#                         options=[{'label': i, 'value': i} for i in ['TRUE', 'FALSE']],
#                         value='FALSE',
#                         style={'color': 'black', 'width': 130, 'display': 'inline-block', 'float': 'right'}
#                     ),
#                     ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 250}),
#             html.Label([
#                 html.Label("PCTECORR", style={'padding': 5}),
#                     dcc.Dropdown(
#                         id='pctecorr-state',
#                         options=[{'label': i, 'value': i} for i in ['OMIT', 'PERFORM']],
#                         value='PERFORM',
#                         style={'color': 'black', 'width': 130, 'display': 'inline-block', 'float': 'right'}
#                     ),
#                     ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 250}),
#             # END outputs Left Col
#         ], style={'display': 'inline-block', 'float': 'left', 'width': 270, 'margin': 10, 'padding': 5}),
#         # INPUTS RIGHT COL
#         html.Div(children=[
#             html.Label([
#                 html.Label("DRIZCORR", style={'padding': 5}),
#                     dcc.Dropdown(
#                         id='drizcorr-state',
#                         options=[{'label': i, 'value': i} for i in ['OMIT', 'PERFORM']],
#                         value='PERFORM',
#                         style={'color': 'black', 'width': 130, 'display': 'inline-block', 'float': 'right'}
#                     ),
#                     ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 250}),
#             html.Label([
#                 html.Label("CRSPLIT", style={'padding': 5}),
#                     daq.NumericInput(
#                         id='crsplit-state',
#                         value=2,
#                         min=0,
#                         max=2,
#                         style={'color': 'black', 'width': 130, 'display': 'inline-block', 'float': 'right'}
#                     ),
#                     ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 250}),
#             html.Label([
#                 html.Label("TOTAL_MB", style={'padding': 5}),
#                     daq.NumericInput(
#                         id='totalmb-state',
#                         value=4,
#                         min=0,
#                         max=900,
#                         style={'color': 'black', 'width': 130, 'display': 'inline-block', 'float': 'right'}
#                     ),
#                     ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 250}),
#             html.Label([
#                 html.Label("N_FILES", style={'padding': 5}),
#                     daq.NumericInput(
#                         id='nfiles-state',
#                         value=2,
#                         min=1,
#                         max=200,
#                         style={'color': 'black', 'width': 130, 'display': 'inline-block', 'float': 'right'}
#                     )
#                     ], style={'display': 'inline-block', 'float':'left', 'margin': 5, 'width': 250}),
            
#     # TODO SUBMIT BUTTON
#             html.Button('Submit', id='submit-button-state', n_clicks=0)

#             # END Input Right COL
#             ], style={'display': 'inline-block', 'float': 'left', 'width': 270, 'margin': 10, 'padding': 5}
#         # END INPUTS (BOTH COLS)
#         )], style={'width': 620, 'display': 'inline-block', 'float': 'left', 'padding': 5, 'background-color': '#242a44'}),
#     # OUTPUTS
#     html.Div(children=[
#         # MEMORY PRED VS ACTUAL
#         html.Div(children=[
#             # Memory Bin Pred vs Actual LED Display Values
#             daq.LEDDisplay(
#                 id='prediction-bin-output',
#                 label="Y PRED",
#                 labelPosition='bottom',
#                 #value='1',
#                 color="cyan",
#                 backgroundColor='#242a44',
#                 style={'padding': 5, 'width': 75, 'display': 'inline-block', 'float': 'left'}
#                 ),
#             # daq.LEDDisplay(
#             #     id='memory-bin-output',
#             #     label="Y TRUE",
#             #     labelPosition='bottom',
#             #     value='2',
#             #     color='lightgreen',
#             #     backgroundColor='#242a44',
#             #     style={'padding': 5, 'width': 75, 'display': 'inline-block', 'float': 'left'}
#             #     )
#             ], style={'width': 100, 'margin': 5, 'padding': 5, 'display': 'inline-block', 'float': 'left', 'background-color': '#242a44'}),
#             # Probabilities
#         html.Div(children=[
#             daq.GraduatedBar(
#                 id='p0',
#                 label='P(0)',
#                 labelPosition='right',
#                 step=0.1,
#                 min=0,
#                 max=1,
#                 #value=0.42,
#                 showCurrentValue=True,
#                 vertical=False,
#                 color='cyan',
#                 style={'color': 'black', 'display': 'inline-block', 'float': 'left', 'padding': 5}
#             ),
#             daq.GraduatedBar(
#                 id='p1',
#                 label='P(1)',
#                 labelPosition='right',
#                 step=0.1,
#                 min=0,
#                 max=1,
#                 #value=0.99,
#                 showCurrentValue=True,
#                 vertical=False,
#                 color='cyan',
#                 style={'color': 'black', 'display': 'inline-block', 'float': 'left', 'padding': 5}
#             ),
#             daq.GraduatedBar(
#                 id='p2',
#                 label='P(2)',
#                 labelPosition='right',
#                 step=0.1,
#                 min=0,
#                 max=1,
#                 #value=0.23,
#                 showCurrentValue=True,
#                 vertical=False,
#                 color='cyan',
#                 style={'color': 'black', 'display': 'inline-block', 'float': 'left', 'padding': 5}
#             ),
#             daq.GraduatedBar(
#                 id='p3',
#                 label='P(3)',
#                 labelPosition='right',
#                 step=0.1,
#                 min=0,
#                 max=1,
#                 #value=0.09,
#                 showCurrentValue=True,
#                 vertical=False,
#                 color='cyan',
#                 style={'color': 'black', 'display': 'inline-block', 'float': 'left', 'padding': 5}
#             )
#             # END Probabilities
#             ], style={'width': 360, 'padding': 10, 'display': 'inline-block', 'float': 'left', 'color': 'white', 'background-color': '#242a44'}),
#         # Memory GAUGE Predicted vs Actual
#         html.Div(children=[
#             daq.Gauge(
#                 id='memory-gauge-predicted',
#                 color={"gradient":True,"ranges":{"yellow":[0,2],"orange":[2,8],"red":[8,16],"blue":[16,64]}},
#                 #value=4.2,
#                 label='Predicted',
#                 labelPosition='bottom',
#                 units='GB',
#                 showCurrentValue=True,
#                 max=64,
#                 min=0,
#                 size=150,
#                 style={'color': 'white', 'display': 'inline-block', 'float': 'left'}
#                 )
#             # daq.Gauge(
#             #     id='memory-gauge-actual',
#             #     color={"gradient":True,"ranges":{"yellow":[0,2],"orange":[2,8],"red":[8,16],"blue":[16,64]}},
#             #     value=16,
#             #     label='Actual',
#             #     labelPosition='bottom',
#             #     units='GB',
#             #     showCurrentValue=True,
#             #     max=64,
#             #     min=0,
#             #     size=150,
#             #     style={'color': 'white', 'display': 'inline-block', 'float': 'left'}
#             #     )
#             ],
#             style={'width': 375, 'display': 'inline-block', 'float': 'left', 'color': 'white', 'background-color': '#242a44'}),
#             ])
#     # END Controls and Outputs
#     ], style={'width': '100%',  'display': 'inline-block', 'float': 'left', 'background-color': '#242a44'})
    
# # END app layout
# ], style={'width': '100%', 'height': '100%','background-color': '#242a44', 'color':'white'})


# @app.callback(
#     Output('cytoscape-learning-weights-output', 'children'),
#     Input('cytoscape-compound', 'tapNodeData'))
# @app.callback(Output('cytoscape-compound', 'stylesheet'),


# @app.callback([
#     Output('prediction-bin-output', 'value'),
#     Output('memory-gauge-predicted', 'value'),
#     Output('p0', 'value'),
#     Output('p1', 'value'),
#     Output('p2', 'value'),
#     Output('p3', 'value')],
#     Input('submit-button-state', 'n_clicks'),
#     State('nfiles-state', 'value'),
#     State('totalmb-state', 'value'),
#     State('drizcorr-state', 'value'),
#     State('pctecorr-state', 'value'),
#     State('crsplit-state', 'value'),
#     State('subarray-state', 'value'),
#     State('detector-state', 'value'),
#     State('dtype-state', 'value'),
#     State('instr-state', 'value')
#     )
# def update_output(n_clicks, n_files, total_mb, drizcorr, pctecorr, crsplit, subarray, detector, dtype, instr):
#     if n_clicks > 0:
#         x_features = predictor.read_inputs(n_files, total_mb, drizcorr, pctecorr, crsplit, subarray, detector, dtype, instr)
#         output_preds = predictor.make_preds(x_features)
#         # predictions = {"memBin": membin, "memVal": memval, "clockTime": clocktime}
#         #{"predictions": predictions, "probabilities": pred_proba}
#         membin = output_preds['predictions']['memBin']
#         memval = output_preds['predictions']['memVal']
#         proba = output_preds['probabilities'][0]
#         print(proba)
#         p0, p1, p2, p3 = proba[0], proba[1], proba[2], proba[3]
        
#         n_clicks=0
#         return [membin, memval, p0, p1, p2, p3]



# @app.callback(
#     Output('cytoscape-tapNodeData-output', 'children'),
#     Input('cytoscape-compound', 'tapNodeData'))
# def displayTapNodeData(data):
#     if data:
#         node = data['id']
#         if node[0] not in ['x', 'i']:
#             b = node_bias_clicks(node)
#         else:
#             b = None
#         return f"bias: {node} = {str(b)}"



# @app.callback(Output('cytoscape-tapEdgeData-output', 'children'),
#                 Input('cytoscape-compound', 'tapEdgeData'))
# def displayTapEdgeData(data):
#     if data:
#         src = data['source'] # x1
#         trg = data['target'] # h1-1
#         w = edge_weight_clicks(src, trg)
#         return f"weight: {src} and {trg} = {str(w)}"


# @app.callback(Output('cytoscape-mouseoverNodeData-output', 'children'),
#                 Input('cytoscape-compound', 'mouseoverNodeData'))
# def displayTapNodeData(data):
#     if data:
#         node = data['id']
#         if node[0] not in ['x', 'i']:
#             b = node_bias_clicks(node)
#         else:
#             b = None
#         return f"bias: {node} = {str(b)}"


# @app.callback(Output('cytoscape-mouseoverEdgeData-output', 'children'),
#                 Input('cytoscape-compound', 'mouseoverEdgeData'))
# def displayTapEdgeData(data):
#     if data:
#         src = data['source'] # x1
#         trg = data['target'] # h1-1
#         w = edge_weight_clicks(src, trg)
#         return f"weight: {src} and {trg} = {str(w)}"


# if __name__ == '__main__':
#     app.run_server(debug=True)




# # 	wallclock	memory	    mem_bin
# # 	26266.0	    14.138928	2.0
# # 'n_files': 45.0, 'total_mb': 700.0,
# # v3 model pred: 2
# # v3 model pred proba: array([[2.5013131e-01, 3.8796006e-04, 4.8727387e-01, 2.6220685e-01]], dtype=float32)
# # input_features = {
# #     'ic0k01010': {
# #         'x_files': 1.847015, 'x_size': 2.705386, 'drizcorr': 1, 'pctecorr': 1, 'crsplit': 2, 'subarray': 0, 'detector': 1, 'dtype': 1, 'instr': 3
# #     }
# # }


# # d1_n0_weights = []
# # for xlabel, weights in input_weights.items():
# #     d1_n0_weights.append(weights[0]) # weights from all incoming connections to neuron 0 (denselayer1)


# # def calculate_weights(node):
# #     x_weights = input_weights[node]
# #     idx_min = np.argmin(weights)
# #     idx_max = np.argmax(weights)
# #     pos_idx = np.array(np.where(weights>0)).ravel()
# #     neg_idx = np.array(np.where(weights<0)).ravel()
    
# #     node_pairs = [(src, trg) for (src, trg) in edge_pairs if src == node]

# #     w_pos, w_neg = [], []
# #     w_minmax = []
# #     for (src, trg) in node_pairs:
# #         idx_trg = int(trg.split('-')[-1])
# #         if idx_trg in [idx_min, idx_max]:
# #             w_minmax.append((src, trg))
# #         for i, j in zip(pos_idx, neg_idx):
# #             if idx_trg == i:
# #                 w_pos.append((src, trg))
# #             elif idx_trg == j:
# #                 w_neg.append((src, trg))
    
# #     return w_pos, w_neg, w_minmax