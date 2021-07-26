import dash
import dash_cytoscape as cyto
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import json
import tensorflow as tf
import itertools

#TODO: show/highlight lines with weight > .5


app = dash.Dash(__name__)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

def get_model(model_path):
    """Loads pretrained Keras functional model"""
    model = tf.keras.models.load_model(model_path)
    return model

def classifier(model, data):
    """Returns class prediction"""
    pred_proba = model.predict(data)
    pred = int(np.argmax(pred_proba, axis=-1))
    return pred, pred_proba


def input_layer_weights(clf):
    neurons = clf.layers[1].weights[0].shape[0]
    input_weights = {}
    for i in list(range(neurons)):
        key = f"x{i}"
        input_weights[key] = np.array(clf.layers[1].weights[0][i])
    return input_weights


def dense_layer_weights(clf, lyr, src):
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


def make_edges():
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
        for source, target in p:
            edge_pairs.append((source, target))
    return edge_pairs


def make_parent_nodes():
    ids = ['inputs', 'dense1', 'dense2', 'dense3', 'dense4', 'dense5', 'dense6', 'outputs']
    labels = ['InputLayer', 'Dense1', 'Dense2', 'Dense3', 'Dense4', 'Dense5', 'Dense6', 'Outputs']
    parent_nodes = list(zip(ids, labels))
    return parent_nodes


def get_positions(layer):
    p_dict = {
    'x': { # 9
        'xs': [50, 50, 50, 50, 50, 50, 50, 50, 50],
        'ys': [50, 150, 250, 350, 450, 550, 650, 750, 850]},
    'h1' : { # 18
        'xs' : [200, 250, 200, 250, 200, 250, 200, 250, 200, 250, 200, 250, 200, 250, 200, 250, 200, 250],
        'ys' : [100, 100, 150, 150, 200, 200, 250, 250, 300, 300, 350, 350, 400, 400, 450, 450, 500, 500]
    },
    'h2' : { # 32
        'xs': [350, 400, 350, 400, 350, 400, 350, 400, 350, 400, 350, 400, 350, 400, 350, 400, 350, 400, 350, 400, 350, 400, 350, 400, 350, 400, 350, 400, 350, 400, 350, 400],
        'ys': [100, 100, 150, 150, 200, 200, 250, 250, 300, 300, 350, 350, 400, 400, 450, 450, 500, 500, 550, 550, 600, 600, 650, 650, 700, 700, 750, 750, 800, 800, 850, 850]
    },
    'h3' : { # 64
        'xs': [500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650, 500, 550, 600, 650],
        'ys': [100, 100, 100, 100, 150, 150, 150, 150, 200, 200, 200, 200, 250, 250, 250, 250, 300, 300, 300, 300, 350, 350, 350, 350, 400, 400, 400, 400, 450, 450, 450, 450, 500, 500, 500, 500, 550, 550, 550, 550, 600, 600, 600, 600, 650, 650, 650, 650, 700, 700, 700, 700, 750, 750, 750, 750, 800, 800, 800, 800, 850, 850, 850, 850]
    },
    'h4' : {
        'xs': [750, 800, 750, 800, 750, 800, 750, 800, 750, 800, 750, 800, 750, 800, 750, 800, 750, 800, 750, 800, 750, 800, 750, 800, 750, 800, 750, 800, 750, 800, 750, 800, 750, 800, 750, 800],
        'ys': [100, 100, 150, 150, 200, 200, 250, 250, 300, 300, 350, 350, 400, 400, 450, 450, 500, 500, 550, 550, 600, 600, 650, 650, 700, 700, 750, 750, 800, 800, 850, 850]
        },
    'h5' : {
        'xs': [900, 950, 900, 950, 900, 950, 900, 950, 900, 950, 900, 950, 900, 950, 900, 950, 900, 950],
        'ys': [100, 100, 150, 150, 200, 200, 250, 250, 300, 300, 350, 350, 400, 400, 450, 450, 500, 500]
    },
    'h6' : {
        'xs': [1050, 1050, 1050, 1050, 1050, 1050, 1050, 1050, 1050],
        'ys': [100, 150, 200, 250, 300, 350, 400, 450, 500]
    },
    'y' : {
        'xs': [1200, 1200, 1200, 1200],
        'ys': [50, 150, 250, 350]
    }}
    return p_dict[layer]


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
    x_pos = get_positions(layer)['xs']
    y_pos = get_positions(layer)['ys']
    nodes = list(zip(ids, labels, parents, x_pos, y_pos))
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

clf = get_model('./models/mem_clf')
input_weights = input_layer_weights(clf)
edge_pairs = make_edges()
parent_nodes = make_parent_nodes()
node_groups = make_node_groups()

nodes = [
    {
        'data': {'id': id, 'label': label}
    }
    for id, label in parent_nodes
]
nodes.extend([
    {
        'data': {'id': id, 'label': label, 'parent': parent},
        'position': {'x': x, 'y': y}
    }
    for id, label, parent, x, y in node_groups
])


edges = [
    {'data': {'source': source, 'target': target}}
    for source, target in edge_pairs
]


def edge_weight_clicks(src, trg):
    lyr, idx = trg.split('-')
    if src[0] == 'x':
        w = input_weights[src][int(idx)]
    elif src[0] == 'h':
        weights = dense_layer_weights(clf, lyr, src)
        w = weights[src][int(idx)]
    else:
        return None
    return w

def node_bias_clicks(node):
    lyr, idx = node.split('-') # h1, 0
    if lyr == 'y':
        layer_num = 7
    else:
        layer_num = int(lyr[1:])
    bias = float(np.array(clf.layers[layer_num].weights[1][int(idx)]))
    return bias



app.layout = html.Div([
html.Div(children=[
    cyto.Cytoscape(
        id='cytoscape-compound',
        layout={'name': 'preset'},
        style={'width': '80vw', 'height': '850px', 'display': 'inline-block', 'float': 'left'},
        stylesheet=[
            {
                'selector': 'node',
                'style': {'content': 'data(label)'}
            },
            {
                'selector': '.layers',
                'style': {'width': 3}
            },
            {
                'selector': '.neurons',
                'style': {'line-style': 'dashes', 'alpha': 0.7}
            }
        ],
        elements=edges+nodes
    ),
    html.Div(children=[
    #html.Pre(id='cytoscape-tapNodeData-json', style=styles['pre']),
    html.P(id='cytoscape-tapNodeData-output', style=styles['pre']),
    html.P(id='cytoscape-tapEdgeData-output', style=styles['pre']),
    html.P(id='cytoscape-mouseoverNodeData-output', style=styles['pre']),
    html.P(id='cytoscape-mouseoverEdgeData-output', style=styles['pre'])
    ], style={'width': '15vw', 'float': 'left'})
    ])
])




@app.callback(Output('cytoscape-tapNodeData-output', 'children'),
                Input('cytoscape-compound', 'tapNodeData'))
def displayTapNodeData(data):
    if data:
        if data['parent'] != 'inputs':
            node = data['id']
            b = node_bias_clicks(node)
        return "neuron: " + data['id'] + " bias = " + str(b)


@app.callback(Output('cytoscape-tapEdgeData-output', 'children'),
                Input('cytoscape-compound', 'tapEdgeData'))
def displayTapEdgeData(data):
    if data:
        src = data['source'] # x1
        trg = data['target'] # h1-1
        w = edge_weight_clicks(src, trg)
        return "weight: " + data['source'] + " and " + data['target'] + " = " + str(w)


@app.callback(Output('cytoscape-mouseoverNodeData-output', 'children'),
                Input('cytoscape-compound', 'mouseoverNodeData'))
def displayTapNodeData(data):
    if data:
        return "hovered neuron: " + data['label']


@app.callback(Output('cytoscape-mouseoverEdgeData-output', 'children'),
                Input('cytoscape-compound', 'mouseoverEdgeData'))
def displayTapEdgeData(data):
    if data:
        return "hovered edge " + data['source'].upper() + " and " + data['target'].upper()

if __name__ == '__main__':
    app.run_server(debug=True)