"""This module loads a pre-trained ANN to predict job resource requirements for HST.
# 1 - load job metadata inputs from text file in s3
# 2 - encode strings as int/float values in numpy array
# 3 - load models and generate predictions
# 4 - return preds as json to parent lambda function
"""
import numpy as np
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf
import pickle

def get_model(model_path):
    """Loads pretrained Keras functional model"""
    model = tf.keras.models.load_model(model_path)
    return model

# load models
# clf = get_model("./models/mem_clf/")
# mem_reg = get_model("./models/mem_reg/")
# wall_reg = get_model("./models/wall_reg/")


def read_inputs(n_files, total_mb, drizcorr, pctecorr, 
    crsplit, subarray, detector, dtype, instr):
    x_features = {
        'n_files': n_files,
        'total_mb': total_mb,
        'drizcorr': drizcorr,
        'pctecorr': pctecorr,
        'crsplit': crsplit,
        'subarray': subarray,
        'detector': detector,
        'dtype': dtype,
        'instr': instr
        }
    return x_features


# input layer
# x = np.array([[7, 20, 0, 0, 0, 1, 0, 0, 0]]).reshape(-1, 1)

#np.array([n_files, total_mb, drizcorr, pctecorr, crsplit, subarray, detector, dtype, instr])

# hidden layers (# neurons)
# h1 = 18
# h2 = 32
# h3 = 64
# h4 = 32
# h5 = 18
# h6 = 9

# output layer
# y = 0

# outputs
# y1 = ['mem_bin']
# y2 = ['memory']
# y3 = ['wallclock']


def classifier(model, data):
    """Returns class prediction"""
    pred_proba = model.predict(data)
    pred = int(np.argmax(pred_proba, axis=-1))
    return pred, pred_proba


def regressor(model, data):
    """Returns Regression model prediction"""
    pred = model.predict(data)
    return pred


class Preprocess:
    def __init__(self, x_features):
        self.x_features = x_features
        self.inputs = None
        self.lambdas = None
        self.f_mean = None
        self.f_sigma = None
        self.s_mean = None
        self.s_sigma = None


    def scrub_keys(self):
        n_files = 0
        total_mb = 0
        detector = 0
        subarray = 0
        drizcorr = 0
        pctecorr = 0
        crsplit = 0

        for k, v in self.x_features.items():
            if k == "n_files":
                n_files = int(v)
            if k == "total_mb":
                total_mb = int(np.round(float(v), 0))
            if k == "detector":
                if v in ["UVIS", "WFC"]:
                    detector = 1
                else:
                    detector = 0
            if k == "subarray":
                if v == "TRUE":
                    subarray = 1
                else:
                    subarray = 0
            if k == "drizcorr":
                if v == "PERFORM":
                    drizcorr = 1
                else:
                    drizcorr = 0
            if k == "pctecorr":
                if v == "PERFORM":
                    pctecorr = 1
                else:
                    pctecorr = 0
            if k == "crsplit":
                if v == 0:
                    crsplit = 0
                elif v == 1:
                    crsplit = 1
                else:
                    crsplit = 2
            if k == "dtype":
                if v == 'ASSOCIATION':
                    dtype = 1
                else:
                    dtype = 0
            if k == "instr":
                if v == 'ACS':
                    instr = 0
                elif v == 'COS':
                    instr = 1
                elif v == 'STIS':
                    instr = 2
                else:
                    instr = 3

        inputs = np.array([n_files, total_mb, drizcorr, pctecorr, crsplit, subarray, detector, dtype, instr])
        return inputs

    
    def load_pt_data(self, pt_file='./data/pt_transform'):
        with open(pt_file, 'rb') as pick:
            pt_data = pickle.load(pick)
        
        self.lambdas = np.array([pt_data['lambdas'][0], pt_data['lambdas'][1]])
        self.f_mean = pt_data['f_mean']
        self.f_sigma = pt_data['f_sigma']
        self.s_mean = pt_data['s_mean']
        self.s_sigma = pt_data['s_sigma']
        
        return self



# {'lambdas': array([-1.80648272,  0.0026787 ]),
#  'f_mean': 0.4992818949480044,
#  'f_sigma': 0.029090406850771935,
#  's_mean': 2.5497485350443014,
#  's_sigma': 1.5009046818802525}
    
    def transformer(self):
        """applies yeo-johnson power transform to first two indices of array (n_files, total_mb) using lambdas, mean and standard deviation calculated for each variable prior to model training.

        Returns: X inputs as 2D-array for generating predictions
        """
        X = self.inputs
        n_files = X[0]
        total_mb = X[1]
        # apply power transformer normalization to continuous vars
        x = np.array([[n_files], [total_mb]]).reshape(1, -1)
        pt = PowerTransformer(standardize=False)
        pt.lambdas_ = self.lambdas #-1.80648272,  0.0026787
        #pt.lambdas_ = np.array([-1.51, -0.12])
        xt = pt.transform(x)
        # normalization (zero mean, unit variance)
        # f_mean, f_sigma = 0.5682815234265285, 0.04222565843608133
        #s_mean, s_sigma = 1.6250374589283951, 1.0396138451086632
        x_files = np.round(((xt[0, 0] - self.f_mean) / self.f_sigma), 5)
        x_size = np.round(((xt[0, 1] - self.s_mean) / self.s_sigma), 5)
        X = np.array([x_files, x_size, X[2], X[3], X[4], X[5], X[6], X[7], X[8]]).reshape(1, -1)
        return X


def make_preds(x_features, NN=None):
    """Predict Resource Allocation requirements for memory (GB) and max execution `kill time` / `wallclock` (seconds) using three pre-trained neural networks.

    MEMORY BIN: classifier outputs probabilities for each of the four bins ("target classes"). The class with the highest probability score is considered the final predicted outcome (y). This prediction variable represents which of the 4 possible memory bins is most likely to meet the minimum required needs for processing an HST dataset (ipppssoot) successfully according to the given inputs (x).

    Memory Bin Sizes (target class "y"):
    0: < 2GB
    1: 2-8GB
    2: 8-16GB
    3: >16GB

    WALLCLOCK REGRESSION: regression generates estimate for specific number of seconds needed to process the dataset using the same input data. This number is then tripled in Calcloud for the sake of creating an extra buffer of overhead in order to prevent larger jobs from being killed unnecessarily.

    MEMORY REGRESSION: A third regression model is used to estimate the actual value of memory needed for the job. This is mainly for the purpose of logging/future analysis and is not currently being used for allocating memory in calcloud jobs.
    """
    global clf
    global mem_reg
    if NN is None:
        clf = get_model('./models/mem_clf')
        mem_reg = get_model("./models/mem_reg/")
        #wall_reg = get_model("./models/wall_reg/")
    else:
        clf = NN['clf']
        mem_reg = NN['mem_reg']
    prep = Preprocess(x_features)
    prep.inputs = prep.scrub_keys()
    prep.load_pt_data()
    X = prep.transformer()
    # Predict Memory Allocation (bin and value preds)
    membin, pred_proba = classifier(clf, X)
    P = pred_proba[0]
    p0, p1, p2, p3 = P[0], P[1], P[2], P[3]
    memval = np.round(float(regressor(mem_reg, X)), 2)
    # Predict Wallclock Allocation (execution time in seconds)
    # clocktime = int(regressor(wall_reg, X))
    # predictions = {"memBin": membin, "memVal": memval, "clockTime": clocktime}
    # return {"predictions": predictions, "probabilities": pred_proba}
    memory_predictions = [membin, memval, p0, p1, p2, p3]
    return memory_predictions

        # predictions = {"memBin": membin, "memVal": memval, "clockTime": clocktime}
        #{"predictions": predictions, "probabilities": pred_proba}
        # membin = output_preds['predictions']['memBin']
        # memval = output_preds['predictions']['memVal']
        # proba = output_preds['probabilities'][0]

        


# input_arr = np.array([1.847015, 2.705386, 1, 1, 2, 0, 1, 1, 3])
def single_neuron(layer_num, input_arr, neuron):
    # get weights for each neuron
    w_dense = np.array(clf.layers[layer_num].weights[0])
    weights = [w[neuron] for w in w_dense]
    # get bias values
    b = np.array(clf.layers[layer_num].bias)[neuron]
    wxb = np.sum(weights * input_arr) + b
    n = np.max([0, wxb])
    
    return n # 1.4060400382443516

def layer_neurons(layer_num, input_arr):
    w_dense = np.array(clf.layers[layer_num].weights[0])
    n_neurons = w_dense.shape[1]
    neuron_values = []
    for n in list(range(n_neurons)):
        s = single_neuron(layer_num, input_arr, n)
        neuron_values.append(s)
    return np.array(neuron_values)

def calculate_neurons(input_arr):
    # n_layers = len(clf.layers)
    # neurons = []
    # count = 1
    # for L in list(range(1, n_layers)):
    #     if count == 1:
    #         input_arr = input_arr
    #     else:
    #         input_arr = neurons[count-1]
    #     N = layer_neurons(L, input_arr)
    #     neurons.append(N)
    #     count+=1
    n1 = layer_neurons(1, input_arr)
    n2 = layer_neurons(2, n1)
    n3 = layer_neurons(3, n2)
    n4 = layer_neurons(4, n3)
    n5 = layer_neurons(5, n4)
    n6 = layer_neurons(6, n5)
    n7 = layer_neurons(7, n6)
    neurons = [n1, n2, n3, n4, n5, n6, n7]
    return neurons

def softmax_activation(neurons):
    # array([8533.97208947, 7670.27113491, 1830.08351255, 2598.71759253])
    out = neurons[-1]
    e0 = out[0]**out[0]
    e1 = out[1]**out[1]
    e2 = out[2]**out[2]
    e3 = out[3]**out[3]
    print(e0, e1, e2, e3)
    p0 = e0 / (e0 + e1 + e2 + e3)
    p1 = e1 / (e0 + e1 + e2 + e3)
    p2 = e2 / (e0 + e1 + e2 + e3)
    p3 = e3 / (e0 + e1 + e2 + e3)
    return p0, p1, p2, p3
# x_features = read_inputs(n_files=10, total_mb=30, drizcorr='perform', pctecorr='perform', crsplit=1.0, subarray='false', detector='UVIS', dtype='ASN', instr='ACS')

# {'n_files': 10,
#  'total_mb': 30,
#  'drizcorr': 'perform',
#  'pctecorr': 'perform',
#  'crsplit': 1.0,
#  'subarray': 'false',
#  'detector': 'UVIS',
#  'dtype': 'ASN',
#  'instr': 'ACS'}

# inputs
# inputs = scrub_keys(x_features)
# array([10, 30,  1,  1,  2,  0,  1,  1,  0])

# X = transformer(inputs)
# array([[1.81, 1.14, 1.  , 1.  , 2.  , 0.  , 1.  , 1.  , 0.  ]])

# membin, pred_proba = classifier(clf, X)
# 0
# array([[5.72e-01, 4.28e-01, 3.33e-31, 0.00e+00]], dtype=float32)