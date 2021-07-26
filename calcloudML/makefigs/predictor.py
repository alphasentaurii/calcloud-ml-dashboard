"""This module loads a pre-trained ANN to predict job resource requirements for HST.
# 1 - load job metadata inputs from text file in s3
# 2 - encode strings as int/float values in numpy array
# 3 - load models and generate predictions
# 4 - return preds as json to parent lambda function
"""
import numpy as np
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf

def get_model(model_path):
    """Loads pretrained Keras functional model"""
    model = tf.keras.models.load_model(model_path)
    return model

# load models
clf = get_model("./models/mem_clf/")
mem_reg = get_model("./models/mem_reg/")
wall_reg = get_model("./models/wall_reg/")

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
                if v == "true":
                    subarray = 1
                else:
                    subarray = 0
            if k == "drizcorr":
                if v == "perform":
                    drizcorr = 1
                else:
                    drizcorr = 0
            if k == "pctecorr":
                if v == "perform":
                    pctecorr = 1
                else:
                    pctecorr = 0
            if k == "crsplit":
                if v == "NaN":
                    crsplit = 0
                elif v == "1.0":
                    crsplit = 1
                else:
                    crsplit = 2
            if k == "dtype":
                if v == 'ASN':
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
        pt.lambdas_ = np.array([-1.51, -0.12])
        xt = pt.transform(x)
        # normalization (zero mean, unit variance)
        f_mean, f_sigma = 0.5682815234265285, 0.04222565843608133
        s_mean, s_sigma = 1.6250374589283951, 1.0396138451086632
        x_files = np.round(((xt[0, 0] - f_mean) / f_sigma), 5)
        x_size = np.round(((xt[0, 1] - s_mean) / s_sigma), 5)
        X = np.array([x_files, x_size, X[2], X[3], X[4], X[5], X[6], X[7], X[8]]).reshape(1, -1)
        return X


def make_preds(x_features):
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
    prep = Preprocess(x_features)
    prep.inputs = prep.scrub_keys()
    X = prep.transformer()
    # Predict Memory Allocation (bin and value preds)
    membin, pred_proba = classifier(clf, X)
    memval = np.round(float(regressor(mem_reg, X)), 2)
    # Predict Wallclock Allocation (execution time in seconds)
    clocktime = int(regressor(wall_reg, X))
    predictions = {"memBin": membin, "memVal": memval, "clockTime": clocktime}
    return {"predictions": predictions, "probabilities": pred_proba}


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