
# STANDARD libraries
import sys, os, glob
import pandas as pd
import numpy as np
from numpy import log 
import IPython.display
from IPython.display import Image
import math
import pickle
import datetime as dt
import tzlocal as tz

# Pre-processing
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split

# modeling
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
# NNs
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

# eval
from sklearn import metrics
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_curve, auc, recall_score
from sklearn.metrics import average_precision_score, precision_score, f1_score


class HubbleSpaceTelescope:

# PREPROCESSING ************
    @staticmethod
    def transformer(n_files, total_mb):
        x = np.array([[n_files],[total_mb]]).reshape(1,-1)
        pt = PowerTransformer(standardize=False)
        pt.lambdas_ = np.array([-0.96074766, -0.32299156])
        xt = pt.transform(x)
        f_mean, f_sigma  = 0.653480238393804, 0.14693259765350208
        s_mean, s_sigma = 1.1648725537429683, 0.7444473983812471
        x_files = np.round(((xt[0,0] - f_mean) / f_sigma), 5)
        x_size = np.round(((xt[0,1] - s_mean) / s_sigma),5)
        print(f"x_files: {x_files}\nx_size: {x_size}")
        return x_files, x_size

    @staticmethod
    def normalize(input_matrix):
        m1 = input_matrix[:, 0]
        M1 = M1 = (m1 - np.mean(m1)) / np.std(m1)
        if len(input_matrix.shape) == 2:
            m2 = input_matrix[:, 1]
            M2 = (m2 - np.mean(m2)) / np.std(m2)
            m_norm = np.stack([M1, M2], axis=1)
            return m_norm
        else:
            return M1

    @staticmethod
    def scrub_keys(data):
        # copy columns to validate before making changes
        df = data.copy()
        cols = df.columns
        # DRIZCORR
        if 'DRIZCORR' in cols:
            df['drizcorr'] = df['DRIZCORR']
            df['drizcorr'].loc[(df['drizcorr'].isna()) | (df['drizcorr'] == 'OMIT')] = 0
            df['drizcorr'].loc[(df['drizcorr'] == 'PERFORM') | (df['drizcorr'] == 'COMPLETE')] = 1
            df['drizcorr'] = df['drizcorr'].astype('int64')
            df.drop(columns=['DRIZCORR'], axis=1, inplace=True)
        # PCTECORR
        if 'PCTECORR' in cols:
            df['pctecorr'] = df['PCTECORR']
            df['pctecorr'].loc[(df['pctecorr'].isna()) | (df['pctecorr'] == 'OMIT')] = 0
            df['pctecorr'].loc[df['pctecorr'] == 'PERFORM'] = 1
            df['pctecorr'] = df['pctecorr'].astype('int64')
            df.drop(columns=['PCTECORR'], axis=1, inplace=True)
        # CRSPLIT  
        if 'CRSPLIT' in cols:
            df['crsplit'] = df['CRSPLIT']
            df['crsplit'].loc[(df['crsplit'].isna()) | (df['crsplit'] == 0)] = 0
            df['crsplit'].loc[df['crsplit'] == 1] = 1
            df['crsplit'].loc[df['crsplit'] > 1] = 2
            df['crsplit'] = df['crsplit'].astype('int64')
            df.drop(columns=['CRSPLIT'], axis=1, inplace=True)
        # SUBARRAY
        if 'SUBARRAY' in cols:
            df['subarray'] = df['SUBARRAY']
            # convert boolean to integer
            df['subarray'].loc[df['subarray'] == True] = 1
            df['subarray'].loc[(df['subarray'] == False) | (df['subarray'].isna())] = 0
            df['subarray'] = df['subarray'].astype('int64')
            df.drop(columns=['SUBARRAY'], axis=1, inplace=True)
        # DETECTOR
        if 'DETECTOR' in cols:
            df['detector'] = df['DETECTOR']
            df['detector'].loc[(df['detector'] == 'UVIS') | 
                            (df['detector'] == 'WFC')] = 1
            df['detector'].loc[(df['detector'] == 'IR') | (df['detector'] == 'HRC') | 
                            (df['detector'] == 'SBC') | (df['detector'] == 'NUV') | 
                            (df['detector'] == 'FUV') | (df['detector'] == 'CCD') |
                            (df['detector'] == 'FUV-MAMA') | (df['detector'] == 'NUV-MAMA') |
                            (df['detector'].isna())] = 0
            df['detector'] = df['detector'].astype('int64')
            df.drop(columns=['DETECTOR'], axis=1, inplace=True)
        if 'ipst' in cols:
            df.drop(columns=['ipst'], axis=1, inplace=True)
        mem = df['memory']
        wall = df['wallclock']
        df.drop(columns=['wallclock', 'memory'], axis=1, inplace=True)
        df['wallclock'] = wall
        df['memory'] = mem
        return df

    # EDA ************
    @staticmethod
    def percentile_means(df, features, target):
        
        means = {}

        for v in features:
            vals = df[v]

            q25 = vals.quantile(.25)
            q50 = vals.quantile(.5)
            q75 = vals.quantile(.75)
            q95 = vals.quantile(.95)

            m0 = df[target].loc[vals <= q25].mean()
            m25 = df[target].loc[(vals >= q25) & (vals < q50)].mean()
            m50 = df[target].loc[(vals >= q50) & (vals < q75)].mean()
            m75 = df[target].loc[(vals >= q75) & (vals < q95)].mean()
            m95 = df[target].loc[vals >= q95].mean()

            means[v] = [m0, m25, m50, m75, m95]

        # line plt
        plt.figure(figsize=(15, 10))
        for m in means.values():
            plt.plot(m)
        plt.legend([f for f in means.keys()])
        xticks = ['min', '25%', '50%', '75%', '95%']
        plt.xticks(ticks=np.arange(5), labels=xticks, rotation=45)

        return means

    @staticmethod
    def hubble_scatter(df, X, Y='memory', instruments=None, bestfit=False):
        fig = plt.figure(figsize=(11,7))
        ax = fig.gca()

        if instruments is None:
            ax.scatter(df[X], df[Y])
            ax.set_xlabel(X)
            ax.set_ylabel(Y)
            ax.set_title(f"{X} vs. {Y}")
        else:
            cols = list(df.columns)
            if 'instr' in cols:
                instr_col = 'instr'

            else:
                instr_col = 'instr_enc'

            for i in instruments:
                if i == 'acs':
                    e = 0
                    c = 'blue'
                elif i == 'cos':
                    e = 1
                    c='lime'
                elif i == 'stis':
                    e = 2
                    c='red'
                elif i == 'wfc3':
                    e = 3
                    c = 'orange'
                if instr_col == 'instr':
                    ax.scatter((df[X].loc[df[instr_col] == i]), (df[Y].loc[df[instr_col] == i]), c=c, alpha=0.7)
                else:
                    ax.scatter((df[X].loc[df[instr_col] == e]), (df[Y].loc[df[instr_col] == e]), c=c, alpha=0.7)
    
                ax.set_xlabel(X)
                ax.set_ylabel(Y)
                ax.set_title(f"{X} vs. {Y}: {[i for i in instruments]}")
                if len(instruments) > 1:
                    ax.legend([i for i in instruments])
                
        if bestfit is True:
            x = df[X]
            y = df[Y]
            m, b = np.polyfit(x, y, 1) #slope, intercept
            plt.plot(x, m*x + b, 'k--'); # best fit line
        else:
            plt.show()


    # KDE Single (single feature, all 4 instruments)
    # # Kernel Density Estimates (distplots) for independent variables
    # plt.style.use('seaborn-bright')
    # fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(13,10))
    # sns.distplot(acs['n_files'], ax=ax[0][0], label='acs')
    # sns.distplot(cos['n_files'], ax=ax[0][1], color='lime')
    # sns.distplot(stis['n_files'],ax=ax[1][0], color='red')
    # sns.distplot(wfc3['n_files'],  ax=ax[1][1], color='blue')

    # ax[0][0].set_title('ACS')
    # ax[0][1].set_title('COS')
    # ax[1][0].set_title('STIS')
    # ax[1][1].set_title('WFC3')

    # fig.tight_layout()

    # # Kernel Density Estimates (distplots) for independent variables
    # plt.style.use('seaborn-bright')
    # fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12,12))
    # sns.distplot(acs['n_proc'], ax=ax[0][0], label='acs')
    # sns.distplot(cos['n_proc'], ax=ax[0][1], color='lime')
    # sns.distplot(stis['n_proc'],  ax=ax[1][0], color='red')
    # sns.distplot(wfc3['n_proc'], ax=ax[1][1], color='blue')
    # ax[0][0].set_title('ACS')
    # ax[0][1].set_title('COS')
    # ax[1][0].set_title('STIS')
    # ax[1][1].set_title('WFC3')
    # fig.tight_layout()


    # KDE_All - all instruments, keywords and target
    # # Kernel Density Estimates (distplots) for independent variables
    # plt.style.use('seaborn-bright')
    # fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(12,12))
    # sns.distplot(df['drizcorr'], ax=ax[0][0])
    # sns.distplot(df['pctecorr'], ax=ax[0][1])
    # sns.distplot(df['crsplit'], ax=ax[1][0])
    # sns.distplot(df['subarray'], ax=ax[1][1])
    # sns.distplot(df['detector'], ax=ax[2][0])
    # sns.distplot(df['memory'], ax=ax[2][1])

    # KDE_Transforms
    # # Kernel Density Estimates (distplots) for independent variables
    # plt.style.use('seaborn-bright')
    # fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(13,10))
    # sns.distplot(df['n_files'], ax=ax[0][0], color='blue')
    # sns.distplot(df_norm['x_files'], ax=ax[0][1], color='red')
    # sns.distplot(df['total_mb'],ax=ax[1][0], color='blue')
    # sns.distplot(df_norm['x_size'],  ax=ax[1][1], color='red')

    # ax[0][0].set_title('N_FILES (raw)')
    # ax[0][1].set_title('N_FILES (power transform)')
    # ax[1][0].set_title('TOTAL_MB (RAW)')
    # ax[1][1].set_title('TOTAL_MB (power transform)')

    # fig.tight_layout()


    # # BAR - 4 instruments
    # # number of processes vs memory for each instrument
    # plt.style.use('seaborn-bright')
    # fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12,12))
    # sns.barplot(data=acs, x='n_proc', y='memory', ax=ax[0][0], order=[0,1,2,3,4,5])
    # sns.barplot(data=cos, x='n_proc', y='memory', ax=ax[0][1], order=[0,1,2,3,4,5])
    # sns.barplot(data=stis, x='n_proc', y='memory', ax=ax[1][0], order=[0,1,2,3,4,5])
    # sns.barplot(data=wfc3, x='n_proc', y='memory', ax=ax[1][1], order=[0,1,2,3,4,5])
    # ax[0][0].set_title('ACS')
    # ax[0][1].set_title('COS')
    # ax[1][0].set_title('STIS')
    # ax[1][1].set_title('WFC3')
    # fig.tight_layout()

    # # number of processes vs memory for each instrument
    # plt.style.use('seaborn-bright')
    # fig, ax = plt.subplots(ncols=4, figsize=(12,6), sharey=True)
    # sns.barplot(data=acs, x='subarray', y='memory', ax=ax[0], order=[0,1])
    # sns.barplot(data=stis, x='subarray', y='memory', ax=ax[1], order=[0,1])
    # sns.barplot(data=wfc3, x='subarray', y='memory', ax=ax[2], order=[0,1])
    # sns.barplot(data=cos, x='subarray', y='memory', ax=ax[3], order=[0,1])
    # ax[0].set_title('ACS')
    # ax[1].set_title('STIS')
    # ax[2].set_title('WFC3')
    # ax[3].set_title('COS')
    # fig.tight_layout()

    # # BAR - 2 instruments
    # # number of processes vs memory for each instrument
    # plt.style.use('seaborn-bright')
    # fig, ax = plt.subplots(ncols=2, figsize=(12,6), sharey=True)
    # sns.barplot(data=acs, x='detector', y='memory', ax=ax[0], order=[0,1])
    # sns.barplot(data=wfc3, x='detector', y='memory', ax=ax[1], order=[0,1])
    # ax[0].set_title('ACS')
    # ax[1].set_title('WFC3')
    # fig.tight_layout()

    # # BAR - 3 Instruments
    # # number of processes vs memory for each instrument
    # plt.style.use('seaborn-bright')
    # fig, ax = plt.subplots(ncols=3, figsize=(12,6), sharey=True)
    # sns.barplot(data=acs, x='crsplit', y='memory', ax=ax[0], order=[0,1,2])
    # sns.barplot(data=stis, x='crsplit', y='memory', ax=ax[1], order=[0,1,2])
    # sns.barplot(data=wfc3, x='crsplit', y='memory', ax=ax[2], order=[0,1,2])
    # ax[0].set_title('ACS')
    # ax[1].set_title('STIS')
    # ax[2].set_title('WFC3')
    # fig.tight_layout()

    # Checking multicollinearity with a heatmap
    @staticmethod
    def multiplot(df, figsize=(15,15), color=None, instr=None, rename=True):
        if instr is None:
            if color is None:
                color="Blues"
            df2 = df.drop(columns=['instr_enc', 'instr', 'n_files', 'total_mb'], axis=1, inplace=False)
            rename_cols = {'drizcorr':'driz', 'pctecorr':'pcte', 'crsplit':'cr', 'subarray':'sub', 
                        'detector':'det','n_proc':'proc', 'wallclock':'wc','memory':'mem', 'mem_bin': 'bin'}
            
        elif instr == 'acs':
            color="Blues"
            acs = df.loc[df['instr'] == 'acs']
            df2 = acs.drop(columns=['n_files', 'total_mb', 'instr_enc'], axis=1, inplace=False)
            rename_cols = {'drizcorr':'driz', 'pctecorr':'pcte', 'crsplit':'cr', 'subarray':'sub', 'detector':'det', 'wallclock':'wc','memory':'mem', 'n_proc':'proc','mem_bin':'bin'})
            
        elif instr == 'cos':
            color="Greens"
            cos = df.loc[df['instr'] == 'cos']
            df2 = cos.drop(columns=['n_files', 'total_mb', 'instr_enc', 'drizcorr',
                            'pctecorr', 'detector', 'crsplit'], axis=1, inplace=False)
            rename_cols={ 'subarray':'sub', 'wallclock':'wc', 'memory':'mem', 'n_proc':'proc','mem_bin':'bin'})
            
        elif instr == 'stis':
            color="Reds"
            stis = df.loc[df['instr'] == 'stis']
            df2 = stis.drop(columns=['n_files', 'total_mb', 'instr_enc', 'drizcorr',
                            'pctecorr', 'detector', 'mem_bin'], axis=1, inplace=False)
            rename_cols={'subarray':'sub', 'crsplit':'cr','wallclock':'wc', 'n_proc':'proc','memory':'mem'})

        elif instr == 'wfc3':
            color="plasma"
            wfc3 = df.loc[df['instr'] == 'wfc3']
            df2 = wfc3.drop(columns=['n_files', 'total_mb','instr_enc'])
            rename_cols={'drizcorr':'driz', 'pctecorr':'pcte', 'crsplit':'cr', 'subarray':'sub', 
                                    'detector':'det', 'wallclock':'wc', 'memory':'mem', 'n_proc':'proc',
                                    'mem_bin':'bin'})

        if rename is True:
            df2 = df2.rename(columns=rename_cols)

        corr = np.abs(df2.corr().round(3)) 

        fig, ax = plt.subplots(figsize=figsize)
        mask = np.zeros_like(corr, dtype=np.bool)
        idx = np.triu_indices_from(mask)
        mask[idx] = True
        #xticks = ['fits', 'img', 'raw','mem', 'sec', 'mb','cnt']
        sns.heatmap(np.abs(corr),
                    square=True, mask=mask, annot=True, cmap=color, ax=ax)#,
                    #xticklabels=xticks, yticklabels=xticks)
        
        ax.set_ylim(len(corr), -.5, .5)

        return fig, ax

# EVALUATE ************
    @staticmethod
    def build_classifier(input_shape=9, layers=[18, 32, 64, 32, 18, 9],
                        input_name="hst_jobs", 
                        output_name="memory_pred"):
        model = Sequential()
        # visible layer
        inputs = keras.Input(shape=(input_shape,), name=input_name)
        # hidden layers
        x = keras.layers.Dense(layers[0], activation='relu', name=f'1_dense{layers[0]}')(inputs)
        for i, layer in enumerate(layers[1:]):
            i+=1
            x = keras.layers.Dense(layer, activation="relu", name=f"{i+1}_dense{layer}")(x)
        # output layer
        outputs = keras.layers.Dense(4, activation="softmax", name=f"output_{output_name}")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="sequential_mlp")
        model.compile(loss="categorical_crossentropy",
                        optimizer='adam', metrics=["accuracy"])
        return model

    @staticmethod
    def build_regressor(input_shape=9, layers=[18, 32, 64, 128, 256, 128, 64, 32, 18, 9], input_name="hst_jobs", output_name="wallclock_reg"):
        if layers is None:
            layers=[18, 32, 64, 32, 18, 9]
        model = Sequential()
        # visible layer
        inputs = keras.Input(shape=(input_shape,), name=input_name)
        # hidden layers
        x = keras.layers.Dense(layers[0], activation='relu', name='dense_1')(inputs)
        for i, layer in enumerate(layers[1:]):
            i+=1
            x = keras.layers.Dense(layer, activation="relu", name=f"dense_{i+1}")(x)
        # output layer
        outputs = keras.layers.Dense(1, name=output_name)(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="sequential_mlp")
        model.compile(loss="mean_squared_error",
                        optimizer='adam', metrics=["accuracy"])
        return model

    @staticmethod
    def fit(model, X_train, y_train, X_test, y_test, 
            verbose=1, epochs=200, batch_size=10, callbacks=None):
        validation_data = (X_test, y_test)
        start = dt.datetime.now(tz=tz.get_localzone()).strftime('%m/%d/%Y - %I:%M:%S %p')
        print("TRAINING STARTED: ", start)
        history = model.fit(X_train, y_train, batch_size=batch_size, 
                            validation_data=validation_data, verbose=verbose, 
                            epochs=epochs, callbacks=callbacks)
        end = dt.datetime.now(tz=tz.get_localzone()).strftime('%m/%d/%Y - %I:%M:%S %p')
        print("TRAINING COMPLETE: ", end)
        model.summary()
        return history

    @staticmethod
    def get_scores(model, X_train, X_test, y_train, y_test, verbose=False):
        train_scores = model.evaluate(X_train, y_train, verbose=2)
        test_scores = model.evaluate(X_test, y_test, verbose=2)
        if verbose:
            print("Train accuracy:", np.round(train_scores[1],2))
            print("Test accuracy:", np.round(test_scores[1],2))
        print("\nTrain loss:", np.round(train_scores[0],2))
        print("\nTest loss:", np.round(test_scores[0],2))

        return test_scores

    @staticmethod
    def predict_clf(model, X_train, X_test, y_train, y_test, verbose=False, proba=False):
        prob = model.predict(X_test)
        preds = np.argmax(prob, axis=-1)
        if verbose is True:
            print("\Probabilities: ", prob)
            print("\nPredictions: ", preds)
        if proba is True:
            return proba
        else:
            return preds

    @staticmethod
    def get_preds(X,y,model=None,verbose=False):
        if model is None:
            model=model
        # class predictions 
        y_true = np.argmax(y, axis=-1) + 1
        y_pred = np.argmax(model.predict(X), axis=-1)
        preds = pd.Series(y_pred).value_counts(normalize=False)
        
        if verbose:
            print(f"y_pred:\n {preds}")
            print("\n")
        return y_true, y_pred

    @staticmethod
    def fusion_matrix(matrix, classes=None, normalize=True, title='Fusion Matrix', cmap='Blues',
        print_raw=False): 
        """
        FUSION MATRIX
        -------------    
        matrix: can pass in matrix or a tuple (ytrue,ypred) to create on the fly 
        classes: class names for target variables
        """


        from sklearn import metrics                       
        from sklearn.metrics import confusion_matrix 
        import itertools
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        
        # make matrix if tuple passed to matrix:
        if isinstance(matrix, tuple):
            y_true = matrix[0].copy()
            y_pred = matrix[1].copy()
            
            if y_true.ndim>1:
                y_true = y_true.argmax(axis=1)
            if y_pred.ndim>1:
                y_pred = y_pred.argmax(axis=1)
            fusion = metrics.confusion_matrix(y_true, y_pred)
        else:
            fusion = matrix
        
        # INTEGER LABELS
        if classes is None:
            classes=set(matrix[0])
            #classes=list(range(len(matrix)))

        #NORMALIZING
        # Check if normalize is set to True
        # If so, normalize the raw fusion matrix before visualizing
        if normalize:
            fusion = fusion.astype('float') / fusion.sum(axis=1)[:, np.newaxis]
            thresh = 0.5
            fmt='.2f'
        else:
            fmt='d'
            thresh = fusion.max() / 2.
        
        # PLOT
        fig, ax = plt.subplots(figsize=(10,10))
        plt.imshow(fusion, cmap=cmap, aspect='equal')
        
        # Add title and axis labels 
        plt.title(title) 
        plt.ylabel('TRUE') 
        plt.xlabel('PRED')
        
        # Add appropriate axis scales
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        #ax.set_ylim(len(fusion), -.5,.5) ## <-- This was messing up the plots!
        
        # Text formatting
        fmt = '.2f' if normalize else 'd'
        # Add labels to each cell
        #thresh = fusion.max() / 2.
        # iterate thru matrix and append labels  
        for i, j in itertools.product(range(fusion.shape[0]), range(fusion.shape[1])):
            plt.text(j, i, format(fusion[i, j], fmt),
                    horizontalalignment='center',
                    color='white' if fusion[i, j] > thresh else 'black',
                    size=14, weight='bold')
        
        # Add a legend
        plt.colorbar()
        plt.show() 
        return fusion, fig

    @staticmethod
    def keras_history(history, figsize=(10,4)):
        """
        side by side sublots of training val accuracy and loss (left and right respectively)
        """
        
        import matplotlib.pyplot as plt
        
        fig,axes=plt.subplots(ncols=2,figsize=(15,6))
        axes = axes.flatten()

        ax = axes[0]
        ax.plot(history.history['accuracy'])
        ax.plot(history.history['val_accuracy'])
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Test'], loc='upper left')

        ax = axes[1]
        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        ax.set_title('Model Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    @staticmethod
    def get_model(model_path):
        model = keras.models.load_model(model_path)
        return model

    @staticmethod
    def classifier(model, data):
        pred = np.argmax(model.predict(data), axis=-1)
        return pred

    @staticmethod
    def save_model(model, version="0001", name='Sequential', save=True, weights=True):
        '''The model architecture, and training configuration (including the optimizer, losses, and metrics) 
        are stored in saved_model.pb. The weights are saved in the variables/ directory.'''
        model_subfolder = './models'
        model_version = version
        model_name = name
        path = os.path.join(model_subfolder, model_name)
        model_path = os.path.join(path, model_version)
        weights_path = f"{model_subfolder}/{model_name}/weights/ckpt_{model_version}"
        if save is True:
            model.save(model_path)
        if weights is True:
            model.save_weights(weights_path)
            #model.save_weights(weights)
        for root, dirs, files in os.walk(path):
                indent = '    ' * root.count(os.sep)
                print('{}{}/'.format(indent, os.path.basename(root)))
                for filename in files:
                    print('{}{}'.format(indent + '    ', filename))
        return model_path

