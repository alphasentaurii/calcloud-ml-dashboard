# import plotly.graph_objs as go
# from plotly import subplots
# from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_score, recall_score, f1_score #plot_precision_recall_curve
# import numpy as np

# # AUC_ROC Plots

# def classifier(model, data):
#     """Returns class prediction"""
#     pred_proba = model.predict(data)
#     pred = int(np.argmax(pred_proba, axis=-1))
#     return pred, pred_proba

# # def precision_recall(model):

# #     p = tp / (tp + fp)
# #     r = tp / (tp + fn)


import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import load_data




def get_pred_proba(results, version):
    y = results[version]['mem_bin']['y_true']
    y_pred = results[version]['mem_bin']['y_pred']
    y_scores = results[version]['mem_bin']['proba']
    return y, y_pred, y_scores

def make_dummies(y):
    y_onehot = pd.get_dummies(y, prefix='bin')
    return y_onehot


def make_roc_curve(y, y_onehot, y_scores):
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    for i in range(y_scores.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    return fig

# if __name__ == ('__main__'):
#     timestamps = [1620351000, 1620740441, 1620929899, 1621096666]
#     versions = ["v0", "v1", "v2", "v3"]
#     meta = load_data.make_meta(timestamps, versions)
#     results = load_data.make_res(meta, versions)
#     y, y_pred, y_scores = get_pred_proba(results, 'v3')
#     y_onehot = make_dummies(y)
#     fig = make_roc_curve(y, y_onehot, y_scores)
#     fig.show()






# np.random.seed(0)

# # Artificially add noise to make task harder
# df = px.data.iris()
# samples = df.species.sample(n=50, random_state=0)
# np.random.shuffle(samples.values)
# df.loc[samples.index, 'species'] = samples.values

# # Define the inputs and outputs
# X = df.drop(columns=['species', 'species_id'])
# y = df['species']

# # Fit the model
# model = LogisticRegression(max_iter=200)
# model.fit(X, y)
# y_scores = model.predict_proba(X)
# array([[0.84, 0.11, 0.05],
# One hot encode the labels in order to plot them
# y_onehot = pd.get_dummies(y, columns=model.classes_)



# Create an empty figure, and iteratively add new lines
# every time we compute a new class






# def auc_roc_plots(y_true, y_pred):
#     n_classes = np.unique(y_true)
#     test_names = ['0', '1', '2', '3']
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    
#     n_epochs = list(range(len(y_true)))
#     data = [go.Scatter(
#         x = n_epochs,
#         y = acc_train,
#         name='Training Accuracy',
#         marker=dict(color='#119dff')
#         ),
#         go.Scatter(
#         x = n_epochs,
#         y = acc_test,
#         name = 'Test Accuracy',
#         marker=dict(color='#66c2a5')
#         )]
#     layout = go.Layout(
#         title='Accuracy',
#         xaxis={'title': 'n_epochs'},
#         yaxis={'title': 'score'},
#         paper_bgcolor='#242a44',
#         plot_bgcolor='#242a44',
#         font={'color': '#ffffff'},
#     )
#     fig = go.Figure(data=data, layout=layout)
#     return fig

#     plt.figure()
#     plt.plot(
#         fpr["micro"], 
#         tpr["micro"], 
#         label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
#     for i in range(n_classes):
#         plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area ={1:0.2f})'.format(test_names[i], roc_auc[i]))
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     roc_mean = np.mean(np.fromiter(roc_auc.values(), dtype=float))
#     plt.title('ROC curve for Memory Classifier '+str(epochs) +' iter Tensorflow (area = %(0:0.2f))'.format(roc_mean))
#     plt.legend(loc="lower right")
#     plt.show()



# import plotly.express as px
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import precision_recall_curve, auc
# from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=500, random_state=0)

# model = LogisticRegression()
# model.fit(X, y)
# y_score = model.predict_proba(X)[:, 1]

# precision, recall, thresholds = precision_recall_curve(y, y_score)

# fig = px.area(
#     x=recall, y=precision,
#     title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
#     labels=dict(x='Recall', y='Precision'),
#     width=700, height=500
# )
# fig.add_shape(
#     type='line', line=dict(dash='dash'),
#     x0=0, x1=1, y0=1, y1=0
# )
# fig.update_yaxes(scaleanchor="x", scaleratio=1)
# fig.update_xaxes(constrain='domain')

# fig.show()