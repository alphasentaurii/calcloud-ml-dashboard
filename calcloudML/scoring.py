import plotly.graph_objs as go
from plotly import subplots
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score #plot_precision_recall_curve
import numpy as np


def accuracy_bars(df):
    acc_train = df.loc['train_acc'].values
    acc_test = df.loc['test_acc'].values
    data = [go.Bar(
        x = list(range(len(acc_train))),
        y = acc_train,
        name='Training Accuracy',
        marker=dict(color='#119dff')
        ),
        go.Bar(
        x = list(range(len(acc_test))),
        y = acc_test,
        name = 'Test Accuracy',
        marker=dict(color='#66c2a5')
        )]
    layout = go.Layout(
        title='Accuracy',
        xaxis={'title': 'training iteration'},
        yaxis={'title': 'score'},
        paper_bgcolor='#242a44',
        plot_bgcolor='#242a44',
        font={'color': '#ffffff'},
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def loss_bars(df):
    loss_train = df.loc['train_loss'].values
    loss_test = df.loc['test_loss'].values
    data = [go.Bar(
        x = list(range(len(loss_train))),
        y = loss_train,
        name='Training Loss',
        marker=dict(color='salmon')
        ),
        go.Bar(
        x = list(range(len(loss_test))),
        y = loss_test,
        name = 'Test Loss',
        marker=dict(color='peachpuff')
        )]
    layout = go.Layout(
        title='Loss',
        xaxis={'title': 'training iteration'},
        yaxis={'title': 'score'},
        paper_bgcolor='#242a44',
        plot_bgcolor='#242a44',
        font={'color': '#ffffff'},
    )
    fig = go.Figure(data=data, layout=layout,)
    return fig


def acc_loss_subplots(acc_fig, loss_fig):
    fig = subplots.make_subplots(rows=1, cols=2,
        subplot_titles=('Accuracy', 'Loss'),
        shared_yaxes = False,
        x_title='Training Iteration',
        y_title='Score',
    )
    fig.add_trace(acc_fig.data[0], 1, 1)
    fig.add_trace(acc_fig.data[1], 1, 1)
    fig.add_trace(loss_fig.data[0], 1, 2)
    fig.add_trace(loss_fig.data[1], 1, 2)
    fig.update_layout(
        title_text='Accuracy vs. Loss', 
        margin=dict(t=50, l=200),
        paper_bgcolor='#242a44',
        plot_bgcolor='#242a44',
        font={'color': '#ffffff',
        })
    return fig


def acc_loss_bars(df, subplots=False):
    acc_fig = accuracy_bars(df)
    loss_fig = loss_bars(df)
    if subplots is True:
        acc_loss_fig = acc_loss_subplots(acc_fig, loss_fig)
        return acc_loss_fig
    else:
        return acc_fig, loss_fig


# KERAS HISTORY

def keras_acc_plot(acc_train, acc_test):
    n_epochs = list(range(len(acc_train)))
    data = [go.Scatter(
        x = n_epochs,
        y = acc_train,
        name='Training Accuracy',
        marker=dict(color='#119dff')
        ),
        go.Scatter(
        x = n_epochs,
        y = acc_test,
        name = 'Test Accuracy',
        marker=dict(color='#66c2a5')
        )]
    layout = go.Layout(
        title='Accuracy',
        xaxis={'title': 'n_epochs'},
        yaxis={'title': 'score'},
        paper_bgcolor='#242a44',
        plot_bgcolor='#242a44',
        font={'color': '#ffffff'},
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def keras_loss_plot(loss_train, loss_test):
    n_epochs = list(range(len(loss_train)))
    data = [go.Scatter(
        x = n_epochs,
        y = loss_train,
        name='Training Loss',
        marker=dict(color='#119dff')
        ),
    go.Scatter(
        x = n_epochs,
        y = loss_test,
        name = 'Test Loss',
        marker=dict(color='#66c2a5')
        )]
    layout = go.Layout(
        title='Loss',
        xaxis={'title': 'n_epochs'},
        yaxis={'title': 'score'},
        paper_bgcolor='#242a44',
        plot_bgcolor='#242a44',
        font={'color': '#ffffff'},
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def keras_plots(history):
    acc_train, acc_test = history['accuracy'], history['val_accuracy']
    loss_train, loss_test = history['loss'], history['val_loss']
    keras_acc = keras_acc_plot(acc_train, acc_test)
    keras_loss = keras_loss_plot(loss_train, loss_test)
    return [keras_acc, keras_loss]


# AUC_ROC Plots
CMS = np.array([
            [[ 4778,    81,     2,     1],
            [    3,   943,     1,     0],
            [    3,     0,    20,     1],
            [    0,     0,     1,     5]],
            
            [[ 8636,   549,     5,     1],
            [    5,  2590,     0,     0],
            [    3,     4,    23,     0],
            [    2,     0,     2,     2]],
            
            [[14567,   915,     3,     2],
            [    8,  3608,     2,     0],
            [    9,     1,    24,     3],
            [    1,     0,     1,     2]]])

def precision_recall(CMS):
    cm = CMS[0]
    tp = cm[0,0]
    fp = cm[1:,0]
    fn = cm[0,1:].sum()
    p = tp / (tp + fp)
    r = tp / (tp + fn)

    


def auc_roc_plots(y_true, y_pred):
    n_classes = np.unique(y_true)
    test_names = ['0', '1', '2', '3']
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    
    n_epochs = list(range(len(y_true)))
    data = [go.Scatter(
        x = n_epochs,
        y = acc_train,
        name='Training Accuracy',
        marker=dict(color='#119dff')
        ),
        go.Scatter(
        x = n_epochs,
        y = acc_test,
        name = 'Test Accuracy',
        marker=dict(color='#66c2a5')
        )]
    layout = go.Layout(
        title='Accuracy',
        xaxis={'title': 'n_epochs'},
        yaxis={'title': 'score'},
        paper_bgcolor='#242a44',
        plot_bgcolor='#242a44',
        font={'color': '#ffffff'},
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

    plt.figure()
    plt.plot(
        fpr["micro"], 
        tpr["micro"], 
        label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area ={1:0.2f})'.format(test_names[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_mean = np.mean(np.fromiter(roc_auc.values(), dtype=float))
    plt.title('ROC curve for Memory Classifier '+str(epochs) +' iter Tensorflow (area = %(0:0.2f))'.format(roc_mean))
    plt.legend(loc="lower right")
    plt.show()


## GROUPED BAR CHART
# df = pd.read_csv("data/batch.csv", index_col=False)
# df.set_index('ipst', drop=True, inplace=True)
# # groups = df.groupby(['instr'])[['memory','mem_bin']]
# groups = df.groupby(['mem_bin'])['instr']
# bin0 = groups.get_group(0.0).value_counts()
# bin1 = groups.get_group(1.0).value_counts()
# bin2 = groups.get_group(2.0).value_counts()
# bin3 = groups.get_group(3.0).value_counts()
# trace1 = go.Bar(
#     x=bin0.index,
#     y=bin0,
#     name = '2G',
#     marker = dict(color='blue')
# )
# trace2 = go.Bar(
#     x=bin1.index,
#     y=bin1,
#     name = '8G',
#     marker=dict(color='orange')
# )
# trace3 = go.Bar(
#     x=bin2.index,
#     y=bin2,
#     name = '16G',
#     marker = dict(color='red')
# )
# trace4 = go.Bar(
#     x=bin3.index,
#     y=bin3,
#     name = '64G',
#     marker = dict(color='purple')
# )
# data = [trace1, trace2, trace3, trace4]
# layout = go.Layout(
#     title = 'Memory Bin Value Counts by Instrument'
# )
# fig5 = go.Figure(data=data, layout=layout)
# pyo.plot(fig5, filename='bar2.html')

