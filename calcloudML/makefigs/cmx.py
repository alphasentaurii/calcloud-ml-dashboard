import numpy as np
import pickle
import plotly.figure_factory as ff
from plotly import subplots

# CMS = np.array([
#             [[ 4778,    81,     2,     1],
#             [    3,   943,     1,     0],
#             [    3,     0,    20,     1],
#             [    0,     0,     1,     5]],

#             [[ 8636,   549,     5,     1],
#             [    5,  2590,     0,     0],
#             [    3,     4,    23,     0],
#             [    2,     0,     2,     2]],

#             [[14567,   915,     3,     2],
#             [    8,  3608,     2,     0],
#             [    9,     1,    24,     3],
#             [    1,     0,     1,     2]]])

# def get_confusion_matrix(cmx_filepath):
#     if cmx_filepath:
#         # v1_2021-05-13/results/cmx
#         cmx = pickle.load(open(cmx_filepath, "rb"))
#     # else:
#     #     cmx = CMS
#     return cmx
def normalize_cmx(cmx):
    cmx_norm = cmx.astype("float") / cmx.sum(axis=1)[:, np.newaxis]
    cmx_norm = np.round(cmx_norm, 3)
    return cmx_norm


def import_cmx(results, cmx_type):
    cmx = []
    for v in results.keys():
        matrix = results[v]["mem_bin"]["matrix"]
        if cmx_type == "normalized":
            matrix = normalize_cmx(matrix)
        cmx.append(matrix)
    return cmx


def make_cmx_figure(results, cmx_type):
    cmx = import_cmx(results, cmx_type)
    if cmx_type == "normalized":
        zmin = 0.0
        zmax = 1.0
    else:
        zmin = 0
        zmax = 100
    # cmx_norm = normalize_cmx(cmx)
    classes = ["2GB", "8GB", "16GB", "64GB"]
    x = classes
    y = x[::-1].copy()
    z1 = cmx[0][::-1]
    z2 = cmx[1][::-1]
    z3 = cmx[2][::-1]
    z1_text = [[str(y) for y in x] for x in z1]
    z2_text = [[str(y) for y in x] for x in z2]
    z3_text = [[str(y) for y in x] for x in z3]

    fig = subplots.make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("v1", "v2", "v3"),
        shared_yaxes=False,
        x_title="Predicted",
        y_title="Actual",
    )
    fig.update_layout(
        title_text="Confusion Matrix",
        paper_bgcolor="#242a44",
        plot_bgcolor="#242a44",
        font={"color": "#ffffff"},
    )
    # make traces
    fig1 = ff.create_annotated_heatmap(
        z=z1,
        x=x,
        y=y,
        annotation_text=z1_text,
        colorscale="Blues",
        zmin=zmin,
        zmax=zmax,
    )
    fig2 = ff.create_annotated_heatmap(
        z=z2,
        x=x,
        y=y,
        annotation_text=z2_text,
        colorscale="Blues",
        zmin=zmin,
        zmax=zmax,
    )
    fig3 = ff.create_annotated_heatmap(
        z=z3,
        x=x,
        y=y,
        annotation_text=z3_text,
        colorscale="Blues",
        zmin=zmin,
        zmax=zmax,
    )

    fig.add_trace(fig1.data[0], 1, 1)
    fig.add_trace(fig2.data[0], 1, 2)
    fig.add_trace(fig3.data[0], 1, 3)

    annot1 = list(fig1.layout.annotations)
    annot2 = list(fig2.layout.annotations)
    annot3 = list(fig3.layout.annotations)
    for k in range(len(annot2)):
        annot2[k]["xref"] = "x2"
        annot2[k]["yref"] = "y2"
    for k in range(len(annot3)):
        annot3[k]["xref"] = "x3"
        annot3[k]["yref"] = "y3"

    new_annotations = []
    annos = [annot1, annot2, annot3]
    for a in annos:
        new_annotations.extend(a)

    # add colorbar
    fig["data"][0]["showscale"] = True
    # annotation values for each square
    for anno in new_annotations:
        fig.add_annotation(anno)
    return fig
