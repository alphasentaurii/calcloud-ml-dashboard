import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
# import load_data

# df = load_data.get_single_dataset('data/hst_data.csv')

# acs = df.groupby('instr').get_group(0)
# cos = df.groupby('instr').get_group(1)
# stis = df.groupby('instr').get_group(2)
# wfc3 = df.groupby('instr').get_group(3)

def df_by_instr(df):
    acs = df[df['instr'] == 0]
    cos = df[df['instr'] == 1]
    stis = df[df['instr'] == 2]
    wfc3 = df[df['instr'] == 3]
    instr_dict = {
        'acs': [acs, '#119dff'], 
        'wfc3': [wfc3, 'salmon'], 
        'cos': [cos, '#66c2a5'], 
        'stis': [stis, 'fuchsia']
        }
    return instr_dict


def make_continuous_figs(acs, cos, stis, wfc3, vars):
    continuous_figs = []

    for v in vars:
        data = [
            go.Box(
                y=acs[v],
                name='acs'
            ),
            go.Box(
                y=cos[v],
                name='cos'
            ),
            go.Box(
                y=stis[v],
                name='stis'
            ),
            go.Box(
                y=wfc3[v],
                name='wfc3'
            )
        ]
        layout = go.Layout(
            title = f'{v} by instrument',
            hovermode='closest',
            paper_bgcolor='#242a44',
            plot_bgcolor='#242a44',
            font={'color': '#ffffff'},
        )
        fig = go.Figure(data=data, layout=layout)
        continuous_figs.append(fig)
    return continuous_figs  


def make_scatter_figs(instr_dict, xaxis_name, yaxis_name):
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
# m, b = np.polyfit(x, y, 1) #slope, intercept
# plt.plot(x, m*x + b, 'k--'); # best fit line

# def make_scatter_plot(df):
#     instr_colors = i_colors()
#     data = go.Scatter(
#             x=df[xaxis_name],
#             y=df[yaxis_name],
#             text=df['ipst'],
#             mode='markers',
#             opacity=0.7,
#             marker={'size': 15, 'color': instr_colors[instrument]},
#             name=instrument
#         )
#     layout = go.Layout(
#         title='Loss',
#         paper_bgcolor='#242a44',
#         plot_bgcolor='#242a44',
#         font={'color': '#ffffff'},
#     )
#     fig = go.Figure(data=data, layout=layout)

# mean_memory = df.groupby("dtype").memory.mean()

# print(mean_memory)

# #  barplot :
# sns.barplot(
#   data=df,
#   x="dtype",
#   y="memory",
# )
# plt.show()



# app = dash.Dash()

# app.layout = html.Div([
#     html.Div(children=[
#         html.Div([
#             dcc.Dropdown(
#                 id='xaxis',
#                 options=[{'label': i, 'value': i} for i in features],
#                 value='n_files'
#             )
#         ],
#         style={'width': '20%', 'display': 'inline-block'}),

#         html.Div([
#             dcc.Dropdown(
#                 id='yaxis',
#                 options=[{'label': i, 'value': i} for i in features],
#                 value='memory'
#             )
#         ],
#         style={'width': '20%', 'display': 'inline-block'}),

#         html.Div([
#             dcc.RadioItems(
#                 id='version',
#                 options=[{'label': i, 'value': i} for i in versions],
#                 value='acs'
#                 )
#             ],
#         style={'width': '28%', 'float': 'right', 'display': 'inline-block'})
#     ], style={'margin': 10, 'padding': 25}),

#     dcc.Graph(
#         id='feature-graphic'
#         )
#     ], style={'backgroundColor': '#242a44', 'color':'white', 'padding':20, 'display': 'inline-block', 'width': '85%'})

# @app.callback(
#     Output('feature-graphic', 'figure'),
#     [Input('xaxis', 'value'),
#      Input('yaxis', 'value'),
#      Input('instrument', 'value')])

# def update_graph(xaxis_name, yaxis_name, version):
#     version_df = df[df['version'] == version]
#     #filtered_df = version_df[version_df['instr_key'] == instrument]
#     #instr_colors = i_colors()
#     traces = []
#     for instrument in filtered_df['instr_key'].unique():
#         df_by_instr = filtered_df[filtered_df['instr_key'] == instrument]
#         traces.append(go.Scatter(
#             x=df_by_instr[xaxis_name],
#             y=df_by_instr[yaxis_name],
#             text=df_by_instr['ipst'],
#             mode='markers',
#             opacity=0.7,
#             marker={'size': 15, 'color': instr_colors[instrument]},
#             name=instrument
#         ))
#     return {
#         'data': traces,
#         'layout': go.Layout(
#             xaxis={'title': xaxis_name},
#             yaxis={'title': yaxis_name},
#             margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
#             hovermode='closest',
#             paper_bgcolor='#242a44',
#             plot_bgcolor='#242a44',
#             font={'color': '#ffffff'},
#         )
#     }


# if __name__ == '__main__':
#     app.run_server(debug=True, dev_tools_hot_reload=True)

# construct a dictionary of dropdown values for the model versions
# version_options = []
# for version in df['version'].unique():
#     version_options.append({'label':str(version),'value':version})

# app.layout = html.Div([
#     dcc.Graph(id='graph-with-slider'),
#     dcc.Dropdown(id='version-picker',options=version_options,value=df['version'].min())
# ])

# @app.callback(Output('graph-with-slider', 'figure'),
#               [Input('version-picker', 'value')])
# def update_figure(selected_version):
#     filtered_df = df[df['version'] == selected_version]
#     traces = []
#     for instrument in filtered_df['instr_key'].unique():
#         df_by_instr = filtered_df[filtered_df['instr_key'] == instrument]
#         traces.append(go.Scatter(
#             x=df_by_instr['mem_bin'],
#             y=df_by_instr['memory'],
#             text=df_by_instr['instr'],
#             mode='markers',
#             opacity=0.7,
#             marker={'size': 15},
#             name=instrument
#         ))

#     return {
#         'data': traces,
#         'layout': go.Layout(
#             xaxis={'type': 'log', 'title': 'Memory Usage (GB)'},
#             yaxis={'title': 'Wallclock Time (seconds)'},
#             hovermode='closest'
#         )
#     }


#     if bestfit is True:
#         x = df[X]
#         y = df[Y]
#         m, b = np.polyfit(x, y, 1) #slope, intercept
#         plt.plot(x, m*x + b, 'k--'); # best fit line



# # Checking multicollinearity with a heatmap
# def multiplot(df, figsize=(20,20), color=None):
#     corr = np.abs(df.corr().round(3)) 
#     fig, ax = plt.subplots(figsize=figsize)
#     if color is None:
#         color="Blues"
#     mask = np.zeros_like(corr, dtype=np.bool)
#     idx = np.triu_indices_from(mask)
#     mask[idx] = True
#     #xticks = ['fits', 'img', 'raw','mem', 'sec', 'mb','cnt']
#     sns.heatmap(np.abs(corr),
#                 square=True, mask=mask, annot=True, cmap=color, ax=ax)#,
#                 #xticklabels=xticks, yticklabels=xticks)
#     ax.set_ylim(len(corr), -.5, .5)
#     return fig, ax

# def boxplots(df, x, name=None):
#     # iterate over categorical vars to build boxplots of distributions
#     # and visualize outliers
#     y = 'memory'
#     if name is None:
#         name = ''
#     plt.style.use('seaborn')
#     fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(11,11))
    
#     # Create keywords for .set_xticklabels()
#     tick_kwds = dict(horizontalalignment='right', 
#                       fontweight='light', 
#                       fontsize='x-large',   
#                       rotation=45)
#     sns.boxplot(data=df, x=x, y=y, ax=axes[0])
#     axes[0].set_xticklabels(axes[0].get_xticklabels(),**tick_kwds)
#     axes[0].set_xlabel(x)
#     axes[0].set_ylabel('RAM (mb)')
#     # Boxplot with outliers
#     axes[0].set_title(f'{name} {x} vs {y}: Boxplot with Outliers')
    
#     sns.boxplot(data=df, x=x, y=y, ax=axes[1], showfliers=False)
#     axes[1].set_xticklabels(axes[1].get_xticklabels(),**tick_kwds)
#     axes[1].set_xlabel(x)
#     axes[1].set_ylabel('RAM (mb)')
#     axes[1].set_title(f'{name} {x} vs {y}: Outliers Removed')
#     fig.tight_layout()



# def resid_plots(res, preds, target, save=False):
#     fig1 = plt.figure(figsize=(11,7))
#     ax1 = fig1.gca()
#     ax1 = sns.regplot(x=res, y=preds[:, 0], data=None, scatter=True, color='red')
#     ax1.legend(['residuals'])
#     ax1.title(f'{target} Residuals')
#     plt.show()
#     if save is True:
#         plt.savefig(f'residuals1_{target}.png')

#     fig2 = plt.figure(figsize=(11,7))
#     ax2 = fig2.gca()
#     ax2 = sns.regplot(x=res, y=preds[:, 0], fit_reg=True, color='red')
#     ax2 = sns.regplot(x=res, y=preds[:, 1], fit_reg=True, color='blue')
#     ax2.legend(['pred', 'true'])
#     ax2.title(f'{target} Residuals: pred v true')
#     plt.show()
#     if save is True:
#         plt.savefig(f'residuals2_{target}.png')

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
# fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(12,12))
# sns.distplot(df['drizcorr'], ax=ax[0][0])
# sns.distplot(df['pctecorr'], ax=ax[0][1])
# sns.distplot(df['crsplit'], ax=ax[1][0])
# sns.distplot(df['subarray'], ax=ax[1][1])
# sns.distplot(df['detector'], ax=ax[2][0])
# sns.distplot(df['memory'], ax=ax[2][1])

# plt.style.use('seaborn-bright')
# fig, ax = plt.subplots(ncols=2, figsize=(12,6), sharey=True)
# sns.barplot(data=acs, x='detector', y='memory', ax=ax[0], order=[0,1])
# sns.barplot(data=wfc3, x='detector', y='memory', ax=ax[1], order=[0,1])
# ax[0].set_title('ACS')
# ax[1].set_title('WFC3')
# fig.tight_layout()

# plt.style.use('seaborn-bright')
# fig, ax = plt.subplots(ncols=2, figsize=(12,6), sharey=True)
# sns.barplot(data=acs, x='pctecorr', y='memory', ax=ax[0], order=[0,1])
# sns.barplot(data=wfc3, x='pctecorr', y='memory', ax=ax[1], order=[0,1])
# ax[0].set_title('ACS')
# ax[1].set_title('WFC3')
# fig.tight_layout()


# plt.style.use('seaborn-bright')
# fig, ax = plt.subplots(ncols=3, figsize=(12,6), sharey=True)
# sns.barplot(data=acs, x='crsplit', y='memory', ax=ax[0], order=[0,1,2])
# sns.barplot(data=stis, x='crsplit', y='memory', ax=ax[1], order=[0,1,2])
# sns.barplot(data=wfc3, x='crsplit', y='memory', ax=ax[2], order=[0,1,2])
# ax[0].set_title('ACS')
# ax[1].set_title('STIS')
# ax[2].set_title('WFC3')
# fig.tight_layout()

# fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(12,5))
# sns.distplot(acs['crsplit'], ax=ax[0], label='acs')
# sns.distplot(stis['crsplit'], ax=ax[1], label='stis',color='red')
# sns.distplot(wfc3['crsplit'], ax=ax[2], label='wfc3', color='orange')

# ax[0].set_title('ACS')
# ax[1].set_title('STIS')
# ax[2].set_title('WFC3')



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



# fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15,5))
# sns.distplot(acs['subarray'], ax=ax[0], label='acs')
# sns.distplot(stis['subarray'], ax=ax[0], label='stis',color='red')
# sns.distplot(cos['subarray'], ax=ax[1], label='cos', color='lime')
# sns.distplot(wfc3['subarray'], ax=ax[2], label='wfc3', color='orange')

# ax[0].set_title('ACS, STIS')
# ax[1].set_title('COS')
# ax[2].set_title('WFC3')


# plt.style.use('seaborn-bright')
# fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12,12))
# sns.barplot(data=acs, x='dtype', y='memory', ax=ax[0][0], order=[1, 0])
# sns.barplot(data=cos, x='dtype', y='memory', ax=ax[0][1], order=[1, 0])
# sns.barplot(data=stis, x='dtype', y='memory', ax=ax[1][0], order=[1, 0])
# sns.barplot(data=wfc3, x='dtype', y='memory', ax=ax[1][1], order=[1, 0])
# ax[0][0].set_title('ACS')
# ax[0][1].set_title('COS')
# ax[1][0].set_title('STIS')
# ax[1][1].set_title('WFC3')
# fig.tight_layout()


# from scipy.stats import norm
# plt.figure(figsize=(11,6))
# sns.distplot(df.memory, fit=norm, label="memory")

# Distribution of memory usage skewed to the right (median is lower than mean).
# print(df.memory.mean())
# print(df.memory.median())

# # Kernel Density Estimates (distplots) for independent variables
# plt.style.use('seaborn-bright')
# fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12,12))
# sns.distplot(acs['memory'], fit=norm, ax=ax[0][0], label='acs')
# sns.distplot(cos['memory'], fit=norm, ax=ax[0][1], color='lime')
# sns.distplot(stis['memory'], fit=norm, ax=ax[1][0], color='red')
# sns.distplot(wfc3['memory'], fit=norm, ax=ax[1][1], color='blue')
# fig.tight_layout()


# # wallclock time vs memory for each instrument
# plt.style.use('seaborn-bright')
# fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12,12))
# sns.barplot(data=acs, x='mem_bin', y='wallclock', ax=ax[0][0], order=[0,1,2,3])
# sns.barplot(data=cos, x='mem_bin', y='wallclock', ax=ax[0][1], order=[0,1,2,3])
# sns.barplot(data=stis, x='mem_bin', y='wallclock', ax=ax[1][0], order=[0,1,2,3])
# sns.barplot(data=wfc3, x='mem_bin', y='wallclock', ax=ax[1][1], order=[0,1,2,3])
# ax[0][0].set_title('ACS')
# ax[0][1].set_title('COS')
# ax[1][0].set_title('STIS')
# ax[1][1].set_title('WFC3')
# fig.tight_layout()

# from keras import utils
# keras.utils.plot_model(clf)