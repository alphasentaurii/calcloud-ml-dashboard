import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np

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

    #return acs_scatter, cos_scatter, stis_scatter, wfc3_scatter 


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



# def hubble_scatter(df, X, Y='memory', instruments=None, bestfit=False):
#     fig = plt.figure(figsize=(11,7))
#     ax = fig.gca()

#     if instruments is None:
#         ax.scatter(df[X], df[Y])
#         ax.set_xlabel(X)
#         ax.set_ylabel(Y)
#         ax.set_title(f"{X} vs. {Y}")
#     else:
#         cols = list(df.columns)
#         if 'instr' in cols:
#             instr_col = 'instr'
#         else:
#             instr_col = 'instr_enc'

#         for i in instruments:
#             if i == 'acs':
#                 e = 0
#                 c = 'blue'
#             elif i == 'cos':
#                 e = 1
#                 c='lime'
#             elif i == 'stis':
#                 e = 2
#                 c='red'
#             elif i == 'wfc3':
#                 e = 3
#                 c = 'orange'
#             if instr_col == 'instr':
#                 ax.scatter((df[X].loc[df[instr_col] == i]), (df[Y].loc[df[instr_col] == i]), c=c, alpha=0.7)
#             else:
#                 ax.scatter((df[X].loc[df[instr_col] == e]), (df[Y].loc[df[instr_col] == e]), c=c, alpha=0.7)
 
#             ax.set_xlabel(X)
#             ax.set_ylabel(Y)
#             ax.set_title(f"{X} vs. {Y}: {[i for i in instruments]}")
#             if len(instruments) > 1:
#                 ax.legend([i for i in instruments])
            
#     if bestfit is True:
#         x = df[X]
#         y = df[Y]
#         m, b = np.polyfit(x, y, 1) #slope, intercept
#         plt.plot(x, m*x + b, 'k--'); # best fit line
#     else:
#         plt.show();



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