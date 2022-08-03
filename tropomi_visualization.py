# import the necessary dependencies
import warnings

import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html, Input, Output
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


# update the 'tropomi_methane' variable with a csv file and read in the data as a DataFrame
tropomi_methane = ''
df = pd.read_csv(tropomi_methane, index_col='time_utc', parse_dates=True)
df.index = df.index.time

days = list(df['day'].sort_values().unique())
threshold = df['ppm'].mean() + (2 * df['ppm'].std())

# dashboard application
app = dash.Dash()

# dashboard layout (Dash components: dropdown menu and visualizations)
app.layout = html.Div([

    # header
    html.H1('TROPOMI Methane Analysis, Mainland USA', style={'text-align': 'center'}),

    # input Dash component: dropdown menu
    dcc.Dropdown(id='dropdown_menu',
                 options=days,
                 placeholder='Select a day',
                 style={'width': '40%'}
                 ),

    # output Dash component: visualizations (see the 'update_graph' function below)
    dcc.Graph(id='visualizations', figure={})
])


# connect each Dash component's id with its corresponding property
@app.callback(
    Output(component_id='visualizations', component_property='figure'),
    Input(component_id='dropdown_menu', component_property='value')
)
def update_graph(date):
    df_day = df[df['day'] == date].copy(deep=True)
    df_abnormal_methane = df_day[df_day['ppm'] > threshold].copy(deep=True)
    df_analysis = pd.DataFrame(
        np.round(df_abnormal_methane.groupby('state')['ppm'].sum() / df_abnormal_methane['ppm'].sum(),
                 6) * 100).reset_index().sort_values('ppm')
    df_analysis.columns = ['state', 'abnormal_methane_percentage']

    # instantiate a Plotly 'make_subplots()' object, specify its input parameters, and assign it to the variable 'fig'
    fig = make_subplots(rows=1, cols=2, specs=[[dict(type='mapbox'), dict(type='bar')]],
                        subplot_titles=('Atmospheric Methane Concentrations, Mainland USA',
                                        'Percentage of Abnormal Methane Concentrations per State'),
                        column_widths=[0.7, 0.3], horizontal_spacing=0.15)

    # interactive map of the mainland USA's daily atmospheric methane concentrations
    fig.add_trace(go.Scattermapbox(name='Map', lon=df_day['longitude'], lat=df_day['latitude'], mode='markers',
                                   marker=go.scattermapbox.Marker(color=df_day['ppm'], cmin=1.8, cmax=2,
                                                                  cmid=df_day['ppm'].median(), colorscale='YlOrRd',
                                                                  opacity=0.25, showscale=True,
                                                                  colorbar=dict(len=1, thickness=30, x=0.6, y=0.5,
                                                                                orientation='v',
                                                                                title='Atmospheric Methane (ppm)',
                                                                                titleside='right', ticks='outside',
                                                                                ticklen=5)),
                                   hoverinfo='text', customdata=np.stack((df_day['state'], np.round(df_day['ppm'], 4),
                                                                          df_day['latitude'], df_day['longitude'],
                                                                          pd.Series(df_day.index),
                                                                          df_day['bin_number'])).T,
                                   hovertemplate='<extra></extra><b>State</b>: %{customdata[0]}' +
                                                 '<br><b>Atmospheric Methane (ppm)</b>: %{customdata[1]}' +
                                                 '<br><b>Latitude</b>: %{customdata[2]}' +
                                                 '<br><b>Longitude</b>: %{customdata[3]}' +
                                                 '<br><b>Coordinated Universal Time (UTC)</b>: %{customdata[4]}' +
                                                 '<br><b>Bin Number</b>: %{customdata[5]}'),
                  row=1, col=1)

    # interactive horizontal bar chart showing the percentage of 'abnormal' methane concentrations per state
    fig.add_trace(go.Bar(name='Bar chart', x=df_analysis['abnormal_methane_percentage'], y=df_analysis['state'],
                         marker=go.bar.Marker(color='red'), hoverinfo='text',
                         hovertemplate='<extra></extra><b>% of Abnormal Methane</b>: %{x}', orientation='h'),
                  row=1, col=2)

    # update the 'fig' variable's layout
    fig.update_layout(height=750, width=2000,
                      mapbox_style='open-street-map',
                      mapbox=dict(center=dict(lon=-98, lat=38), pitch=0, bearing=0, zoom=3.5), hovermode='closest')

    # output the 'fig' variable's visualizations
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)