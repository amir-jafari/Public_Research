#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# Loading the dataset
abspath = "/Users/sayanpatra/Downloads"
path_lunch = "/combined_lunch_data.csv"
path_breakfast = "/combined_breakfast_data.csv"

# Define the data loading function
def load_data(path):
    df = pd.read_csv(abspath + path)
    return df

# Load breakfast data
df_breakfast = load_data(path_breakfast)
print("Breakfast Data Preview:")
print(df_breakfast.head())
print(df_breakfast.describe())

# Load lunch data
df_lunch = load_data(path_lunch)
print("\nLunch Data Preview:")
print(df_lunch.head())
print(df_lunch.describe())

# Combine both
df_all = pd.concat([df_lunch, df_breakfast], ignore_index=True)
print("\nCombined Data Preview:")
print(df_all.head())

# Data preparation
df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
df_all['Served_Non-Reimbursable_Cost'] = pd.to_numeric(
    df_all['Served_Non-Reimbursable'].str.replace('$', '').str.strip(), errors='coerce'
)
df_all['Left_Over_Total'] = pd.to_numeric(df_all['Left_Over_Total'], errors='coerce')
df_all['Left_Over_Percent_of_Offered'] = pd.to_numeric(df_all['Left_Over_Percent_of_Offered'], errors='coerce')
df_all['Left_Over_Cost'] = pd.to_numeric(df_all['Left_Over_Cost'], errors='coerce')
df_all['Production_Cost_Total'] = pd.to_numeric(df_all['Production_Cost_Total'], errors='coerce')

# Prepare melted data
df_reimb = df_all[['School_Name', 'Name', 'Date', 'Served_Reimbursable']].copy()
df_reimb['Category'] = 'Reimbursable'
df_reimb.rename(columns={'Served_Reimbursable': 'Value'}, inplace=True)

df_non_reimb = df_all[['School_Name', 'Name', 'Date', 'Served_Non-Reimbursable_Cost']].copy()
df_non_reimb['Category'] = 'Non-Reimbursable'
df_non_reimb.rename(columns={'Served_Non-Reimbursable_Cost': 'Value'}, inplace=True)

df_melted = pd.concat([df_reimb, df_non_reimb], ignore_index=True)
df_melted = df_melted.dropna(subset=['Value'])

# Aggregate
df_agg = df_melted.groupby(['School_Name', 'Name', 'Date', 'Category'])['Value'].sum().reset_index()
forecast_base = df_agg.copy()
forecast_base['Date'] = pd.to_datetime(forecast_base['Date'])
forecast_base = forecast_base.sort_values('Date')

# Daily item sales summed
forecast_grouped = forecast_base.groupby(['Name', 'Date'])['Value'].sum().reset_index()

# 7-day moving average forecast per item
forecast_grouped['7day_avg'] = forecast_grouped.groupby('Name')['Value'].transform(lambda x: x.rolling(7, min_periods=1).mean())

# Get latest available date
latest_date = forecast_grouped['Date'].max()
forecast_tomorrow = forecast_grouped[forecast_grouped['Date'] == latest_date][['Name', '7day_avg']].sort_values('7day_avg', ascending=False).head(5)
forecast_tomorrow.rename(columns={'7day_avg': 'Predicted_Sales'}, inplace=True)

# Initialize Dash
app = Dash(__name__)
app.title = "Interactive School Item Sales Dashboard"

app.layout = html.Div([
    html.H2("School Food Sales Dashboard"),

    html.Div([
        html.Label("Select Meal Type:"),
        dcc.Dropdown(
            id='meal-type-dropdown',
            options=[
                {'label': 'Breakfast', 'value': 'breakfast'},
                {'label': 'Lunch', 'value': 'lunch'}
            ],
            value='lunch',
            clearable=False,
            style={'width': '300px'}
        ),
        html.Label("Select Metric:"),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': 'Served Reimbursable', 'value': 'Served_Reimbursable'},
                {'label': 'Served Non-Reimbursable Cost', 'value': 'Served_Non-Reimbursable_Cost'},
                {'label': 'Left Over Total', 'value': 'Left_Over_Total'},
                {'label': 'Left Over % of Offered', 'value': 'Left_Over_Percent_of_Offered'},
                {'label': 'Left Over Cost', 'value': 'Left_Over_Cost'},
                {'label': 'Production Cost Total', 'value': 'Production_Cost_Total'}
            ],
            value='Served_Reimbursable',  # default
            clearable=False,
            style={'width': '300px'}
        ),

        html.Label("Select Chart Type:"),
        dcc.Dropdown(
            id='chart-type-dropdown',
            options=[
                {'label': 'Line Chart', 'value': 'line'},
                {'label': 'Bar Chart', 'value': 'bar'},
                {'label': 'Scatter Plot', 'value': 'scatter'},
                {'label': 'Area Chart', 'value': 'area'},
                {'label': 'Box Plot', 'value': 'box'},
                {'label': 'Pie Chart', 'value': 'pie'},
                {'label': 'Violin Plot', 'value': 'violin'},
                {'label': 'Histogram', 'value': 'histogram'},
                {'label': 'Funnel Chart', 'value': 'funnel'},
                {'label': 'Network Graph', 'value': 'network'}
            ],
            value='line',
            clearable=False,
            style={'width': '300px'}
        ),

        html.Label("Select Theme:"),
        dcc.Dropdown(
            id='theme-dropdown',
            options=[
                {'label': 'Plotly (Default)', 'value': 'plotly'},
                {'label': 'Plotly Dark', 'value': 'plotly_dark'},
                {'label': 'GGPlot2', 'value': 'ggplot2'},
                {'label': 'Seaborn', 'value': 'seaborn'},
                {'label': 'Simple White', 'value': 'simple_white'}
            ],
            value='plotly',
            clearable=False,
            style={'width': '300px'}
        ),

        html.Label("Select Item(s):"),
        dcc.Dropdown(
            id='item-dropdown',
            multi=True,
            style={'width': '500px'}
        ),

        html.Label("Select School(s):"),
        dcc.Dropdown(
            id='school-dropdown',
            multi=True,
            style={'width': '500px'}
        ),
    ], style={'columnCount': 2, 'marginBottom': '20px'}),

    html.Hr(),

    dcc.Graph(
        id='time-series-chart',
        config={'displayModeBar': True}
    ),

    html.H3("Recommended Items to Stock Tomorrow"),

    dcc.Graph(
        id='recommendation-graph',
        config={'displayModeBar': False}
    )
])

# Update item options
@app.callback(
    Output('item-dropdown', 'options'),
    Output('item-dropdown', 'value'),
    Input('meal-type-dropdown', 'value'),
    Input('metric-dropdown', 'value')
)
def update_items(meal_type, metric):
    # Filter your df accordingly
    if meal_type == 'breakfast':
        df_filtered = df_breakfast.copy()
    else:
        df_filtered = df_lunch.copy()

    items = df_filtered['Name'].unique()
    options = [{'label': i, 'value': i} for i in sorted(items)]
    return options, None

# Update school options
@app.callback(
    Output('school-dropdown', 'options'),
    Output('school-dropdown', 'value'),
    Input('metric-dropdown', 'value'),
    Input('item-dropdown', 'value')
)
def update_schools(metric, selected_items):
    filtered = df_agg.copy()
    if selected_items:
        filtered = filtered[filtered['Name'].isin(selected_items)]
    schools = filtered['School_Name'].unique()
    options = [{'label': school, 'value': school} for school in sorted(schools)]
    top_schools = filtered.groupby('School_Name')['Value'].sum().nlargest(3).index.tolist()
    return options, None

# Update graph
@app.callback(
    Output('time-series-chart', 'figure'),
    Input('meal-type-dropdown', 'value'),
    Input('metric-dropdown', 'value'),
    Input('item-dropdown', 'value'),
    Input('school-dropdown', 'value'),
    Input('chart-type-dropdown', 'value'),
    Input('theme-dropdown', 'value')
)
def update_graph(meal_type, metric, selected_items, selected_schools, chart_type, theme):
    import networkx as nx
    import plotly.graph_objects as go

    # Select dataset
    df = df_breakfast if meal_type == 'breakfast' else df_lunch

    # Prepare 'Value' based on metric
    if metric == 'Served_Reimbursable':
        df['Value'] = pd.to_numeric(df['Served_Reimbursable'], errors='coerce')
    elif metric == 'Served_Non-Reimbursable_Cost':
        df['Value'] = pd.to_numeric(df['Served_Non-Reimbursable'].str.replace('$','').str.strip(), errors='coerce')
    elif metric == 'Left_Over_Total':
        df['Value'] = pd.to_numeric(df['Left_Over_Total'], errors='coerce')
    elif metric == 'Left_Over_Percent_of_Offered':
        df['Value'] = pd.to_numeric(df['Left_Over_Percent_of_Offered'], errors='coerce')
    elif metric == 'Left_Over_Cost':
        df['Value'] = pd.to_numeric(df['Left_Over_Cost'], errors='coerce')
    elif metric == 'Production_Cost_Total':
        df['Value'] = pd.to_numeric(df['Production_Cost_Total'], errors='coerce')
    else:
        df['Value'] = pd.to_numeric(df['Served_Reimbursable'], errors='coerce')

    filtered = df.dropna(subset=['Value'])

    if selected_items:
        filtered = filtered[filtered['Name'].isin(selected_items)]
    if selected_schools:
        filtered = filtered[filtered['School_Name'].isin(selected_schools)]

    if filtered.empty or not selected_items or not selected_schools:
        fig = px.scatter()
        fig.update_layout(
            
            title="Welcome! Please select Metric, Item(s), and School(s) to see data.",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text="Select filters to get started!",
                x=0.5, y=0.5,
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(size=20)
            )],
            template=theme
        )
        return fig

    plot_map = {
        'line': px.line,
        'bar': px.bar,
        'scatter': px.scatter,
        'area': px.area,
        'box': px.box,
        'pie': px.pie,
        'violin': px.violin,
        'histogram': px.histogram,
        'funnel': px.funnel,
    }

    if chart_type == 'pie':
        fig = px.pie(
            filtered,
            names='Name',       # categories for pie slices
            values='Value',     # numeric values for slices
            title=f"{metric.replace('_', ' ')} - Pie Chart",
            template=theme
        )

    elif chart_type == 'network':
        G = nx.Graph()
        nodes = filtered['School_Name'].unique().tolist() + filtered['Name'].unique().tolist()
        for node in nodes:
            G.add_node(node)

        for _, row in filtered.iterrows():
            G.add_edge(row['School_Name'], row['Name'])

        pos = nx.spring_layout(G, seed=42)  # layout

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition='top center',
            marker=dict(
                showscale=False,
                color='LightSkyBlue',
                size=10,
                line_width=2
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f"{metric.replace('_', ' ')} - Network Graph of Schools and Items",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

    else:
        plot_func = plot_map.get(chart_type, px.line)
        fig = plot_func(
            filtered,
            x='Date',
            y='Value',
            color='School_Name',
            title=f"{metric.replace('_', ' ')} - {chart_type.title()} Chart",
            template=theme
        )

    fig.update_layout(
        xaxis_title='Date' if chart_type != 'box' and chart_type != 'pie' else 'Category',
        yaxis_title='Sales / Cost / Rate',
        legend_title_text='School'
    )
    return fig


if __name__ == '__main__':
    import socket

    def get_free_port():
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    port = get_free_port()
    app.run(debug=True, port=port)





# %%
