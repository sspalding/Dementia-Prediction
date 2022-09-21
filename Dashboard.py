# make a dashboard using plotly dash to show the more interesting results that were found

# Import libraries
import site
import pandas as pd
import sqlite3
import dash
from dash import html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, dash_table

# get the data from the database
conn = sqlite3.connect("Dementia.db")
sql_string = 'SELECT * FROM Data_Wrangled'
data_explore = pd.read_sql(sql_string, conn)
Anova_pVal = pd.read_sql('select * from P_Values', conn)

# Create a dash application
app = dash.Dash(__name__)

# markdown text - beginning information
intro_text = ''' Sarah Spalding   

The goal of this dashboard is to provide an exploration of the features in the Dementia Prediction dataset and their impact on patient outcome.   

Features:
- SES: Social Economic Status  
- EDUC: Education Level  
- eTIV: Estimated Total Intracranial Volume  
- nWBV: Normalized Whole Brain Volume  
- ASF: Atlas Scaling Factor  
- Visit: Number of Subject Visits  
- Age: Patient Age in Years
- M/F: The Sex of the Patient  
    - 0: Female
    - 1: Male  

Patient Outcome Classification Type:
- MMSE: Mini-Mental Stat Examination Score  
    - Range: 1-30  
    - <10: Extreme Impairement  
    - 10-19: Moderate Dementia  
    - 19-24: Early stage Alzheimer's dementia  
    - 25<: Normal   
- CDR: Clinical Dementia Ratio  
    - Range 0-3  
    - 0: None  
    - 0.5: Very-Mild  
    - 1: Mild  
    - 2: Moderate  
    - 3: Extreme  
- Group: Overall Classification of the Patient  
    - Demented: Classified as Having Dementia
    - Nondemented: Classified as Not Having Dementia  
    - Converted: Classified as Developed Dementia During the Time of the Study  

[Link to dataset] (https://www.kaggle.com/datasets/shashwatwork/dementia-prediction-dataset)  
[Link to Github] (https://github.com/sspalding/Dementia-Prediction)
'''
bargraph_text = '''The bar graph below is intended to give a visual representation of which features have the most impact on patient outcome.  
**Instructions:**  
Hover your mouse over the column you are interested in, this will display the group and the feature average value for that group.
'''
scatterplot_text = '''The scatter plots below are intended to provide an interactive indepth exploration on features impact on outcome classification type.  
**Instructions:**  
Choose a patient outcome classification type from the drop down menu on the right.  
Choose a feature from the drop down menu on the left.   
The scatter plot on the left will display the data points for all the patients at visit 1 for the charachteristics chosen from the drop down menu.  
The scatter plot on the right will display the data of one patient across all the visits that patient went to for the outcome type chosen from the drop down menu.   
If you hover your mouse over a data point on the scatter plot on the left it will display that patient's data on the scatter plot on the right. 
'''
Anova_Text = ''' The table below gives the p-values from one-way ANOVA tests performed on the features for each group.  
The null hypothesis was that the groups were not significantly different.  
The alpha value for the test was 0.05.  
A p-value less than alpha indicated that the null hypothesis should be rejected, therefore signifying the groups being compared were significantly different.  
Values in pink in the table indicate cases where the groups being compared were significantly different.  
From this analysis it can be concluded that features where groups were significantly different (labeled in pink) would have the most impact on the outcome of the patients.  
These factors were: normalized whole brain volume, education level, social economic status, sex, number of visits the patient attended, and age of the patient. 
'''
# make bargraphs of different factors
barplot = make_subplots(rows=4,cols=2,horizontal_spacing=0.075,vertical_spacing=0.075,
                        subplot_titles=('Average Age',
                                        'Average Sex (M/F, 0 = Female, 1 = Male)',
                                        'Average Social Economic Status (SES)',
                                        'Average Education Level (EDUC)',
                                        'Average Estimated Total Intracranial Volume (eTIV)',
                                        'Average Normalized Whole Brain Volume (nWBV)',
                                        'Average Atlas Scaling Factor (ASF)',
                                        'Average Number of Visits'))
# get the data for the subplots from our database
SubPlots_Data = pd.read_sql('select avg("Age") as "Avg_Age",\
                            avg("M/F") as "Avg_Sex",\
                            avg("SES") as "Avg_SES",\
                            avg("EDUC") as "Avg_EDUC",\
                            avg("eTIV") as "Avg_eTIV",\
                            avg("nWBV") as "Avg_nWBV",\
                            avg("ASF") as "Avg_ASF",\
                            avg("Visit") as "Avg_Visit",\
                            "Group" from Data_Wrangled group by "Group"', conn) 
# add subplots                                
barplot.add_trace(go.Bar(x =SubPlots_Data['Group'], y=SubPlots_Data['Avg_Age']), row=1,col=1)
barplot.add_trace(go.Bar(x =SubPlots_Data['Group'], y=SubPlots_Data['Avg_Sex']), row=1,col=2)
barplot.add_trace(go.Bar(x =SubPlots_Data['Group'], y=SubPlots_Data['Avg_SES']), row=2,col=1)
barplot.add_trace(go.Bar(x =SubPlots_Data['Group'], y=SubPlots_Data['Avg_EDUC']), row=2,col=2)
barplot.add_trace(go.Bar(x =SubPlots_Data['Group'], y=SubPlots_Data['Avg_eTIV']), row=3,col=1)
barplot.add_trace(go.Bar(x =SubPlots_Data['Group'], y=SubPlots_Data['Avg_nWBV']), row=3,col=2)
barplot.add_trace(go.Bar(x =SubPlots_Data['Group'], y=SubPlots_Data['Avg_ASF']), row=4,col=1)
barplot.add_trace(go.Bar(x =SubPlots_Data['Group'], y=SubPlots_Data['Avg_Visit']), row=4,col=2)
# add Axis title
barplot.update_yaxes(title_text = 'Average Age (years)', row=1,col=1)
barplot.update_yaxes(title_text = 'Average Sex', row=1,col=2)
barplot.update_yaxes(title_text = 'Average SES', row=2,col=1)
barplot.update_yaxes(title_text = 'Average Education', row=2,col=2)
barplot.update_yaxes(title_text = 'Average eTIV', row=3,col=1)
barplot.update_yaxes(title_text = 'Average nWBV', row=3,col=2)
barplot.update_yaxes(title_text = 'Average ASF', row=4,col=1)
barplot.update_yaxes(title_text = 'Average Visits', row=4,col=2)
barplot.update_yaxes(title_font=dict(size=17))
barplot.update_xaxes(title_font=dict(size=17))
# add figure title and format
barplot.update_layout(showlegend=False, height=1000)
barplot.update_annotations({'font': {'size': 20}})

# Create an app layout
app.layout = html.Div(children=[
        # make title
        html.H1('Dementia Prediction',style={'textAlign': 'center', 'color': '#503D36','font-size': 40}),
        # put the markdown text at the top left of the screen
        dcc.Markdown(children = intro_text,style={'textAlign': 'left', 'color': '#503D36','font-size': 25}),
        # plot the bar plot
        html.H2('Comparison of Features Between Groups',style={'textAlign': 'left', 'color': '#503D36','font-size': 30}),
        dcc.Markdown(children = bargraph_text,style={'textAlign': 'left', 'color': '#503D36','font-size': 25}),
        dcc.Graph(id='Group_of_BarPlots',figure=barplot),
        # insert a break
        html.Br(),
        # make the drop down menus side by side
        html.H2('In-depth Exploration of Features for Patient Outcome Types',style={'textAlign': 'left', 'color': '#503D36','font-size': 30}),
        dcc.Markdown(children = scatterplot_text,style={'textAlign': 'left', 'color': '#503D36','font-size': 25}),
        html.Div(className = "row", children=[
                html.Div([
                        # make a drop down menu to choose the dementia classication mode
                        html.H2("Patient Outcome Classification Type",style={'textAlign': 'left', 'color': '#503D36','font-size': 30}),
                        dcc.Dropdown(id='site-dropdown',
                                options=[
                                        {'label': 'Clinical Dementia Ratio', 'value': 'CDR'},
                                        {'label': 'Mini Mental Stat Examination Score', 'value': 'MMSE'},
                                        {'label': 'Group', 'value': 'Group'},
                                        ],
                                style={'color': '#503D36','font-size': 25},
                                value='ALL',
                                placeholder="Select Patient Outcome Classification Type",
                                searchable=True,
                                )], style=dict(width='50%')
                ),
                html.Div([
                        # make a drop down menu to choose the Factor of interest 
                        html.H2("Feature of Interest",style={'textAlign': 'left', 'color': '#503D36','font-size': 30}),
                        dcc.Dropdown(id='site-dropdown2',
                                options=[
                                        {'label': 'Age', 'value': 'Age'},
                                        {'label': 'Sex', 'value': 'M/F'},
                                        {'label': 'Social Economic Status', 'value': 'SES'},
                                        {'label': 'Education Level', 'value': 'EDUC'},
                                        {'label': 'Estimated Total Intracranial Volume', 'value': 'eTIV'},
                                        {'label': 'Normalized Whole Brain Volume', 'value': 'nWBV'},
                                        {'label': 'Atlas Scaling Factor', 'value': 'ASF'},
                                ],
                                style={'color': '#503D36','font-size': 25},
                                value='ALL',
                                placeholder="Select Feature of Interest",
                                searchable=True,
                                )],style=dict(width='50%')
                        ),
                ], style=dict(display='flex')),
        html.Div([
                html.Div([
                        # make a scatter plot
                        dcc.Graph(id='scatter-chart', hoverData={'points':[{'customdata':'OAS2_0002'}]}),
                ], style=dict(width='50%')),
                html.Div([
                        # make a scatter plot
                        dcc.Graph(id='sub-graph'),
                ], style=dict(width='50%')),
        ], style=dict(display='flex')),
        
        html.Div([
                # make a table with ANOVA results
                html.H2("Table of P-Values from One-Way ANOVA Tests",style={'textAlign': 'left', 'color': '#503D36','font-size': 30}),
                dcc.Markdown(children = Anova_Text,style={'textAlign': 'left', 'color': '#503D36','font-size': 25}),
                dash_table.DataTable(Anova_pVal.to_dict('records'),
                                     style_cell={'font_size':20},
                                     style_data_conditional=[
                                        {'if':{'filter_query':'{Age}<0.05', 'column_id':'Age'},'backgroundColor':'pink'},
                                        {'if':{'filter_query':'{Visit}<0.05', 'column_id':'Visit'},'backgroundColor':'pink'},
                                        {'if':{'filter_query':'{M/F}<0.05', 'column_id':'M/F'},'backgroundColor':'pink'},
                                        {'if':{'filter_query':'{SES}<0.05', 'column_id':'SES'},'backgroundColor':'pink'},
                                        {'if':{'filter_query':'{EDUC}<0.05', 'column_id':'EDUC'},'backgroundColor':'pink'},
                                        {'if':{'filter_query':'{ASF}<0.05', 'column_id':'ASF'},'backgroundColor':'pink'},
                                        {'if':{'filter_query':'{eTIV}<0.05', 'column_id':'eTIV'},'backgroundColor':'pink'},
                                        {'if':{'filter_query':'{nWBV}<0.05', 'column_id':'nWBV'},'backgroundColor':'pink'},
                                        ])
        ])
])

# make main scatter plot - inputs from two dropdown menus
@app.callback(Output(component_id='scatter-chart', component_property='figure'),
                Input(component_id='site-dropdown', component_property='value'),
                Input(component_id='site-dropdown2', component_property='value'),
                )
def make_scatter_chart(site_dropdown_choice, site_dropdown2_choice):
        graph_data = data_explore.where(data_explore['Visit'] == 1)
        fig2 = px.scatter(graph_data, 
                          x=graph_data.loc[:,site_dropdown2_choice], 
                          y=graph_data.loc[:,site_dropdown_choice], 
                          hover_name=graph_data['Subject ID'] 
                          )
        fig2.update_traces(customdata = data_explore.loc[:site_dropdown_choice]['Subject ID'])
        fig2.update_yaxes(title_font=dict(size=17))
        fig2.update_xaxes(title_font=dict(size=17))
        fig2.update_layout(title =f"{site_dropdown2_choice} vs. {site_dropdown_choice} for the Subject's First Visit",title_font=dict(size=20))
        return fig2
#create other scatter plot that shows the details of one patient
def create_subgraph(subgraph_Data, site_dropdown_choice,subject_id):
        fig3 = px.scatter(subgraph_Data, 
                          x=subgraph_Data['Visit'], 
                          y=subgraph_Data.loc[:,site_dropdown_choice]
                          )
        fig3.update_yaxes(title_font=dict(size=17))
        fig3.update_xaxes(title_font=dict(size=17))
        fig3.update_layout(title =f"Visit vs. {site_dropdown_choice} for Subject {subject_id}",title_font=dict(size=20))
        return fig3  
# create the callback and function that will update the second scatter plot
@app.callback(Output(component_id='sub-graph', component_property='figure'),
                Input(component_id='scatter-chart', component_property='hoverData'),
                Input(component_id='site-dropdown', component_property='value'),
                )
def update_subgraph(hoverData, site_dropdown_choice):
    subject_id = hoverData['points'][0]['customdata']
    subgraph_Data = data_explore.where(data_explore['Subject ID']==subject_id)
    return create_subgraph(subgraph_Data,site_dropdown_choice,subject_id)

# Run the app
if __name__ == '__main__':
    app.run_server()