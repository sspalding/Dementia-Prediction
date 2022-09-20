# make a dashboard using plotly dash to show the more interesting results that were found
# https://dash.plotly.com/interactive-graphing
# Import libraries
import pandas as pd
import sqlite3
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# get the data from the database
conn = sqlite3.connect("Dementia.db")
sql_string = 'SELECT * FROM Data_Wrangled'
data_explore = pd.read_sql(sql_string, conn)

# Create a dash application
app = dash.Dash(__name__)

# markdown text - beginning information
markdown_text = '''Examination of the features in the Dementia Prediction dataset.  
[Link to dataset] (https://www.kaggle.com/datasets/shashwatwork/dementia-prediction-dataset)  
[Link to Github] (https://github.com/sspalding/Dementia-Prediction)
'''
# make bargraphs of different factors
barplot = make_subplots(rows=4,cols=2,horizontal_spacing=0.075,vertical_spacing=0.075,
                        subplot_titles=('Average Age',
                                        'Average Sex(0 = Female, 1 = Male)',
                                        'Average Social Economic Status',
                                        'Average Education Level',
                                        'Average Estimated Total Intracranial Volume',
                                        'Average Normalized Whole Brain Volume',
                                        'Average Atlas Scaling Factor',
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
barplot.update_layout(showlegend=False, height=1000, title = 'Comparison of Factors Between Groups',font_size=20)
barplot.update_annotations({'font': {'size': 20}})

# Create an app layout
app.layout = html.Div(children=[html.H1('Dementia Prediction',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # put the markdown text at the top left of the screen
                                dcc.Markdown(children = markdown_text,style={'textAlign': 'left', 'color': '#503D36',
                                               'font-size': 30}),
                                # plot the bar plot
                                dcc.Graph(id='Group_of_BarPlots',
                                              figure=barplot
                                            ),
                                html.Br(),
                                # make a drop down menu to choose the dementia classication mode
                                html.P("Dementia Classification"),
                                dcc.Dropdown(id='site-dropdown',
                                                options=[
                                                    {'label': 'Clinical Dementia Ratio', 'value': 'CDR'},
                                                    {'label': 'Mini Mental Stat Examination Score', 'value': 'MMSE'},
                                                    {'label': 'Group', 'value': 'Croup'},
                                                ],
                                                value='ALL',
                                                placeholder="Select Dementia Classification Here",
                                                searchable=True
                                                ),
                                # make a scatter plot

                                # make sliders
                                # Patient age Slider
                                html.P("Patient Age (Years)"),
                                dcc.RangeSlider(id='Age-slider',min=50,max=100,step=5,value=[50,100]),
                                # Patient Sex Slider
                                html.P("Patient Sex"),
                                dcc.RangeSlider(id='Sex-slider',min=0,max=1,step=1,value=[0,1],marks={0: 'Female', 1:'Male'}),
                                # Patient SES Slider
                                html.P("Patient Social Economic Status"),
                                dcc.RangeSlider(id='SES-slider',min=0,max=5,step=1,value=[0,5]),
                                # Patient Education Level Slider
                                html.P("Patient Education Level"),
                                dcc.RangeSlider(id='EDUC-slider',min=5,max=25,step=1,value=[5,25]),
                                # Patient eTIV Slider
                                html.P("Patient Estimated Total Intracranial Volume"),
                                dcc.RangeSlider(id='eTIV-slider',min=1100,max=2100,step=100,value=[1101,2100],
                                                marks={1100: '1100', 1200:'1200',1300:'1300',1400:'1400',1500:'1500',1600:'1600',1700:'1700',1800:'1800',1900:'1900',2000:'2000',2100:'2100'}),
                                # Patient nWBV Slider
                                html.P("Patient Normalized Whole Brain Volume"),
                                dcc.RangeSlider(id='nWBV-slider',min=0.5,max=1,step=0.1,value=[0.5,1]),
                                # Patient ASF Slider
                                html.P("Patient Atlas Scaling Factor"),
                                dcc.RangeSlider(id='ASF-slider',min=0.8,max=1.6,step=0.1,value=[0.8,1.6]),
                                # Patient Visit Slider
                                html.P("Patient Number of Visits"),
                                dcc.RangeSlider(id='Visit-slider',min=0,max=5,step=1,value=[0,5]),
                                ])
# Run the app
if __name__ == '__main__':
    app.run_server()