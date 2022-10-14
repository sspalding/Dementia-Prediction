# make a dashboard using plotly dash to show the more interesting results that were found

## at the end remove all unused imports
# Import libraries
import pandas as pd
import sqlite3
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
from dash import dash_table
import pickle
import numpy as np
from sklearn import preprocessing
import gunicorn
import dash_bootstrap_components as dbc

# get the data from the database
conn = sqlite3.connect("Dementia.db")                           # connect to the database                   
data_explore = pd.read_sql('SELECT * FROM Data_Wrangled', conn) # this will be the main dataframe used for the dashboard
anova_pval = pd.read_sql('select * from P_Values', conn)        # dataframe of ANOVA P-values calculated in exploratory data analysis

# load ML model
with open('svm_model.pickle', 'rb') as f:
    svm_model = pickle.load(f)

# round anova values to four decimal places
anova_pval = anova_pval.round(4)

# create normalization variable for machine 
X = data_explore[['M/F','EDUC','SES','nWBV']].values
sc = preprocessing.StandardScaler()
sc.fit(X)

# Create a dash application
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1.0,maximim-scale=1.2,minimum-scale=0.5"}])
app.title = 'Dementia Prediction'                               # change the name of the application
server = app.server 

# Text
# markdown text - beginning information
intro_text = '''Sarah Spalding   

The goal of this dashboard is to provide an exploration of the features in the Dementia Prediction dataset and their impact on patient outcome.   

Features:
- SES: Social Economic Status  
- MR Delay: Magnetic Resonance Delay
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

[Link to dataset] (https://www.kaggle.com/datasets/shashwatwork/dementia-prediction-dataset)  
[Link to Github] (https://github.com/sspalding/Dementia-Prediction)
'''
# markdown test - instruction to go above the bargraph section
bargraph_text = '''The bar graph below is intended to give a visual representation of which features have the most impact on patient outcome.  
**Instructions:**  
Hover your mouse over the column you are interested in, this will display the group and the feature average value for that group.
'''
# markdown text - instuctions to go above the scatterplot section
scatterplot_text = '''The scatter plots below are intended to provide an interactive indepth exploration on features impact on outcome classification type.  
**Instructions:**  
Choose a patient outcome classification type from the drop down menu on the right.  
Choose a feature from the drop down menu on the left.   
The first scatter plot will display the data points for all the patients at visit 1 for the charachteristics chosen from the drop down menu.  
The second scatter plot will display the data of one patient across all the visits that patient went to for the outcome type chosen from the drop down menu.   
If you hover your mouse over a data point on the first scatter plot it will display that patient's data on the second scatter plot. 
'''
# markdown text - explanation to go above the anova section
Anova_Text = ''' The table below gives the p-values from one-way ANOVA tests performed on the features for each group.  
The null hypothesis was that the groups were not significantly different.  
The alpha value for the test was 0.05.  
A p-value less than alpha indicated that the null hypothesis should be rejected, therefore signifying the groups being compared were significantly different.  
Values in pink in the table indicate cases where the groups being compared were significantly different.  
From this analysis it can be concluded that features where groups were significantly different (labeled in Dark Yellow) would have the most impact on the outcome of the patients.    
These factors were: normalized whole brain volume, education level, and sex of the patient.   
Social Economic Status (SES) had a p-value less than 0.1 (labeled in yellow) indicating that the groups were very close to being significantly different.
Click for the complete analysis: [ANOVA Test](https://github.com/sspalding/Dementia-Prediction/blob/c2773a763b14b496acd9cb123d1fdbe72b8db8a8/Exploratory%20Data%20Analysis.ipynb)
'''
# markdown text - explanation of machine learning model tool
ML_text = ''' The machine learning model chosen was Support Vector Machine (SVM) model. Logistic regression, k nearest neighbors, \
        support vector machine, and decision tree models were compared and the SVM model was determined to be the one with the best performance. \
        Click the link below to view the full analysis of these models and the specific parameters of the SVM model chosen.  
        [Machine Learning Model](https://github.com/sspalding/Dementia-Prediction/blob/d0c888f2ee76d9f9862f3d1f8fd0779e988df73c/Machine%20Learning.ipynb)  
        **Instructions:**  
        - Fill out the following fields below.  
        - If a field has a red border then the value that has been entered is invalid. Ensure the value entered is a number in the correct range.  
        - Once all of the fields have been filled out click the submit button to view the prediction based on the entered parameters. 
'''

# barplots
# get the data for the barlots from our database
barplots_data = pd.read_sql('select avg("Age") as "Avg_Age",\
                            avg("MR Delay") as "Avg_MR_Delay",\
                            avg("M/F") as "Avg_Sex",\
                            avg("SES") as "Avg_SES",\
                            avg("EDUC") as "Avg_EDUC",\
                            avg("eTIV") as "Avg_eTIV",\
                            avg("nWBV") as "Avg_nWBV",\
                            avg("ASF") as "Avg_ASF",\
                            avg("Visit") as "Avg_Visit",\
                            "Group" from Data_Wrangled group by "Group"', conn) 
# make bargraphs of different features
age_fig = px.bar(barplots_data, x='Group',y='Avg_Age')
mrdelay_fig = px.bar(barplots_data, x ='Group', y='Avg_MR_Delay')
sex_fig = px.bar(barplots_data, x='Group',y='Avg_Sex')
ses_fig = px.bar(barplots_data, x='Group',y='Avg_SES')
educ_fig = px.bar(barplots_data, x='Group',y='Avg_EDUC')
etiv_fig = px.bar(barplots_data, x='Group',y='Avg_eTIV')
nwbv_fig = px.bar(barplots_data, x='Group',y='Avg_nWBV')
asf_fig = px.bar(barplots_data, x='Group',y='Avg_ASF')
visit_fig = px.bar(barplots_data, x='Group',y='Avg_Visit')
# define function to format the barplots
def formatBarplots(figname,yaxis):
        figname.update_traces(marker_color=['#488A99','#DBAE58','#AC3E31'])
        figname.update_yaxes(title_text = yaxis, title_font=dict(size=17,color='#20283E'))
        figname.update_xaxes(title_font=dict(size=17,color='#20283E'))
        figname.update_layout(plot_bgcolor = '#DADADA')
        figname.update_layout(title =f"Group vs. {yaxis} for the Subject's First Visit",title_font=dict(size=20, color='#20283E'))
        return
# call the format function for each bar plot
formatBarplots(age_fig, 'Average Age (years)')
formatBarplots(mrdelay_fig, 'MR Delay')
formatBarplots(sex_fig, 'Sex (Male=1, Female=0)')
formatBarplots(ses_fig, 'Social Economic Status')
formatBarplots(educ_fig, 'Education Level')
formatBarplots(etiv_fig, 'Estimated Total Intracranial Volume')
formatBarplots(nwbv_fig, 'Normalized Whole Brain Volume')
formatBarplots(asf_fig, 'Atlas Scaling Factor')
formatBarplots(visit_fig, 'Number of Visits')

# style variables
# markdown style
markdown_style = {'text-align':'left','color': '#20283E;','font-size':20,'backgroundColor': '#FFFFFF', 'padding':'5px'}
# Tab format
tab_style = {'borderTop': '1px solid #d6d6d6','borderBottom': '1px solid #d6d6d6','backgroundColor': '#DADADA','color':'#20283E','padding': '1px','font-size': 20}
tab_selected_style = {'borderTop': '1px solid #d6d6d6','borderBottom': '1px solid #d6d6d6','backgroundColor':  '#AC3E31','color': '#DADADA','padding': '1px','font-size': 20}
# input style
input_style = {'font-size':15,'color':'#20283E','margin': '5px','padding': '5px'}

# Create an app layout
app.layout = html.Div([
        # make title
        html.H1('Dementia Prediction'),
        dcc.Tabs([
                
                # overview tab
                dcc.Tab(label='Overview',style=tab_style, selected_style=tab_selected_style, children=[
                        # put the markdown text at the top left of the screen
                        html.Div([html.H2('Overview of Project'),
                        dcc.Markdown(children = intro_text, style=markdown_style),
                        # bar plot
                                html.Div([                                                                              
                                        # give bar plot section a title 
                                        html.H2('Comparison of Features Between Groups'),
                                        # call barplot markdown 
                                        dcc.Markdown(children = bargraph_text, style=markdown_style),
                                        # graph barplots, use dbc rows and columns to format
                                        dbc.Row([
                                                dbc.Col(dcc.Graph(id='Age-Barplot',figure=age_fig)),
                                                dbc.Col(dcc.Graph(id='Sex-Barplot',figure=sex_fig)),
                                                dbc.Col(dcc.Graph(id='SES-Barplot',figure=ses_fig)),
                                        ]),
                                        dbc.Row([
                                                dbc.Col(dcc.Graph(id='EDUC-Barplot',figure=educ_fig)),
                                                dbc.Col(dcc.Graph(id='eTIV-Barplot',figure=etiv_fig)),
                                                dbc.Col(dcc.Graph(id='nWBV-Barplot',figure=nwbv_fig))
                                        ]),
                                        dbc.Row([ 
                                                dbc.Col(dcc.Graph(id='ASF-Barplot',figure=asf_fig)),
                                                dbc.Col(dcc.Graph(id='Visit-Barplot',figure=visit_fig)),
                                                dbc.Col(dcc.Graph(id='MRDelay-Barplot', figure=mrdelay_fig))
                                        ]),
                                ]),
                        ]),
                ]),

                # Comparison of features tab
                dcc.Tab(label='Interactive Comparison of Features', style=tab_style, selected_style=tab_selected_style, children=[
                        # give the scatter plot interactive section a title
                        html.H2('In-depth Exploration of Features for Patient Outcome Types'),
                        # call the scatterplot markdown
                        dcc.Markdown(children = scatterplot_text, style=markdown_style),
                        html.Div(className = "row", children=[                                                  
                                html.Div([
                                        # put the name of the dropdown menus in a row
                                        dbc.Row([
                                                dbc.Col(html.P("Patient Outcome Classification Type:")),
                                                dbc.Col(html.P("Feature of Interest:"))
                                        ]),
                                        # make drop down menus in columns of one row
                                        dbc.Row([
                                                dbc.Col(dcc.Dropdown(id='site-dropdown',
                                                                     options=[                                                       # set the options for the dropdown
                                                                             {'label': 'Clinical Dementia Ratio', 'value': 'CDR'},
                                                                             {'label': 'Mini Mental Stat Examination Score', 'value': 'MMSE'},
                                                                             {'label': 'Group', 'value': 'Group'},
                                                                             ],
                                                                     value='ALL',
                                                                     placeholder="Select Patient Outcome Classification Type",       # give the dropdown a placeholder
                                                                     searchable=True,)                                               # allow dropdown to be searchable
                                                ),
                                                dbc.Col(dcc.Dropdown(id='site-dropdown2',
                                                                     options=[                                                       # set the options for the dropdown
                                                                             {'label': 'Age', 'value': 'Age'},
                                                                             {'label': 'Sex', 'value': 'M/F'},
                                                                             {'label': 'Social Economic Status', 'value': 'SES'},
                                                                             {'label': 'Education Level', 'value': 'EDUC'},
                                                                             {'label': 'Estimated Total Intracranial Volume', 'value': 'eTIV'},
                                                                             {'label': 'Normalized Whole Brain Volume', 'value': 'nWBV'},
                                                                             {'label': 'Atlas Scaling Factor', 'value': 'ASF'},
                                                                             {'label': 'MR Delay', 'value':'MR Delay'},
                                                                     ],
                                                                     value='ALL',
                                                                     placeholder="Select Feature of Interest",                       # give the dropdown a placeholder
                                                                     searchable=True,)                                               # allow the dropdown to be searchable
                                                )
                                        ]),
                                        # graph the scatter plots
                                        dbc.Row([
                                                dbc.Col(dcc.Graph(id='scatter-chart', hoverData={'points':[{'customdata':'OAS2_0002'}]}))
                                        ]),
                                        dbc.Row([
                                                dbc.Col(dcc.Graph(id='sub-graph'))
                                        ]),
                                                                             
                                ]),
                        ]),
                ]),

                # ANOVA test results tab
                dcc.Tab(label='ANOVA Test Results', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div([                                                                             
                                html.H2("Table of P-Values from One-Way ANOVA Tests"),
                                # call markdown explaining anova results
                                dcc.Markdown(children = Anova_Text, style=markdown_style),
                                # make a table with ANOVA results
                                dbc.Row([dash_table.DataTable(anova_pval.to_dict('records'),
                                                     style_cell={'font_size':20,'font_family':'times-new-roman'},       # change the font of the anova table
                                                     style_data_conditional=[                                           # make the cells with alpha<0.5 #DBAE58
                                                        {'if':{'filter_query':'{Age}<0.05', 'column_id':'Age'},'backgroundColor':'#DBAE58'},
                                                        {'if':{'filter_query':'{Visit}<0.05', 'column_id':'Visit'},'backgroundColor':'#DBAE58'},
                                                        {'if':{'filter_query':'{MR Delay}<0.05', 'column_id':'MR Delay'},'backgroundColor':'#DBAE58'},
                                                        {'if':{'filter_query':'{M/F}<0.05', 'column_id':'M/F'},'backgroundColor':'#DBAE58'},
                                                        {'if':{'filter_query':'{SES}<0.1', 'column_id':'SES'},'backgroundColor':'#FFE800'},
                                                        {'if':{'filter_query':'{EDUC}<0.05', 'column_id':'EDUC'},'backgroundColor':'#DBAE58'},
                                                        {'if':{'filter_query':'{ASF}<0.05', 'column_id':'ASF'},'backgroundColor':'#DBAE58'},
                                                        {'if':{'filter_query':'{eTIV}<0.05', 'column_id':'eTIV'},'backgroundColor':'#DBAE58'},
                                                        {'if':{'filter_query':'{nWBV}<0.05', 'column_id':'nWBV'},'backgroundColor':'#DBAE58'},
                                                        ])
                                ]),
                        ]),
                ]),

                # Machine Learning Model Tab
                dcc.Tab(label='Machine Learning Model', style=tab_style, selected_style=tab_selected_style, children=[
                html.Div([                      # Machine Learning Model interactive tool
                                html.H2("Machine Learning Model Interactive Tool"),
                                # call markdown explaining the tool
                                dcc.Markdown(children = ML_text, style=markdown_style),
                                # format the input fields and titles as rows and columns
                                dbc.Row([       # let the user choose the sex from a dropdown menu
                                        dbc.Col(html.P('Sex of the Patient: ')),
                                        dbc.Col(dcc.Dropdown(id='site-dropdown-sex',options=[{'label': 'Male', 'value': 1},{'label': 'Female', 'value': 0},],
                                                             value='ALL',placeholder="Select Sex of Patient",searchable=True,))
                                ]),
                                dbc.Row([       # have the user input the SES
                                        dbc.Col(html.P('Social Economic Status of the Patient: ')),
                                        dbc.Col(dcc.Input(id='U_SES', type="number", placeholder="Between 1 and 5", min=1, max=5, style=input_style))
                                ]),
                                dbc.Row([       # have the user input the EDUC
                                        dbc.Col(html.P('Education Level of the Patient: ')),
                                        dbc.Col(dcc.Input(id='U_EDUC', type="number", placeholder="Below 23", min=0, max=23, style=input_style))
                                ]),
                                dbc.Row([       # have the user input the nWBV
                                        dbc.Col(html.P('Normalized Whole Brain Volume of the Patient: ')),
                                        dbc.Col(dcc.Input(id='U_nWBV', type="number", placeholder="Between 0 and 1", min=0, max=1, style=input_style))
                                ]),
                                dbc.Row([
                                        # add a button that tracks the number of clicks
                                        dbc.Col(html.Button('Submit', id='submit-val', n_clicks=0)),
                                        # print the prediction of the machine learning model
                                        dbc.Col(html.Div(id='Prediction', style={'font-size':25,'color':'#20283E'}))
                                ]),
                                
                        ])
                ])
        ])
])

# create the callback and the function for the main scatter plot - inputs from two dropdown menus
@app.callback(Output(component_id='scatter-chart', component_property='figure'),
                Input(component_id='site-dropdown', component_property='value'),                # get the outcome from the first dropdown menu
                Input(component_id='site-dropdown2', component_property='value'),               # get the feature from the second dropdown menu
                )
def make_scatter_chart(site_dropdown_choice, site_dropdown2_choice):
        graph_data = data_explore.where(data_explore['Visit'] == 1)                             # only get the data from visit 1 of all the patients
        fig2 = px.scatter(graph_data, 
                          x=graph_data.loc[:,site_dropdown2_choice],                            # make x the feature from the second drop down menu
                          y=graph_data.loc[:,site_dropdown_choice],                             # make y the outcome from the first drop down menu 
                          hover_name=graph_data['Subject ID']                                   # display the patient id when you hover over a data point
                          )
        fig2.update_traces(customdata = data_explore.loc[:site_dropdown_choice]['Subject ID'])  # assign subject id to customvariable
        fig2.update_yaxes(title_font=dict(size=17))                                             # change the font size of the y axis
        fig2.update_xaxes(title_font=dict(size=17))                                             # change the font size of the x axis
        fig2.update_traces(marker_color = '#AC3E31')                                            # change the plot color
        fig2.update_layout(plot_bgcolor = '#DADADA')                                            # change the background color
        fig2.update_yaxes(title_font=dict(size=17,color='#20283E'))                             # change the y axis title
        fig2.update_xaxes(title_font=dict(size=17,color='#20283E'))                             # change the x axis title
        # assign a title to the graph and change the font size
        fig2.update_layout(title =f"{site_dropdown2_choice} vs. {site_dropdown_choice} for the Subject's First Visit",title_font=dict(size=20, color='#20283E'))
        return fig2                                                                             # return the figure

#create other scatter plot that shows the details of one patient
def create_subgraph(subgraph_Data, site_dropdown_choice,subject_id):
        fig3 = px.scatter(subgraph_Data, 
                          x=subgraph_Data['Visit'],                                             # assign the x axis to be the Visit
                          y=subgraph_Data.loc[:,site_dropdown_choice]                           # assign the y axis to be the outcome chosen from the dirst dropdown menu
                          )
        fig3.update_yaxes(title_font=dict(size=17))                                             # change the font size of the y axis
        fig3.update_xaxes(title_font=dict(size=17))                                             # change the font size of the x axis
        fig3.update_traces(marker_color = '#488A99', marker_size = 10)                          # change the plot color
        fig3.update_layout(plot_bgcolor = '#DADADA')                                            # change the background color
        fig3.update_yaxes(title_font=dict(size=17,color='#20283E'))                             # change the y axis title
        fig3.update_xaxes(title_font=dict(size=17,color='#20283E'))                             # change the x axis title 
        # assign a title to the graph and change the font size
        fig3.update_layout(title =f"Visit vs. {site_dropdown_choice} for Subject {subject_id}",title_font=dict(size=20, color='#20283E'))
        return fig3                                                                             # return the figure 

# create the callback and function that will update the second scatter plot
@app.callback(Output(component_id='sub-graph', component_property='figure'),                    # output the figure of the following function
                Input(component_id='scatter-chart', component_property='hoverData'),            # get the patient the user hovers over from the first graph
                Input(component_id='site-dropdown', component_property='value'),                # get the outcome from the first dropdown menu
                )
def update_subgraph(hoverData, site_dropdown_choice):
    subject_id = hoverData['points'][0]['customdata']                                           # get the subject id from what point the user hovers over
    subgraph_Data = data_explore.where(data_explore['Subject ID']==subject_id)                  # get the data for only that subject 
    return create_subgraph(subgraph_Data,site_dropdown_choice,subject_id)                       # return the subgraph function that was created earlier

# create the callback function for the machine learning model
@app.callback(Output(component_id='Prediction', component_property='children'),                 # output the prediction from the following model
                Input(component_id='submit-val',component_property='n_clicks'),                 # get the number of times the button has been clicked as the input
                State(component_id='site-dropdown-sex', component_property='value'),
                State(component_id='U_SES', component_property='value'),
                State(component_id='U_EDUC', component_property='value'),
                State(component_id='U_nWBV', component_property='value'),
        )
# create a function to run the machine learning model using the user input
def run_model(n_clicks,sex,SES,EDUC,nWBV, ):
        user_input = np.array([[sex, EDUC, SES, nWBV,]])         # create an array of the user input
        user_input = sc.transform(user_input)                                           # normalize that array based on the training data
        prediction = svm_model.predict(user_input)                                      # run the model to get a prediction                                                     # set the previous number of clicks to the current number of clicks
        return f'Prediction: {prediction}'                                              # return the model

# Run the app
if __name__ == '__main__':
    app.run_server()
