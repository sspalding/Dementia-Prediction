# make a dashboard using plotly dash to show the more interesting results that were found

## at the end remove all unused imports
# Import libraries
import pandas as pd
import sqlite3
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import dash_table
import pickle
import numpy as np
from sklearn import preprocessing

# get the data from the database
conn = sqlite3.connect("Dementia.db")                           # connect to the database
query = 'SELECT * FROM Data_Wrangled'                   
data_explore = pd.read_sql(query, conn)                         # this will be the main dataframe used for the dashboard
Anova_pVal = pd.read_sql('select * from P_Values', conn)        # dataframe of ANOVA P-values calculated in exploratory data analysis

# load ML model
with open('svm_model.pickle', 'rb') as f:
    svm_model = pickle.load(f)

# create normalization variable for machine 
X = data_explore[['Visit','M/F','Age','EDUC','SES','eTIV','nWBV','ASF']].values
sc = preprocessing.StandardScaler()
sc.fit(X)

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
The scatter plot on the left will display the data points for all the patients at visit 1 for the charachteristics chosen from the drop down menu.  
The scatter plot on the right will display the data of one patient across all the visits that patient went to for the outcome type chosen from the drop down menu.   
If you hover your mouse over a data point on the scatter plot on the left it will display that patient's data on the scatter plot on the right. 
'''
# markdown text - explanation to go above the anova section
Anova_Text = ''' The table below gives the p-values from one-way ANOVA tests performed on the features for each group.  
The null hypothesis was that the groups were not significantly different.  
The alpha value for the test was 0.05.  
A p-value less than alpha indicated that the null hypothesis should be rejected, therefore signifying the groups being compared were significantly different.  
Values in pink in the table indicate cases where the groups being compared were significantly different.  
From this analysis it can be concluded that features where groups were significantly different (labeled in pink) would have the most impact on the outcome of the patients.  
These factors were: normalized whole brain volume, education level, social economic status, sex, number of visits the patient attended, and age of the patient. 
'''
# markdown text - explanation of machine learning model tool
ML_text = ''' The machine learning model chosen was Support Vector Machine (SVM) model. Logistic regression, k nearest neighbors, \
        support vector machine, and decision tree models were compared and the SVM model was determined to be the one with the highest accuracy. \
        Click the link below to view the full analysis of these models and the specific parameters of the SVM model chosen.  
        [Machine Learning Model Exploration](https://github.com/sspalding/Dementia-Prediction/blob/d0c888f2ee76d9f9862f3d1f8fd0779e988df73c/Machine%20Learning.ipynb)  
        **Instructions:**  
        - Fill out the following fields below.  
        - If a field has a red border then the value that has been entered is invalid. Ensure the value entered is a number in the correct range.  
        - Once all of the fields have been filled out click the submit button to view the prediction based on the entered parameters. 
'''
# make bargraphs of different features
row_count = 4
col_count = 2
barplot = make_subplots(rows=row_count,cols=col_count,horizontal_spacing=0.075,vertical_spacing=0.075,
                        subplot_titles=('Average Age',
                                        'Average Sex (M/F, 0 = Female, 1 = Male)',
                                        'Average Social Economic Status (SES)',
                                        'Average Education Level (EDUC)',
                                        'Average Estimated Total Intracranial Volume (eTIV)',
                                        'Average Normalized Whole Brain Volume (nWBV)',
                                        'Average Atlas Scaling Factor (ASF)',
                                        'Average Number of Visits'))
# get the data for the subplots from our database
subplots_data = pd.read_sql('select avg("Age") as "Avg_Age",\
                            avg("M/F") as "Avg_Sex",\
                            avg("SES") as "Avg_SES",\
                            avg("EDUC") as "Avg_EDUC",\
                            avg("eTIV") as "Avg_eTIV",\
                            avg("nWBV") as "Avg_nWBV",\
                            avg("ASF") as "Avg_ASF",\
                            avg("Visit") as "Avg_Visit",\
                            "Group" from Data_Wrangled group by "Group"', conn) 

# add subplots                                
barplot.add_trace(go.Bar(x =subplots_data['Group'], y=subplots_data['Avg_Age']), row=1,col=1)
barplot.add_trace(go.Bar(x =subplots_data['Group'], y=subplots_data['Avg_Sex']), row=1,col=2)
barplot.add_trace(go.Bar(x =subplots_data['Group'], y=subplots_data['Avg_SES']), row=2,col=1)
barplot.add_trace(go.Bar(x =subplots_data['Group'], y=subplots_data['Avg_EDUC']), row=2,col=2)
barplot.add_trace(go.Bar(x =subplots_data['Group'], y=subplots_data['Avg_eTIV']), row=3,col=1)
barplot.add_trace(go.Bar(x =subplots_data['Group'], y=subplots_data['Avg_nWBV']), row=3,col=2)
barplot.add_trace(go.Bar(x =subplots_data['Group'], y=subplots_data['Avg_ASF']), row=4,col=1)
barplot.add_trace(go.Bar(x =subplots_data['Group'], y=subplots_data['Avg_Visit']), row=4,col=2)

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
# format barplot
barplot.update_layout(showlegend=False, height=1000)
barplot.update_annotations({'font': {'size': 20}})

# Create an app layout
app.layout = html.Div([
        # make title
        html.H1('Dementia Prediction',style={'textAlign': 'center', 'color': '#503D36','font-size': 40}),
        dcc.Tabs([
                dcc.Tab(label='Overview', style={'font-size':20}, children=[
                        # put the markdown text at the top left of the screen
                        dcc.Markdown(children = intro_text,style={'textAlign': 'left', 'color': '#503D36','font-size': 25}),
                        # bar plot
                        html.Div([                                                                              
                                # give bar plot section a title 
                                html.H2('Comparison of Features Between Groups',style={'textAlign': 'left', 'color': '#503D36','font-size': 30}),
                                # call barplot markdown 
                                dcc.Markdown(children = bargraph_text,style={'textAlign': 'left', 'color': '#503D36','font-size': 25}),
                                # graph barplot
                                dcc.Graph(id='Group_of_BarPlots',figure=barplot),
                        ]),
                ]),
                dcc.Tab(label='Interactive Comparison of Features', style={'font-size':20}, children=[
                        # give the scatter plot interactive section a title
                        html.H2('In-depth Exploration of Features for Patient Outcome Types',style={'textAlign': 'left', 'color': '#503D36','font-size': 30}),
                        # call the scatterplot markdown
                        dcc.Markdown(children = scatterplot_text,style={'textAlign': 'left', 'color': '#503D36','font-size': 25}),
                        html.Div(className = "row", children=[                                                  # drop down menus
                                html.Div([
                                        # make a drop down menu to choose the dementia classication mode
                                        html.H2("Patient Outcome Classification Type",style={'textAlign': 'left', 'color': '#503D36','font-size': 30}),
                                        dcc.Dropdown(id='site-dropdown',
                                                options=[                                                       # set the options for the dropdown
                                                        {'label': 'Clinical Dementia Ratio', 'value': 'CDR'},
                                                        {'label': 'Mini Mental Stat Examination Score', 'value': 'MMSE'},
                                                        {'label': 'Group', 'value': 'Group'},
                                                        ],
                                                style={'color': '#503D36','font-size': 25},
                                                value='ALL',
                                                placeholder="Select Patient Outcome Classification Type",       # give the dropdown a placeholder
                                                searchable=True,                                                # allow the user to search the dropdown
                                                )], style=dict(width='50%')                                     # make drop down take up 50% of screen
                                ),
                                html.Div([
                                        # make a drop down menu to choose the Feature of interest 
                                        html.H2("Feature of Interest",style={'textAlign': 'left', 'color': '#503D36','font-size': 30}),
                                        dcc.Dropdown(id='site-dropdown2',
                                                options=[                                                       # set the options for the dropdown
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
                                                placeholder="Select Feature of Interest",                       # give the dropdown a placeholder
                                                searchable=True,                                                # allow the user to search the dropdown 
                                                )],style=dict(width='50%')                                      # make drop down take up 50% of screen
                                        ),
                                ], style=dict(display='flex')),                                                 # put drop downs side by side 
                        html.Div([                                                                              # input the scatter plots
                                html.Div([
                                        # make a scatter plot
                                        dcc.Graph(id='scatter-chart', hoverData={'points':[{'customdata':'OAS2_0002'}]}),
                                ], style=dict(width='50%')),                                                    # make scatter plot take up 50% of screen
                                html.Div([
                                        # make the second scatter plot
                                        dcc.Graph(id='sub-graph'),
                                ], style=dict(width='50%')),                                                    # make scatter plot take up 50% of screen
                        ], style=dict(display='flex')),                                                         # put the scatter plots side by side

                ]),
                dcc.Tab(label='ANOVA Test Results', style={'font-size':20}, children=[
                        html.Div([                                                                              # Anova table
                                html.H2("Table of P-Values from One-Way ANOVA Tests",style={'textAlign': 'left', 'color': '#503D36','font-size': 30}),
                                # call markdown explaining anova results
                                dcc.Markdown(children = Anova_Text,style={'textAlign': 'left', 'color': '#503D36','font-size': 25}),
                                # make a table with ANOVA results
                                dash_table.DataTable(Anova_pVal.to_dict('records'),
                                                     style_cell={'font_size':20},                               # change the font of the anova table
                                                     style_data_conditional=[                                   # make the cells with alpha<0.5 pink
                                                        {'if':{'filter_query':'{Age}<0.05', 'column_id':'Age'},'backgroundColor':'pink'},
                                                        {'if':{'filter_query':'{Visit}<0.05', 'column_id':'Visit'},'backgroundColor':'pink'},
                                                        {'if':{'filter_query':'{M/F}<0.05', 'column_id':'M/F'},'backgroundColor':'pink'},
                                                        {'if':{'filter_query':'{SES}<0.05', 'column_id':'SES'},'backgroundColor':'pink'},
                                                        {'if':{'filter_query':'{EDUC}<0.05', 'column_id':'EDUC'},'backgroundColor':'pink'},
                                                        {'if':{'filter_query':'{ASF}<0.05', 'column_id':'ASF'},'backgroundColor':'pink'},
                                                        {'if':{'filter_query':'{eTIV}<0.05', 'column_id':'eTIV'},'backgroundColor':'pink'},
                                                        {'if':{'filter_query':'{nWBV}<0.05', 'column_id':'nWBV'},'backgroundColor':'pink'},
                                                        ])
                        ]),
                ]),
                dcc.Tab(label='Machine Learning Model', style={'font-size':20}, children=[
                        html.Div([                                                                              # Machine Learning Model interactive tool
                                html.H2("Machine Learning Model Interactive Tool",style={'textAlign': 'left', 'color': '#503D36','font-size': 30}),
                                # call markdown explaining the tool
                                dcc.Markdown(children = ML_text,style={'textAlign': 'left', 'color': '#503D36','font-size': 20}),
                                # have the user input the subject information
                                html.Div([html.P('Sex of the Patient: ',style={'font-size':20}),
                                        dcc.Dropdown(id='site-dropdown-sex',options=[{'label': 'Male', 'value': 1},
                                                                                     {'label': 'Female', 'value': 0},],
                                                     style={'font-size': 20},value='ALL',placeholder="Select Sex of Patient",searchable=True,)
                                ],style=dict(width=263)),
                                html.Div([
                                        html.P('Number of Visits the Patient Attended: ',style={'font-size':20}),
                                        dcc.Input(id='U_visit',type="number",placeholder="Between 1 and 5", min=1, max=5,style={'font-size':20})
                                ]),
                                html.Div([
                                        html.P('Age of the Patient: ',style={'font-size':20}),
                                        dcc.Input(id='U_Age',type="number",placeholder="Years", min=1, max=120, style={'font-size':20})
                                ]),
                                html.Div([
                                        html.P('Social Economic Status of the Patient: ',style={'font-size':20}),
                                        dcc.Input(id='U_SES', type="number", placeholder="Between 1 and 5", min=1, max=5, style={'font-size':20})
                                ]),
                                html.Div([
                                        html.P('Education Level of the Patient: ',style={'font-size':20}),
                                        dcc.Input(id='U_EDUC', type="number", placeholder="Below 23", min=0, max=23, style={'font-size':20})
                                ]),
                                html.Div([
                                        html.P('Atlas Scaling Factor of the Patient',style={'font-size':20}),
                                        dcc.Input(id='U_ASF', type="number", placeholder="Between 0 and 2", min=0, max=2, style={'font-size':20})
                                ]),
                                html.Div([
                                        html.P('Estimated Total Intracranial Volume of the Patient: ',style={'font-size':20}),
                                        dcc.Input(id='U_eTIV', type="number", placeholder="Between 0 and 3000", min=0, max=3000, style={'font-size':20})
                                ]),
                                html.Div([
                                        html.P('Normalized Whole Brain Volume of the Patient: ',style={'font-size':20}),
                                        dcc.Input(id='U_nWBV', type="number", placeholder="Between 0 and 1", min=0, max=1, style={'font-size':20})
                                ]),
                                html.Br(),
                                html.Div([html.Button('Submit', id='submit-val', n_clicks=0, style={'font-size':25})]),

                                # output to user the predicted subject group
                                html.Div([html.Div(id='Prediction',style={'font-size':25})])
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
        # assign a title to the graph and change the font size
        fig2.update_layout(title =f"{site_dropdown2_choice} vs. {site_dropdown_choice} for the Subject's First Visit",title_font=dict(size=20))
        return fig2                                                                             # return the figure

#create other scatter plot that shows the details of one patient
def create_subgraph(subgraph_Data, site_dropdown_choice,subject_id):
        fig3 = px.scatter(subgraph_Data, 
                          x=subgraph_Data['Visit'],                                             # assign the x axis to be the Visit
                          y=subgraph_Data.loc[:,site_dropdown_choice]                           # assign the y axis to be the outcome chosen from the dirst dropdown menu
                          )
        fig3.update_yaxes(title_font=dict(size=17))                                             # change the font size of the y axis
        fig3.update_xaxes(title_font=dict(size=17))                                             # change the font size of the x axis
        # assign a title to the graph and change the font size
        fig3.update_layout(title =f"Visit vs. {site_dropdown_choice} for Subject {subject_id}",title_font=dict(size=20))
        return fig3                                                                             # return the figure 

# create the callback and function that will update the second scatter plot
@app.callback(Output(component_id='sub-graph', component_property='figure'),
                Input(component_id='scatter-chart', component_property='hoverData'),            # get the patient the user hovers over from the first graph
                Input(component_id='site-dropdown', component_property='value'),                # get the outcome from the first dropdown menu
                )
def update_subgraph(hoverData, site_dropdown_choice):
    subject_id = hoverData['points'][0]['customdata']                                           # get the subject id from what point the user hovers over
    subgraph_Data = data_explore.where(data_explore['Subject ID']==subject_id)                  # get the data for only that subject 
    return create_subgraph(subgraph_Data,site_dropdown_choice,subject_id)                       # return the subgraph function that was created earlier

# create the callback function for the machine learning model
@app.callback(Output(component_id='Prediction', component_property='children'),
                Input(component_id='submit-val',component_property='n_clicks'),
                Input(component_id='U_visit', component_property='value'),
                Input(component_id='U_Age', component_property='value'),
                Input(component_id='site-dropdown-sex', component_property='value'),
                Input(component_id='U_SES', component_property='value'),
                Input(component_id='U_EDUC', component_property='value'),
                Input(component_id='U_ASF', component_property='value'),
                Input(component_id='U_eTIV', component_property='value'),
                Input(component_id='U_nWBV', component_property='value'),
        )
# create a function to run the machine learning model using the user input
def run_model(n_clicks,visit,age,sex,SES,EDUC,ASF,eTIV,nWBV):
        prev_nclicks = 0                                                                        # set previous number of button clicks to 0
        if n_clicks > prev_nclicks:                                                             # if the submit button is clicked run the model
                user_input = np.array([[visit, sex, age, EDUC, SES, eTIV, nWBV,  ASF]])         # create an array of the user input
                user_input = sc.transform(user_input)                                           # normalize that array based on the training data
                prediction = svm_model.predict(user_input)                                      # run the model to get a prediction
                prev_nclicks = n_clicks                                                         # set the previous number of clicks to the current number of clicks
                return f'Prediction: {prediction}'                                              # return the model

# Run the app
if __name__ == '__main__':
    app.run_server()
