# Dementia Prediction  
View the completed [Dashboard](https://dementia-prediction.herokuapp.com/)  
## Overview:
The goal of this project was to examine a dataset of patients with and without dementia and determine which features had the largest impact on patient outcome as well as build a machine learning model to predict likelihood of dementia in future patients.  

The purpose of this branch of the project is to store the files that the dashboard deploys from. The dashboard is deployed using a combination of git and heroku. The app.py file is the main code for the dashboard, it provides all of details for the layout, graphs, text, and features of the dashboard. The dementia.db file is the database for the project, it holds all the data that is being displayed on the dashboard. The requirements.txt file lists all the libraries that are necessary to run the dashboard. Heroku uses this file to set up the environment that the dashboard is run in. The procfile lists which commands are executed upon startup of the app. The .pickle file is the machine learning model for the dashboard. The assets folder stores the css file and the ico file. The css provides the formating and style for the dashboard. The ico file provides the icon for the dashboard. 

The dashboard was made using Plotly Dash. The [dashboard](https://dementia-prediction.herokuapp.com/) shows an overview of the project, an interactive comparison of features, the ANOVA test results, and an interactive machine learning model. The aim of the dashboard is to provide users with a deeper understanding of the features influence on patient outcome as well as a tool to predict the outcome of a new patient.  
Click [here](https://github.com/sspalding/Dementia-Prediction/blob/3f3d47b89e72999dbad910962c7b3988e9ae9133/app.py) to view the code behind the dashboard.  

## References:
The dataset is from [kaggle.com](https://www.kaggle.com/datasets/shashwatwork/dementia-prediction-dataset)     
Battineni, Gopi; Amenta, Francesco; Chintalapudi, Nalini (2019), “Data for: MACHINE LEARNING IN MEDICINE: CLASSIFICATION AND PREDICTION OF DEMENTIA BY SUPPORT VECTOR MACHINES (SVM)”, Mendeley Data, V1, doi: 10.17632/tsy6rbc5d4.1

## Disclaimer:  
The Dementia Prediction project is meant to be an education tool and not a medical diagnostic tool.
