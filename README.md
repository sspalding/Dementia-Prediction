# Dementia Prediction  
View the completed [Dashboard](https://dementia-prediction.herokuapp.com/)  
## Overview:
The goal of this project was to examine a dataset of patients with and without dementia and determine which features had the largest impact on patient outcome as well as build a machine learning model to predict likelihood of dementia in future patients.

Features:
- SES: Social Economic Status  
- EDUC: Education Level  
- MR Delay: Magnetic Resonance Delay
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

## Data Collection and Wrangling:
Data collection and wrangling is the first and arguably most important step in the data analysis process as it sets the stage for the rest of the project. The data for this project came from [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dementia-prediction-dataset). The csv file was read in using the Pandas library and was stored in a local database using sqlite3. Storing the data in a database as such will allow for the easy access and storing of datatables across multiple files. The dataset did have some null values, these were replaced with the average value of that column. The sex variable was converted from categorical to continous by assigning male to 1 and female to 0. All of the patients in the trial were right handed so the hand column was deleted. The patients who had gone from nondemented to demented during the time of the trial were originally labeled as converted. This labeling adds uneccessary ambiguity and does not provide any new information to the dataset. Therefore the converted label was relabeled as either demented or non-demented depending on the CDR of the patient. The wrangled data was stored in the database as a new table to be accessed by other programs.  
Click [here](https://github.com/sspalding/Dementia-Prediction/blob/3f3d47b89e72999dbad910962c7b3988e9ae9133/DataCollection%20and%20Wrangling.ipynb) to view the full data mining and wrangling process.  

## Data Exploration:
Data visualization, statistical analysis, and SQL queries were performed to gain an understanding of which features impacted patient outcome the most. From SQL queries it was learned that of the men in the trial 64.5% were in the demented group, and 42.9% were  in the nondemented group. In comparison, of the women in the trial 43.2% were in the demented group, and 68.2% were in the nondemented group. The overlap in percentages is due to some patients having converted from demented to nondemented during the time of the study. This lends the impression that sex of the patient may be a large determining feature in outcome.  
Using data visualization it was confirmed that MMSE, Group, and CDR are all very correlated, this is reasuring given that these were three measures of patient outcome that should all yeild similar information. Data visualization also revealed a potential trend between nWBV and patient outcome, with nondemented patients having higher nWBV than demented patients.  
Statistical analysis was used to confirm the outcomes shown in the SQL queeries and data visualization. ANOVA tests revealed that nWBV, education level, SES, and sex of the patient all were significantly different between the demented and nondemented groups. Correlation testing revealed that nWBV and patient sex were strongly correlated with patient outcome.   
Click [here](https://github.com/sspalding/Dementia-Prediction/blob/d0c888f2ee76d9f9862f3d1f8fd0779e988df73c/Exploratory%20Data%20Analysis.ipynb) to view the full data exploration.

## Machine Learning Model:
The inputs for the machine learning models were Sex, Educ, SES, and nWBV. The other features were not included because statistical analysis showed they did not significantly change between groups. Four different machine learning models were tested: K nearest neighbor, decision tree, support vector machine, and logistic regression. The best parameters were determined for each model using grid search. Once the best parameters for each model was found the accuracy score, f1 score, jaccard score, and confusion matrix were found for each model. These measures of model accuracy were compared to determine which model was the best for this dataset. Machine learning models are often prone to over fitting, so to further ensure that the best model was being chosen, k fold cross validation was performed. Combining all of these led to the decision to move forward with the support vector machine model, which had an average accuracy of 62% and an F1 score of 72%.  
Click [here](https://github.com/sspalding/Dementia-Prediction/blob/d0c888f2ee76d9f9862f3d1f8fd0779e988df73c/Machine%20Learning.ipynb) to view the full machine learning model analysis. 

## Reporting Results:
Reporting the results of the analysis is essential to the data analysis process. It is incredibly important to provide results to the audience in a accessible format. A dashboard was made using Plotly Dash. The dashboard is deployed using a combination of git and heroku. The [dashboard](https://dementia-prediction.herokuapp.com/) shows an overview of the project, an interactive comparison of features, the ANOVA test results, and an interactive machine learning model. The dashboard aims to provide users with a deeper understanding of the features influence on patient outcome as well as a tool to predict the outcome of a new patient.  
Click [here](https://github.com/sspalding/Dementia-Prediction/blob/3f3d47b89e72999dbad910962c7b3988e9ae9133/app.py) to view the code behind the dashboard.  

## Build Instructions  
1. Download the dataset from [kaggle.com](https://www.kaggle.com/datasets/shashwatwork/dementia-prediction-dataset)  
2. CLone the main branch of this repository  
3. Ensure the program files and dataset are saved to the same directory  
4. Run the program files in the following order: 
    1. DataCollection and Wrangling.ipynb
    2. Exploratory Data Analysis.ipynb
    3. Machine Learning.ipynb
    4. App.py

## References:
The dataset is from [kaggle.com](https://www.kaggle.com/datasets/shashwatwork/dementia-prediction-dataset)     
Battineni, Gopi; Amenta, Francesco; Chintalapudi, Nalini (2019), “Data for: MACHINE LEARNING IN MEDICINE: CLASSIFICATION AND PREDICTION OF DEMENTIA BY SUPPORT VECTOR MACHINES (SVM)”, Mendeley Data, V1, doi: 10.17632/tsy6rbc5d4.1

## Disclaimer:  
The Dementia Prediction project is meant to be an education tool and not a medical diagnostic tool. 
