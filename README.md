
# Predicting Annual Medical Expenses


#### Chris Richards
#### Practicum 1, Summer 2020
#### Regis University

## Overview
### Project goal
The goal of this project was to create various models and assess their usefullness in predicting the annual dollar amount of health care spending.  

## Resources/libraries
* Anaconda 3
* Python 3
* Jupyter Notebooks  

## Libraries: 
* Pandas - data analysis and manipulation
* numpy - array processing
* sci-kit learn - modules for machine learning
* pandas profiling - automated exploratory data analysis
* graphviz - displaying decision trees
* seaborn - data visualization library
* matplotlib - plotting
* xgboost - extreme gradient boosting module
* phik - correlation analyzer package
* statsmodels  - statistical computation and models
* scipy - scientific computation

## Data Overview
Data for this project was taken from the book, "Machine Learning with R", by Brett Lantz.  It is available for download on Kaggle.com and on github at: https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv

The data itself consists of 1,337 rows and the following features:  
* age: age of primary beneficiary
* sex: female, male
* bmi: Body mass index
* children: Number of children covered by health insurance / Number of dependents
* smoker: Smoking?
* region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
* charges: Individual medical costs billed by health insurance

## Exploratory Data Analysis (EDA)
Typical EDA activities were performed including:
* Quantitative stats
* Checking for null values and mitigating
* Pairs plot for correlations
* Checking for duplicates and mitigating
* Detecting outliers using boxplots and scatterplots
* Distributions using histograms
  
 Detailed steps can be found in the practicum_eda.ipynb notebook in this repository.
 
 ## Feature Engineering
 Feature engineering was fairly light.  An extraneous index column was removed from the intitial data set.  Categorical features were encoded using different techniques.  
 Features that were poorly correlated with the target variable, "charges", were removed and saved for later model building.  
 An additional feature, "weight category", based on the BMI categories was added as part of the EDA process.  The categories were used for analysis of the relationship between BMI categories and several other features, including "charges.  Visualizations of the analysis can be found in the EDA notebook.
 
 Detailed feature engineering steps can be found in the practicum_feature_engineering_2.ipynb notebook in this repository.  
   
 ## Models: Linear Regression and Support Vector Regression (SVR)
 The first attempts at model building for this project were performed using regression models from several libraries.  
 The models were:  
* Linear regression model (scikit-learn)
* Polynomial Regression model using SVR (scikit-learn)
* Linear regression model (statsmodels)

Data used for the model building was divided with 70% used for training and the remaining 30% for testing.  The models were then fitted using the training data set and predictions made on the test data set.  Accuracy using the R^2 scoring metric was recorded.  

Each of the models was fitted using data containing different features.  One set of data consisted of the complete feature set, another set contained only "age" and "smoker", while the third set consisted of "age", "smoker", and "bmi".

In addition, feature importance based on each model was visualized.  

Detailed steps can be found in the following notebooks in this repository:  
* practicum_linear_regression_all_features_2.ipynb
* practicum_linear_regression_age_and_smoker_2.ipynb
* practicum_linear_regression_age_bmi_smoker_2.ipynb

## Model: XGBoost
The extreme gradient boosting (XGBoost) algorithm was implemented for the second round of model building.  The XGBoost algorithm is capable of both classification and regression modeling.  This project implemented it as a regressor similar to the earlier linear and polynomial regression experiments.  Unlike the earlier models, XGBoost uses decision trees to arrive at predictions.  The trees are "boosted" in that the algorithm seeks to improve on earlier trees by learning from their mistakes.  
  
Two XGBoost models were implemented.  The first used a set of parameters of low values as a "baseline".  The second model's hyperparameters were tuned for optimizing the accuracy scoring metric.  

Data for this model followed the earlier 70/30 split of testing and training sets.  

Detailed steps can be found in the  notebooks in this repository: 
example of image link:
<img src="images/weight_cat.PNG" raw=true/>
