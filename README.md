# Predicting Cardiovascular Disease

__Enter Subtitle here if any__

# Overview
Cardiovascular disease(CVD), also known as heart disease is the leading cause of death worldwide taking an estimated 17.9 million lives each year. In the United states, one person dies every 36 seconds from cardiovascular disease.From 2014 to 2015, the Heart disease costs in United States was about $219 billion. Therefore, it is very important to understand the correlations with the risk factors. In this project I will explore the dataset to understand the correlation and use a machine learning model to predict heart disease. Also, I will compare among different models and present the accuracy of each model.

The model accuracy must be sufficiently high(at least 95\%). Because it is not wise to predict if a person has heart disease based on a model that is not accurate enough. Therefore, if our model prediction accuracy is fairly accurate, we can say that the project is successful. 

There are several factor like High blood pressure, high blood cholesterol smoking, glucose, obesity etc. are key risk factors for CVD. In, this project, I have explored the dataset and tried to find the correlation among them. Whatever the accuracy is, We can still gain some insight by just exploring the data which is why this problem is interesting.

- Tell us who might be interested in your project.
Healthcare professionals, researchers who are involved in biomedical research, insurance companies and researcher who are developing point of care testing kit might be interested in this project.

- What has already been done on the problem you are working on?
Although similar works like finding correlation among the features and fitting different machine learning model etc.have already been done, sufficient accuracy is yet to be achieved.


# Getting the data

Who collected the original data.
The data is obtained  from data repository of Svetlana Ulianova in kaggle.

When is the data collected?
The data was collected in Jan 2019.

The data is stored in a csv file with a delimiter of ';'.

It has 70000 rows and 13 columns. There is no missing value in the dataset.

The size of the data is 2.81 MB
The data can be found in this [link](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset)

This is a supervised learning problem. Our target column is the 13th column named 'cardio' which is important. Besides, age, cholesterol etc.these columns are also important

# EDA

import pandas as pd
df = pd.read_csv('cardio.csv',delimiter=';')
df.head()

df.shape

df.info()

df.describe()

# Preparing Data
I have converted the  column 'age' to 'years' and dropped the 'id' column. 


X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values

# Modeling

I have used Random Forest Classifier, KNearest neighbor and Decision Tree classifier. Only two are presented here. 
I think at least 90% accuracy should be a good baseline.
I will focus on the accuracy of the models. Because, accuracy should be the first priority. We have to be carefull about it as this deals with human lives.

## Decision tree model
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
#Decision tree
dt = DecisionTreeClassifier(random_state=0)
dt_scores = cross_val_score(dt, X, Y, cv=5)
print(dt_scores.mean())

## KNearest Neighbors
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(20)
score=cross_val_score(knn_classifier,X,Y,cv=5)
Accuracy = score.mean()
Accuracy

# Fine Tune 

dec_scores =[]
for x in range(1,15):
    dec = DecisionTreeClassifier(max_depth=x)
    score = cross_val_score(dec, X, Y, cv=5)
    dec_scores.append(score.mean())

import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
plt.plot([x for x in range(1,15)],dec_scores)
for i in range(1,15):
    plt.text(i,dec_scores[i-1].round(3),(i,dec_scores[i-1].round(3)))
    plt.xticks([i for i in range(1, 15)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('Classifier scores')

# Solution
Our KNN accuracy is 52% and Decision tree accuracy was 58%.
By fine tuning our model, we get the model accuracy of 72.6% at max depth of 3 which is not good for a sensitive issue like this. Therefore, we can not say that our project can successfully predic heart disease.
There are other features and factors that are not considered in this project.If the number of feature increses, our model accuracy may increase.  
