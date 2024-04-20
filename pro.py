import streamlit as st
import numpy as np
import pandas as pd
st.title('Diabetes Detection Web App')
df = pd.read_csv("C:\\Users\\AtulKumar\\Downloads\\diabetes.csv")
st.write(df.head(3))
def feature_input():
    Pregnancies=st.number_input("enter pregnency",0,10)
    Glucose= st.number_input("enter glucose",0,200)
    BloodPressure = st.number_input("enter blood pressure",0,200)
    SkinThickness = st.number_input("enter the skin thickness",0,100)
    Insulin = st.number_input("ener the insuline",0,200)
    BMI = st.number_input("enter the bmi ",0,200)
    DiabetesPedigreeFunction = st.number_input("enter diabetes pedegree function",0,100)
    Age = st.number_input("enter the age",0,110)
    data= {
        'Pregnancies' :Pregnancies ,
        'Glucose' : Glucose,
        'BloodPressure' : BloodPressure,
        'SkinThickness' : SkinThickness,
        'Insulin' : Insulin,
        'BMI' : BMI,
        'DiabetesPedigreeFunction' : DiabetesPedigreeFunction,
        'Age' : Age,
        }
    feature = pd.DataFrame(data,index=[0])
    return feature

df4 = feature_input()

df1=df[~(df.BloodPressure==0)]
df2=df1[~(df1.Glucose==0)]
X=df2.drop('Outcome',axis=1)

Y=df2.Outcome

from sklearn import  linear_model

classifier= linear_model.LogisticRegression()  
classifier.fit(X, Y)
pr=classifier.predict(df4)  
if (pr ==0):
    st.write("you don't have diabetes")
else:
    st.write("you have diabetes")


