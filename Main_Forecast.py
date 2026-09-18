#!/usr/bin/env python            
from time import sleep
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import openpyxl
from datetime import date

def main():
    data_xls = pd.read_excel('data.xlsx', 'Sheet1', index_col=None)
    data_xls.to_csv('data.csv', encoding='utf-8', index=False)

    #importing dataset
    dataset=pd.read_csv("data.csv")
    print(dataset.head())

    # standardize all the data to same scale
    lab=LabelEncoder()
    dataset.iloc[:,0]=lab.fit_transform(dataset.iloc[:,0])
    dataset.iloc[:,1]=lab.fit_transform(dataset.iloc[:,1])
    dataset.iloc[:,2]=lab.fit_transform(dataset.iloc[:,2])
    dataset.iloc[:,3]=lab.fit_transform(dataset.iloc[:,3])
    dataset.iloc[:,4]=lab.fit_transform(dataset.iloc[:,4])
    dataset.iloc[:,5]=lab.fit_transform(dataset.iloc[:,5])
    dataset.iloc[:,6]=lab.fit_transform(dataset.iloc[:,6])
    dataset.iloc[:,7]=lab.fit_transform(dataset.iloc[:,7])
    dataset.iloc[:,8]=lab.fit_transform(dataset.iloc[:,8])
    dataset.iloc[:,9]=lab.fit_transform(dataset.iloc[:,9])
    dataset.iloc[:,10]=lab.fit_transform(dataset.iloc[:,10])
    dataset = dataset.infer_objects()

    print ("Dataset (after nomalized): ")
    print(dataset.head())
    #deviding data into dependant and independant sets
    x = dataset[['Day','Month','Year']]
    y = dataset['heat']
    z = dataset['wet']
    dataset.hist(figsize = (30, 30))
    plt.savefig("dataset.png")
    plt.show()
    
    #deviding data into training and testing sets
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
    x_train,x_test,z_train,z_test=train_test_split(x,z,test_size=0.20,random_state=42)
    
    #importing decision tree model and fitting training data to it
    classifier_y=DecisionTreeClassifier(criterion='entropy',random_state=0, max_depth = 3)
    clf = classifier_y.fit(x,y)
    tree.plot_tree(clf)
    plt.show()
   #predicting values for x_test and compairing result with y_test
    print("Heat:-")
    y_pred=classifier_y.predict(x_test)
    print("predicted values:-")
    print(y_pred.astype(int))
    y_test_arr=np.array(y_test)
    print("original values:-")
    print(y_test_arr)
     
    classifier_z=DecisionTreeClassifier(criterion='entropy',random_state=0, max_depth = 3)
    classifier_z.fit(x,z)
    clf = classifier_z.fit(x,z)
    tree.plot_tree(clf)
    plt.show()
    #cheaking accuracy of our model
    accuracy=accuracy_score(y_test,y_pred)
    print("Accuracy :{}%".format(accuracy*100))

    #predicting values for x_test and compairing result with y_test
    print("Wet:-")
    z_pred=classifier_z.predict(x_test)
    print("predicted values:-")
    print(z_pred.astype(int))
    z_test_arr=np.array(z_test)
    print("original values:-")
    print(z_test_arr)
    accuracy=accuracy_score(z_test,z_pred)
    print("Accuracy :{}%".format(accuracy*100))
    
    # 
    now = date.today()
    filepath = "forecast-"+str(now)+".xlsx"
    wb = openpyxl.Workbook()
    wb.save(filepath)
    print("Predict Weather Data for ")
    idx = 0
    weatherdata = []
    filename = filepath
    wb = openpyxl.load_workbook(filename=filename)
    sheet = wb['Sheet']
    new_row = ['Day','Month','Year','Heat','Wet']
    print(new_row)
    sheet.append(new_row)
    while idx < 31:
        # print('Predict weather data ' + str(idx))
        year = date.today().year
        yr = year - 1999
        if (date.today().day == 31):
            year += 1
            yr += 1
        y_pred = classifier_y.predict([[idx, 0, yr]])
        # print(y_pred.astype(int))
        z_pred = classifier_z.predict([[idx, 0, yr]])
        # print(z_pred.astype(int))
        tmp_d = idx + 1
        tmp_m = 8
        tmp_y = year
        tmp_h = y_pred.astype(int)
        tmp_w = z_pred.astype(int)
        if (int(tmp_h) == 0):
            tmp_h_char = "NO"
        else:
            tmp_h_char = "YES"
        if (int(tmp_w) == 0):
            tmp_w_char = "NO"
        else:
            tmp_w_char = "YES"
        new_row = [int(tmp_d), int(tmp_m), int(tmp_y), tmp_h_char, tmp_w_char]
        sheet.append(new_row)
        weatherdata.append(new_row)
        print(new_row)
        idx += 1

    wb.save(filename)
    
main()

