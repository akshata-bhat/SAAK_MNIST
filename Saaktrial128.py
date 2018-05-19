import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


df1 = pd.read_csv('Set1.csv', delimiter = ",", header = None )
print("-------------------SET 1 READ-----------------------")

df2 = pd.read_csv('Set2.csv', delimiter = ",", header = None )
print("-------------------SET 2 READ-----------------------")

df3 = pd.read_csv('Set3.csv', delimiter = ",", header = None )
print("-------------------SET 3 READ-----------------------")

df4 = pd.read_csv('Set4.csv', delimiter = ",", header = None )
print("-------------------SET 4 READ-----------------------")

df5 = pd.read_csv('Set5.csv', delimiter = ",", header = None )
print("-------------------SET 5 READ-----------------------")

df6 = pd.read_csv('Set6.csv', delimiter = ",", header = None )
print("-------------------SET 6 READ-----------------------")

frames = [df1, df2, df3, df4, df5, df6]

X_train = pd.concat(frames)
print("-------------------CONCATINATION DONE-----------------------")


y = pd.read_csv('Trainlabels.csv', delimiter = ",",header= None)
print("-------------------TRAIN LABELS DONE-----------------------")

X_test = pd.read_csv('Testdata.csv', delimiter = ",",header= None)
print("-------------------TEST DATA DONE-----------------------")

y_test = pd.read_csv('Testlabels.csv', delimiter = ",",header= None)
print("-------------------TEST LABELS DONE-----------------------")


dimension = [1000]
for n in dimension:
    
    print("Starting PCA")
    from sklearn.decomposition import PCA
    pca = PCA(n_components = n)
    pca.fit(X_train)
    X=pca.transform(X_train)
    X_test1 = pca.transform(X_test)


    print("Starting SVM")
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    clf = SVC()
    clf.fit(X,y)
    y_pred = clf.predict(X_test1)
    y_pred_train = clf.predict(X)
    print(" Training accuracy Svm : ")
    print(n)
    print(accuracy_score(y,y_pred_train))

    print(" Testing accuracy svm : ")
    print(n)
    print(accuracy_score(y_test,y_pred))

    ###RFC
    print("Starting RFC")
    from sklearn.ensemble import RandomForestClassifier
    clf1 = RandomForestClassifier()
    clf1.fit(X,y)
    y_pred1 = clf1.predict(X_test1)
    y_pred_train1 = clf1.predict(X)
    print(" Training accuracy rfc : ")
    print(n)
    print(accuracy_score(y,y_pred_train1))

    print(" Testing accuracy rfc : ")
    print(n)
    print(accuracy_score(y_test,y_pred1))
