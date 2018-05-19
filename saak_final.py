import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

df1 = pd.read_csv('Set1.csv', delimiter = ",", header = None )
df2 = pd.read_csv('Set2.csv', delimiter = ",", header = None )
df3 = pd.read_csv('Set3.csv', delimiter = ",", header = None )
df4 = pd.read_csv('Set4.csv', delimiter = ",", header = None )
df5 = pd.read_csv('Set5.csv', delimiter = ",", header = None )
df6 = pd.read_csv('Set6.csv', delimiter = ",", header = None )

frames = [df1, df2, df3, df4, df5, df6]
X_train = pd.concat(frames)

print("Reading TrainLabels.csv file")
y = pd.read_csv('TrainLabels.csv', delimiter = ",",header= None)
print("Reading test")
X_test = pd.read_csv('TestData.csv', delimiter = ",",header= None)
y_test = pd.read_csv('TestLabels.csv', delimiter = ",",header= None)
print("Completed reading .csv files")

dimension = [32,64,128]
for n in dimension:
    
    print("Starting PCA")
    from sklearn.decomposition import PCA
    pca = PCA(n_components = n)
    pca.fit(X_train)
    X=pca.transform(X_train)
    X_test = pca.transform(X_test)


    print("Starting SVM")
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    clf = SVC()
    clf.fit(X,y)
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X)
    print(" Training accuracy SVM : ")
    print(n)
    print(accuracy_score(y,y_pred_train))

    print(" Testing accuracy SVM : ")
    print(n)
    print(accuracy_score(y_test,y_pred))

    ###RFC
    print("Starting RFC")
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X,y)
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X)
    print(" Training accuracy RFFC : ")
    print(n)
    print(accuracy_score(y,y_pred_train))

    print(" Testing accuracy RFC : ")
    print(n)
    print(accuracy_score(y_test,y_pred))
