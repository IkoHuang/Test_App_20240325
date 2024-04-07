import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Machine Learning：Classification")

# 左側選單 (選擇資料集及分類器)
data = st.sidebar.selectbox("### Select Dataset：", ["IRIS","WINE","BREAST_CANCER"])
clf = st.sidebar.selectbox("### Select Classifier：", ["SVM","KNN","RandomForest"])

# 下載資料並取得 X、y
def loadData(dset):
    # myData = None
    if dset == "IRIS":
        myData = datasets.load_iris()
    elif dset == "WINE":
        myData = datasets.load_wine()
    else:
        myData = datasets.load_breast_cancer()

    X = myData.data     # data --> Dataset 的內容
    y = myData.target     # target --> Class Label
    yName = myData.target_names     # target_names --> Class Name
    return X,y,yName

X, y, yName = loadData(data)
st.write("### Dataset Info. (Data Volume，Attributes)：",X.shape)     # X.shape --> 顯示 Dataset 的資料總數及特徵值數量
st.write("### Dataset Class Label：",len(np.unique(y)))     # 顯示 Class Label (計算欄位數量，並把重複值移除)
st.write("### Dataset Class Name：")
for i in yName:     # 用迴圈依序顯示 Class Name
    st.write("###",i)
st.write("### Printing the First 5 Rows of the Dataset：")
st.table(X[:5])     # 0 ~ 4 Rows，並顯示所有特徵值 (Columns)

# 定義每個模型的參數設定 (Parameter)
def model(m):
    p = {}     # 參數名稱 (C、K、N、D) 為 key 值，滑動條設好的數值會存到 value
    if m == "SVM":
        C = st.sidebar.slider("Adjust Parameter C：", 0.01, 10.0, 5.0)
        p["C"] = C     # 舉例，此字典 p 回傳的內容是 {'C':5.0}，對字典的 key 值進行操作時，要使用中括號 []
    elif m == "KNN":
        K = st.sidebar.slider("Adjust Parameter K：", 1, 10, 5)
        p["K"] = K   
    else:
        N = st.sidebar.slider("Adjust n_estimators：", 10, 500, 255)
        D = st.sidebar.slider("Adjust max_depth：", 1, 100, 50)
        p["N"] = N
        p["D"] = D  
    return p

# 建立模型
p = model(clf)
def myModel(clf,p):     # 傳入模型種類，及參數設定值
    # new_clf = None
    if clf == "SVM":
        new_clf = SVC(C = p["C"])     # 舉例，建立 SVM 模型，並代入參數 C 設定值
    elif clf == "KNN":
        new_clf = KNeighborsClassifier(n_neighbors = p["K"])
    else:
        new_clf = RandomForestClassifier(n_estimators = p["N"],max_depth = p["D"])
    return new_clf

myclf = myModel(clf, p)    # 建立模型及參數設定

# 分割訓練,測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify = y)     # stratify = y --> 使訓練及測試集的資料分佈比例一致

# 進行訓練計算及預測
myclf.fit(X_train, y_train)     # 後續還要進行降維，所以先做 fit
y_pred = myclf.predict(X_test)     # fit 後對測試集進行 predict，將結果存入 y_pred (y-hat)

# 進行評分
acc = accuracy_score(y_test, y_pred)     # 比對 y 及 y-hat，計算準確率 (accuracy_score 的 return 值為 float)
st.write("### Classification Accuracy：",acc)

# PCA 降維
pca = PCA(2)     # 降至 2 維，以進行視覺化
newX = pca.fit_transform(X)     # 降維時才做 transform

# matplotlib 繪圖
fig = plt.figure(figsize = (7,5))
plt.scatter(newX[:,0], newX[:,1], c = y, alpha = 0.5)
# 繪製散點圖，x 和 y 座標的 scale 為 newX 的 Column 0 及 Column 1，散點顏色由 y 決定，散點透明度為 0.5
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.grid()
plt.show()
st.pyplot(fig)