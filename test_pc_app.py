import streamlit as st
import joblib

# 頁面標題
st.title("IRIS  Class  Predict")
st.header("Class 0 : Setosa")
st.header("Class 1 : Versicolor")
st.header("Class 2 : Virginica")

# 載入模型
svm = joblib.load("app/svc_model.joblib")
knn = joblib.load("app/knn_model.joblib")
lr = joblib.load("app/lr_model.joblib")
rf = joblib.load("app/rf_model.joblib")

# 左側選單 (選擇分類器)
classifier = st.sidebar.selectbox("### Select Classifier:",
                             ("KNN","LogisticRegression","SVM","RandomForest"))
if classifier == "KNN":
    model = knn
elif classifier == "LogisticRegression":
    model = lr
elif classifier == "SVM":
    model = svm
elif classifier == "RandomForest":
    model = rf

# 右側選單 (接收資料並預測)，滑桿的設定範圍，要參考原始資料集的特徵範圍
s1 = st.slider("sepal length:", 3.5, 8.5, 4.0)     # 原始值：min --> 4.3，max --> 7.9，設定範圍要更大及更小一點，要能把 min 及 max 值包住，4.0 為預設起始點
s2 = st.slider("sepal width:", 1.5, 5.5, 3.0)     # 原始值：min --> 2.0，max --> 4.4，預設起始點 3.0
s3 = st.slider("petal length:", 0.6, 7.5, 4.0)     # 原始值：min --> 1.0，max --> 6.9，預設起始點 4.0
s4 = st.slider("petal width:", 0.05, 3.5, 2.0)     # 原始值：min --> 0.1，max --> 2.5，預設起始點 2.0

labels = ["Setosa","Versicolor","Virginica"]
act = st.button("Predict Now")

if act:
    X = [[s1,s2,s3,s4]]     # 把滑桿設定好的二維資料，送進模型進行預測
    y_hat = model.predict(X)
    st.write("### Predict Label : ", y_hat[0])     # y_hat 結果會是一維的 list，但 Predict Label 會在三個結果中取一個，所以要加上 [0] 這個 index 值
    st.write("### IRIS Class : ", labels[y_hat[0]])