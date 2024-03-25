import streamlit as st
import joblib

# 頁面標題
st.title("IRIS 品種預測")

# 載入模型
svm = joblib.load("app/svc_model.joblib")
knn = joblib.load("app/knn_model.joblib")
lr = joblib.load("app/lr_model.joblib")
rf = joblib.load("app/rf_model.joblib")

#左側：選單欄
name = st.sidebar.selectbox("### 請選擇分類模型：", ("KNN", "LogisticRegression", "SVM", "RandomForest"))

if name == "KNN":
    model = knn
elif name == "LogisticRegression":
    model = lr
elif name == "SVM":
    model = svm
elif name == "RandomForest":
    model = rf

#右側：接收資料並預測
s1 = st.slider("花萼長度：",3.5, 8.5, 4.0)
s2 = st.slider("花萼寬度：",1.5, 5.5, 3.0)
s3 = st.slider("花瓣長度：",0.6, 7.5, 4.0)
s4 = st.slider("花瓣寬度：",0.05, 3.5, 2.0)

labels = ['setosa', 'versicolor', 'virginica']
re = st.button("進行預測")

if re:
    X = [[s1,s2,s3,s4]]
    y_hat = model.predict(X)
    st.write("### 預測結果：", y_hat[0])
    st.write("### 品種名稱：", labels[y_hat[0]])
