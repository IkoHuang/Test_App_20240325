import streamlit as st
import numpy as np
import pandas as pd

st.title("Iko Huang")
st.write("Iko Huang")
st.header("Iko Huang")
st.text("Iko Huang")

st.write("# Iko Huang")
st.write("## Iko Huang")

a = np.array([[10,20,30],[100,200,300]])
st.write(a)
st.table(a)

b = pd.DataFrame([[10,20,30],[100,200,300]])
st.write(b)
st.table(b)

name = "Iko Huang"
st.write(name, b[0])
st.info(name)

# 核取方塊 1
st.write("### CheckBox 1")
re1 = st.checkbox("Hamburger")
if re1:
    st.info("like Hamburger")
else:
    st.info("don't like Hamburger")
re2 = st.checkbox("Sport")
if re2:
    st.info("like Sport")
else:
    st.info("don't like Sport")

# 核取方塊 2
st.write("### CheckBox 2")
res = st.columns(3)
with res[0]:
    c1 = st.checkbox("A")
    if c1:
        st.info("A")
with res[1]:
    c2 = st.checkbox("B")  
    if c2:
        st.info("B")
with res[2]:
    c3 = st.checkbox("C") 
    if c3:
        st.info("C") 
 
# 選項按鈕 1
st.write("### SelectButton 1")
re3 = st.radio("Gender：", ("Male","Female","None"), index = 0)
st.write(re3)

re31 = st.radio("Gender：", ("Male","Female","None"), index = 1, key = 'a')
st.write(re31)

# 選項按鈕 2
st.write("### SelectButton 2")
col1, col2 = st.columns(2)
with col1:
    num1 = st.number_input("Please enter any integer")
with col2:
    num2 = st.number_input("Please enter any integer", key = "num2")
re4 = st.radio("Calculate：",("＋",'－',"＊","／"),key = "b")
if re4 == "＋":
    st.write("{}+{}={}".format(num1, num2, num1 + num2))
elif re4 == "－":
    st.write("{}-{}={}".format(num1, num2, num1 - num2))
elif re4 == "＊":
    st.write("{}*{}={}".format(num1, num2, num1 * num2))
elif re4 == "／":
    st.write("{}/{}={}".format(num1, num2, num1 / num2))

# 滑桿 
st.write("### Slider")
re5 = st.slider("Quantity：", 1.0, 20.0, step = 1.0)
st.info(re5)

# 下拉選單  
st.write("### SelectBox")
re6 = st.selectbox("Classifier", ("KNN", "SVC", "TREE"))
st.info(re6)