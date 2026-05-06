import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Titanic App", layout="wide")

df = pd.read_csv("train.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.title("Titanic Machine Learning App")

tab1, tab2, tab3 = st.tabs(["Home", "Insights", "Prediction"])

with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.metric("Model Accuracy", f"{acc:.2f}")

with tab2:
    st.subheader("Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Survival Count")
        st.write(df["Survived"].value_counts())

    with col2:
        st.write("Passenger Class Distribution")
        st.write(df["Pclass"].value_counts())

with tab3:
    st.subheader("Make Prediction")

    pclass = st.selectbox("Pclass", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 1, 80, 25)
    sibsp = st.slider("SibSp", 0, 5, 0)
    parch = st.slider("Parch", 0, 5, 0)
    fare = st.slider("Fare", 0.0, 500.0, 50.0)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])

    sex_val = le_sex.transform([sex])[0]
    embarked_val = le_embarked.transform([embarked])[0]

    input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])

    if st.button("Predict"):
        result = model.predict(input_data)[0]

        if result == 1:
            st.success("Survived")
        else:
            st.error("Not Survived")
