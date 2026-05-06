import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import plotly.express as px

st.set_page_config(page_title="Titanic Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        background-color: #0e1117;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

st.title("Titanic Machine Learning Dashboard")

tab1, tab2, tab3 = st.tabs(["Home", "Insights", "Prediction"])

with tab1:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Passengers", len(df))
    col2.metric("Model Accuracy", f"{acc:.2f}")
    col3.metric("Missing Age", df["Age"].isna().sum())

    st.dataframe(df.head())

with tab2:
    st.subheader("Insights")

    col1, col2 = st.columns(2)

    with col1:
        survival_counts = df["Survived"].value_counts()

        fig1 = px.pie(
            values=survival_counts.values,
            names=["Not Survived", "Survived"],
            title="Survival Rate",
            color_discrete_sequence=["#ff4b4b", "#00cc96"]
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        class_counts = df["Pclass"].value_counts().sort_index()

        fig2 = px.pie(
            values=class_counts.values,
            names=["Class 1", "Class 2", "Class 3"],
            title="Passenger Class Distribution",
            color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"]
        )

        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(
        df,
        x="Age",
        nbins=30,
        title="Age Distribution",
        color_discrete_sequence=["#636EFA"]
    )

    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Prediction")

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
