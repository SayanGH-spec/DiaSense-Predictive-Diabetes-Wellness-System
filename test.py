import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pyttsx3

def speak_intro():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    female_voice = None
    for voice in voices:
        if "female" in voice.name.lower():
            female_voice = voice
            break
    if female_voice:
        engine.setProperty("voice", female_voice.id)
    engine.setProperty("rate", 150)
    engine.say("Hi, I am DiaSense, a diabetes predictive wellness system. Here, I can help you determine whether you are diabetic or not. Let's begin.")
    if engine._inLoop:
        engine.endLoop()
    engine.runAndWait()

def speak_outcome(outcome):
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    female_voice = None
    for voice in voices:
        if "female" in voice.name.lower():
            female_voice = voice
            break
    if female_voice:
        engine.setProperty("voice", female_voice.id)
    engine.setProperty("rate", 150)
    engine.say(outcome)
    if engine._inLoop:
        engine.endLoop()
    engine.runAndWait()

diabetes_df = pd.read_csv("diabetes.csv")
X = diabetes_df.drop(columns=['Outcome'])
y = diabetes_df['Outcome']
feature_list = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))

def predict_diabetes(input_data):
    rf_prediction = rf_model.predict(input_data)
    svm_prediction = svm_model.predict(input_data)
    diabetes_probability = svm_model.predict_proba(input_data)[:, 1][0]
    if rf_prediction == 1 or svm_prediction == 1:
        return "You might have diabetes. It is recommended to consult a doctor for a checkup.", None
    else:
        return f"You are not predicted to have diabetes. However, your probability of getting diabetes in the future is: {diabetes_probability * 100:.2f}%", diabetes_probability

st.set_page_config(page_title="DiaSense: Predictive Diabetes Wellness", page_icon=":hospital:", layout="wide")

st.markdown(
    """
    <style>
    .css-1aumxhk {
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .st-bw {
        background-color: #ffffff;
    }
    .st-br {
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("DiaSense: Predictive Diabetes Wellness")
speak_intro()

st.sidebar.header("User Input")
user_input = {}
for feature in feature_list:
    user_input[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)

if st.sidebar.button("Predict", key="predict_button"):
    input_data = np.array([[user_input[feature] for feature in feature_list]])
    prediction_result, diabetes_probability = predict_diabetes(input_data)
    st.write(prediction_result)
    speak_outcome(prediction_result)
    if diabetes_probability is not None:
        fig = px.bar(x=['Diabetic', 'Non-Diabetic'], y=[diabetes_probability, 1 - diabetes_probability], 
                     color=['Diabetic', 'Non-Diabetic'], labels={'x': 'Outcome', 'y': 'Probability'},
                     title='Probability of Getting Diabetes in the Future', barmode='group')
        st.plotly_chart(fig)

st.header("Model Comparison")
st.subheader("Accuracy Comparison")

accuracy_data = {
    'Model': ['Random Forest', 'Support Vector Machine'],
    'Accuracy': [rf_accuracy, svm_accuracy]
}
accuracy_df = pd.DataFrame(accuracy_data)

fig = px.bar(accuracy_df, x='Model', y='Accuracy', color='Model', labels={'Accuracy': 'Accuracy Score', 'Model': 'Model'})
st.plotly_chart(fig)

st.markdown("---")
st.write("Made with ❤️ by group 21")

st.set_option('deprecation.showPyplotGlobalUse', False)
