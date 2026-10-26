# STEP 77: FAKE SOCIAL MEDIA ACCOUNT DETECTION APP

import streamlit as st
import pandas as pd
import joblib

# Load the saved machine learning model
model = joblib.load("fake_account_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Set up the application page
st.set_page_config(
    page_title="Fake Account Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fake Social Media Account Detection")
st.write(
    "Enter the account details below to get a prediction "
    "from our machine learning model."
)

st.info(
    "This tool provides a prediction, not proof that an "
    "account is fake or genuine."
)

# Collect account details
with st.form("account_form"):
    st.subheader("Account Information")

    profile_pic = st.selectbox(
        "Does the account have a profile picture?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    username_length = st.number_input(
        "Numbers-to-length ratio in username",
        min_value=0.0, max_value=1.0, value=0.25
    )

    fullname_words = st.number_input(
        "Number of words in full name",
        min_value=0, max_value=100, value=2
    )

    fullname_length = st.number_input(
        "Numbers-to-length ratio in full name",
        min_value=0.0, max_value=1.0, value=0.20
    )

    name_username = st.selectbox(
        "Is the name the same as the username?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    description_length = st.number_input(
        "Description length",
        min_value=0, value=100
    )

    external_url = st.selectbox(
        "Does the account have an external URL?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    private = st.selectbox(
        "Is the account private?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    posts = st.number_input(
        "Number of posts", min_value=0, value=50
    )

    followers = st.number_input(
        "Number of followers", min_value=0, value=500
    )

    follows = st.number_input(
        "Number of accounts followed", min_value=0, value=300
    )

    submitted = st.form_submit_button("Check Account")

# Make a prediction when the form is submitted
if submitted:
    account_data = {
        "profile pic": profile_pic,
        "nums/length username": username_length,
        "fullname words": fullname_words,
        "nums/length fullname": fullname_length,
        "name==username": name_username,
        "description length": description_length,
        "external URL": external_url,
        "private": private,
        "#posts": posts,
        "#followers": followers,
        "#follows": follows
    }

    account_df = pd.DataFrame([account_data])
    account_df = account_df[feature_names]

    prediction = model.predict(account_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning("The model predicts this may be a FAKE account.")
    else:
        st.success("The model predicts this may be a GENUINE account.")

    st.caption(
        "Predictions may be incorrect. Review other evidence "
        "before making decisions about an account."
    )