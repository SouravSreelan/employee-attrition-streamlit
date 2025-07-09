import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

model = joblib.load("attrition_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🔍 Employee Attrition Predictor")

uploaded_file = st.file_uploader("📁 Upload employee data CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    drop_cols = ["EmployeeNumber", "Over18", "EmployeeCount", "StandardHours", "Attrition"]
    for col in drop_cols:
        if col in data.columns:
            data = data.drop(col, axis=1)

    for col in data.columns:
        if col in label_encoders:
            le = label_encoders[col]
            data[col] = data[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    st.subheader("📊 Uploaded Data")
    st.dataframe(data)

    predictions = model.predict(data)
    data["Attrition_Predicted"] = predictions
    data["Attrition_Predicted"] = data["Attrition_Predicted"].map({0: "No", 1: "Yes"})

    st.subheader("🧠 Prediction Results")
    st.dataframe(data[["Attrition_Predicted"]])

    st.subheader("📈 Attrition Prediction Summary")

    attr_counts = data["Attrition_Predicted"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(attr_counts, labels=attr_counts.index, autopct='%1.1f%%', startangle=90, colors=["#66b3ff", "#ff9999"])
    ax1.axis('equal')
    st.pyplot(fig1)

    if "JobRole" in uploaded_file.name or "JobRole" in data.columns:
        try:
            raw_df = pd.read_csv(uploaded_file)
            merged_df = raw_df.copy()
            merged_df["Attrition_Predicted"] = data["Attrition_Predicted"]
            if "JobRole" in label_encoders:
                inv_le = label_encoders["JobRole"]
                merged_df["JobRole"] = merged_df["JobRole"].map(lambda x: inv_le.inverse_transform([x])[0] if isinstance(x, int) else x)
            job_role_counts = merged_df.groupby("JobRole")["Attrition_Predicted"].value_counts().unstack().fillna(0)
            st.subheader("🏢 Attrition by Job Role")
            st.plotly_chart(px.bar(job_role_counts, barmode="group", title="Attrition by Job Role"))
        except Exception as e:
            st.warning(f"Could not visualize JobRole breakdown: {e}")

    st.download_button("⬇ Download Predictions", data.to_csv(index=False), "attrition_predictions.csv", "text/csv")
