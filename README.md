# 🧠 Employee Attrition Prediction App

A smart web app that predicts whether an employee is likely to leave a company based on HR data.  
Built with **Python**, **Machine Learning**, and **Streamlit** — just upload your CSV and get instant results!

---

## 🚀 What This App Does

✅ Predicts employee attrition (Yes/No) using real-world HR data  
✅ Upload your own employee dataset (CSV)  
✅ Shows interactive charts and insights  
✅ Lets you download the predictions  
✅ Runs in the browser — no setup needed!

---

## 🌐 Live Demo

🔗 [Click to Try the App]([https://yourusername-employee-attrition.streamlit.app](https://employee-attrition-app-t8tcpjf4xvsrrwd5zuxekn.streamlit.app/))

---

## 📦 How to Run Locally

```bash
# 1. Clone this repo
git clone https://github.com/SouravSreelan/mployee-attrition-streamlit
cd mployee-attrition-streamlit

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (if not already)
python train_model.py

# 4. Run the app
streamlit run main.py

---

## 📥 Example Input (CSV Format)

Make sure your uploaded CSV looks like this:

```csv
Age,Department,JobRole,OverTime,MonthlyIncome,YearsAtCompany
35,Sales,Sales Executive,Yes,5000,5
42,Research & Development,Laboratory Technician,No,4200,7
29,Human Resources,Human Resources,Yes,3900,2

