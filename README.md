# 🇮🇳 India-GDP-Prediction

## 📌 Project Title
India GDP Prediction Using Polynomial Regression & Scenario Analysis (1960–2050)

---

## 📊 Overview
This project analyzes historical GDP data of India (1960–2021) and predicts future GDP trends using machine learning techniques.

Short-term predictions (2026–2040) are generated using Polynomial Regression, while long-term projections (2041–2050) are estimated using scenario-based growth analysis.

---

## 🎯 Objective
- Analyze historical GDP growth of India  
- Build a machine learning model for prediction  
- Forecast GDP for future years  
- Perform long-term scenario analysis  

---

## ⚙️ Tech Stack
- **Programming Language:** Python  
- **Libraries:**
  - pandas → Data cleaning & processing  
  - numpy → Numerical operations  
  - matplotlib → Data visualization  
  - scikit-learn → Machine Learning (Polynomial Regression)  
- **Tool:** Visual Studio Code  

---

## 📁 Dataset
- **File:** `India_GDP_1960-2025.csv`  
- Historical GDP data from **1960 to 2021**  
- Extended data (2022–2025) added programmatically  

### 📌 Columns:
- Year  
- GDP (Billion USD)  
- Growth Percentage  

---

## 🔧 Methodology

### 1. Data Preprocessing
- Removed commas and converted GDP to numeric format  
- Handled missing values  
- Sorted data chronologically  
- Corrected inconsistent values (e.g., 1960 growth)  
- Automatically extended dataset (2022–2025)  

---

### 2. Model Building
- Used **Polynomial Regression (degree = 2)**  
- Pipeline includes:
  - PolynomialFeatures  
  - StandardScaler  
  - LinearRegression  

---

### 3. Prediction Strategy

#### 📈 Short-Term Prediction (2026–2040)
- Based on trained ML model  
- Captures historical GDP trends  

#### 🔮 Long-Term Scenario (2041–2050)
- Based on assumed **5% annual growth rate**  
- Provides realistic future projection  

---

## 📊 Visualization
- Scatter plot for actual GDP  
- Smooth regression curve  
- Predicted future points  
- Dashed line for scenario analysis  

---

## 🚀 Output
- GDP predictions from **2026 to 2040**  
- Scenario projections from **2041 to 2050**  
- Combined graph showing:
  - Actual vs Predicted vs Scenario  

---

## 📌 Key Highlights
✔ Automated data preprocessing  
✔ Machine Learning-based prediction  
✔ Scenario-based long-term analysis  
✔ Clean and professional visualization  

---

## 🧠 Conclusion
The model effectively predicts short-term GDP trends using historical data.  
For long-term forecasting, scenario analysis is applied due to economic uncertainties.

---

## 📷 Output Graph
![GDP Graph](GDP_Output)


---

## 👨‍💻 Author
Vivek Manjhi
