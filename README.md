<div align="center">

# 🏦 Bank Risk Controller Systems

</div>

## 📈 Problem Statement

Banks and financial institutions face significant financial risks due to loan defaults. Accurately predicting which customers are likely to default on their loans is crucial for effective risk management.
This project addresses the challenge of developing a robust and reliable system that can predict the probability of loan default based on historical customer data. By building such a system, we aim to provide a valuable tool for banks to make more informed lending decisions, mitigate potential losses, and optimize their risk management strategies.

---
## 🎯 Objective:

- The core task: Predicting loan default.
- The goal: Providing a tool for risk management and better decision-making for banks.
  
---
## 🔍 Dataset Overview

The project involves two datasets: **loan_data** and **data_dictionary**. 

- The **loan_data Dataset** comprises 14,13,701 entries, each detailing an individual client loan details. This includes information such as `NAME_CONTRACT_TYPE_x`, `CODE_GENDER`, `FLAG_OWN_CAR`, `CNT_CHILDREN`, `AMT_INCOME_TOTAL` and `AMT_CREDIT_x`(pricing details), as well as `TARGET` (which needs to be predicted). This dataset offers a thorough view of loan details, covering client details and loan details. 

- The **data_dictionary Dataset** consists of 161 entries that describe the columns of the loan_data. This dataset provides detailed insights into the composition of each column and the amount of importance required.
---

## 💡 Business Use Cases

This loan default prediction model enables several key business applications for banks:

- ***Risk-Aware Loan Approvals:*** Facilitates more informed lending decisions by quantifying the risk of applicant default.
- ***Risk-Based Customer Segmentation:*** Allows for the categorization of customers based on their default probability to tailor financial products.
- ***Early Fraud Indicators:*** Can help identify potentially fraudulent loan applications based on risk patterns.

## 🛠️ Approach

### I. Data Preprocessing 

**Data Cleaning** ensures the dataset's accuracy and consistency through:

- **Handling Missing Data**:
  - Detected missing values.
  - Replaced missing values using mean, median and mode.

- **Removing Inconsistent Data**:
  - Checked for format consistency and valid ranges.
  - Fixed inconsistencies, such as standardizing text and correcting typos.
 
### II. Exploratory Data Analysis (EDA)

**Exploratory Data Analysis (EDA)** discovers patterns, relationships, and anomalies in the data.

- 

### Handling Outliers with Log Transformation



#### i) Feature Engineering

Created new variables from the raw data to improve the model’s performance, such as:


#### ii) Model Selection

Model Selection involves choosing the most suitable model for our loan data:



#### iii) Model Training

Model Training involves fitting the chosen model to historical loan data:

- Split the data into training and test sets to evaluate model performance. 
- Trained the model on the training set by adjusting parameters to minimize prediction errors.
- Optimized model performance by tuning hyperparameters using techniques like cross-validation or grid search.

### iv) 📊 Model Evaluation











