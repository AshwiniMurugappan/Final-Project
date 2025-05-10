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

- Disribution of Contract Types
- Distribution of Loan Applicants based on Owning a Car
- Distribution of Loan Applicants based on Owning a Reality
- Income Distribution of Applicants
- Distribution of Applicant House Type
- Occupation of Loan Applicants
- Education of Loan Applicants
- Coorelation Heatmap

### Handling Outliers with Log Transformation

In this project, we addressed the issue of outliers present in numerical features such as AMT_INCOME_TOTAL and AMT_CREDIT by applying a logarithmic transformation using the numpy.log1p() function.
Outliers can significantly skew the distribution of data and negatively impact the performance of machine learning models. Logarithmic transformation is a common technique used to:

- ***Reduce the impact of extreme values:*** It compresses the range of values, bringing high values closer to the rest of the distribution.
- ***Make the distribution more normal:*** Many real-world datasets exhibit skewed distributions, and log transformation can help to make them more symmetrical, which can be beneficial for some models.

### III. Model Training and Testing

This section outlines the process of training and evaluating the loan default prediction model.

#### i) Feature Engineering

Created new variables from the raw data to improve the model’s performance, such as:
- **Age Group**: Extracted the age of the applicant from the loan data to capture age and categorized into groups.
- **Children Category**: Extracted the number of children from the loan data of the applicant for categorizing the same.

#### ii) Model Selection

Model Selection involves choosing the most suitable model for our loan data from:

a) Logistic Regression
b) Decision Tree Classifier
c) Random Forest Classifier
d) XGBoost Classifier

#### iii) Model Training

Model Training involves fitting the chosen model to historical loan data:

- Split the data into training and test sets to evaluate model performance. 
- Trained the model on the training set by adjusting parameters to minimize prediction errors.
- Optimized model performance by tuning hyperparameters using techniques.

### iv) 📊 Model Evaluation

After training, the model's performance was evaluated on the testing set to assess its ability to predict loan defaults on unseen data. We used several key evaluation metrics relevant to binary classification problems, especially those with potential class imbalance:

- ***Accuracy:*** The overall percentage of correctly classified instances.
- ***Precision:*** The proportion of correctly predicted defaulters out of all instances predicted as defaulters. High precision indicates that when the model predicts a default, it is likely to be correct.
- ***Recall (Sensitivity):*** The proportion of correctly predicted defaulters out of all actual defaulters. High recall indicates that the model is good at identifying actual defaulters.
- ***F1-Score:*** The harmonic mean of precision and recall, providing a balanced measure of the model's performance, especially useful when dealing with imbalanced classes.   
- ***Area Under the Receiver Operating Characteristic Curve (AUC-ROC):*** This metric measures the model's ability to distinguish between the positive (default) and negative (non-default) classes across various classification thresholds. A higher AUC-ROC indicates better discriminatory power.   

#### a) Logistic Regression:

Logistic Regression, a linear model, served as a baseline for comparison. While interpretable, it often struggles to capture complex non-linear relationships in the data.
Our initial results with Logistic Regression yielded moderate accuracy but tended to have lower recall in identifying potential defaulters. This suggested that the model might be too simplistic for the underlying patterns in the data.

#### b) Decision Tree Classifier:

The Decision Tree Classifier, a non-linear model, can capture intricate relationships. However, single decision trees are prone to overfitting the training data, potentially leading to poor generalization on unseen data.
While the Decision Tree achieved good accuracy on the training set, its performance on the testing set was less promising, exhibiting a noticeable drop and indicating overfitting.

#### c) Random Forest Classifier:

The Random Forest Classifier, an ensemble method based on multiple decision trees, aims to mitigate the overfitting issue of single trees by averaging their predictions. It also provides feature importance estimates and is generally robust.
The Random Forest model achieved comparable or slightly better performance than other on the testing set across most key metrics. It demonstrated a good balance between precision and recall and exhibited strong generalization capabilities. Furthermore, it proved to be less sensitive to hyperparameter tuning compared to Decision tree and XGBoost in our specific context.

#### d) XGBoost Classifier:

XGBoost (Extreme Gradient Boosting) is a powerful gradient boosting algorithm known for its high performance and scalability. It often outperforms other tree-based methods.
Our XGBoost model showed significant improvement over Logistic Regression and the single Decision Tree, achieving higher accuracy, precision, recall, and AUC-ROC. It demonstrated a better ability to balance the trade-off between correctly identifying defaulters and minimizing false positives.

### 🧩 Model Comparison:

|Model	  |Model Name	              |accuracy	    |precision	  |recall	      |F1score       |
|---------|-------------------------|-------------|-------------|-------------|------------|
|Model 1	|Logistic Regression	    |58.6167109	  |58.61922198	|58.6167109	  |58.61284885 |
|Model 2	|Decision Tree Classifier	|92.21662727	|92.24373795	|92.21662727	|92.21530872 |
|Model 3	|Random Forest Classifier	|95.92265414	|95.936989	  |95.92265414	|95.92236023 |
|Model 4	|XGBoost classifier	      |91.8245934	  |91.8255994	  |91.8245934	  |91.82452987 |


### Deployment using Streamlit:

To make the loan default prediction model accessible and user-friendly, we deployed it as an interactive web application using the Python library Streamlit. Streamlit allows us to create interactive data science and machine learning applications with minimal coding.

#### 1. Application Structure:

The Streamlit application was structured to provide a clear and intuitive user experience. It typically includes the following components:

- ***Data Exploration:*** An interactive section allowing users to explore the underlying loan application data through tables and visualizations. This can build trust and provide context.
- ***Model Performance:*** A section showcasing the model's performance metrics (accuracy, AUC-ROC, etc.) on the held-out test set to provide context on its reliability.
- ***Prediction Interface:*** The core of the application, where users can input the relevant features of a loan applicant. These input fields correspond to the features used by our trained model.
- ***Prediction Result:*** A section displaying the model's prediction (e.g., "Likely to Default" or "Likely Not to Default") based on the user's input.
- ***Probability Output:*** Displaying the probability score associated with the prediction can provide more nuanced information about the model's confidence.

#### 2. Feature Input:

For the prediction interface, we utilized various Streamlit input widgets to allow users to provide the necessary feature values. These widgets included:

- st.selectbox(): For categorical features with a limited number of options (e.g., Gender, Income Type, Education Type, Family Status, Housing Type, Occupation Type, Organization Type, Age Group, Children Category). The options presented to the user mirrored the categories the model was trained on.
- st.number_input(): For numerical features (e.g., Annual Income, Credit Amount). We might have included appropriate ranges and step values for these inputs based on the data distribution.

#### 3. Model Loading and Prediction:

The trained and saved Random Forest model (bankingRFmodel3.pkl) and the associated encoder objects were loaded into the Streamlit script using the pickle library. When the user provided the input features and triggered the prediction (e.g., by clicking a "Predict" button using st.button()), the preprocessed input DataFrame was passed to the model's predict() or predict_proba() method.


## Conclusion

Although XGBoost is a highly effective algorithm, the Random Forest Classifier provided a robust, well-performing, and relatively easier-to-deploy solution for this specific loan default prediction problem.
By deploying our loan default prediction model as a Streamlit web application, we provided an accessible and interactive tool for users to obtain predictions based on loan applicant information. The application incorporates the necessary preprocessing steps and utilizes the trained Random Forest model to deliver timely and informative results. This deployment strategy allows for easy sharing and utilization of the developed machine learning model.

## 🏆 Results
The final model chosen for deployment is the Random Forest Classifier, trained with the optimized hyperparameters identified during our model tuning phase.
Its performance on the held-out testing set met our requirements for accuracy, recall, and the ability to effectively distinguish between potential defaulters and non-defaulters.






