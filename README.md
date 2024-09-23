# Financial Inclusion in Africa: Bank Account Prediction

## Overview
This project focuses on **Financial Inclusion in Africa**, leveraging a dataset provided by the Zindi platform. The dataset contains demographic information and details about financial service usage by approximately 33,600 individuals across East Africa. The goal is to build a machine learning model that predicts whether an individual is likely to have or use a bank account. This is crucial in helping governments, organizations, and financial institutions better understand and improve financial inclusion.

## What is Financial Inclusion?
**Financial inclusion** refers to the process by which individuals and businesses gain access to affordable financial products and services—such as transactions, payments, savings, credit, and insurance—that meet their needs in a responsible and sustainable way.

---

## Dataset Information
The dataset contains various demographic features that can help in predicting whether an individual has a bank account. Each row represents one individual from East Africa, and the dataset provides the following columns (features):

| Feature Name       | Description                                                   |
|--------------------|---------------------------------------------------------------|
| **ID**             | Unique identifier for each individual                         |
| **Country**        | Country of the individual                                      |
| **Year**           | The year in which the data was collected                       |
| **Bank Account**   | Whether the individual has a bank account (target variable)    |
| **Location Type**  | Type of location (Urban/Rural)                                 |
| **Cellphone Access** | Whether the individual has access to a cellphone             |
| **Household Size** | Number of people in the individual’s household                 |
| **Age of Respondent** | Age of the individual                                       |
| **Gender**         | Gender of the individual                                       |
| **Marital Status** | Marital status of the individual                               |
| **Education Level** | Highest level of education achieved by the individual         |
| **Job Type**       | Type of employment                                             |

➡️ [Link to Dataset Image](https://i.imgur.com/UNUZ4zR.jpg)

➡️ [Link to Dataset ](https://drive.google.com/file/d/1FrFTfUln67599LTm2uMTSqM8DjqpAaKL/view)

---

## Project Workflow

### 1. **Install Required Packages**
To get started, you will need to install the necessary Python libraries. The key libraries used in this project include:
- `pandas` for data manipulation
- `numpy` for numerical operations
- `matplotlib` and `seaborn` for visualizations
- `scikit-learn` for machine learning
- `streamlit` for web application deployment

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit pandas-profiling
```

### 2. **Data Exploration and Preprocessing**
- **Import Data**: Load the dataset and inspect its structure.
- **Basic Exploration**: Perform basic exploration such as checking for missing values, duplicates, data types, and general statistics.
  
```python
import pandas as pd
df = pd.read_csv('financial_inclusion_africa.csv')
df.info()
df.describe()
```

- **Pandas Profiling**: Use the `pandas-profiling` package to generate an extensive report for a quick overview of the dataset.

```python
from pandas_profiling import ProfileReport
profile = ProfileReport(df, title="Financial Inclusion Report")
profile.to_widgets()
```

### 3. **Data Cleaning**
- **Handle Missing Values**: Handle missing and corrupted data using methods like imputation or removal.
- **Remove Duplicates**: Check and remove duplicate entries to avoid redundant information.
- **Handle Outliers**: Visualize and remove or treat outliers using statistical methods.

### 4. **Feature Engineering**
- **Encode Categorical Features**: Convert categorical features such as gender, marital status, etc., into numerical values using techniques like one-hot encoding or label encoding.
  
```python
df = pd.get_dummies(df, drop_first=True)
```

### 5. **Model Training**
- **Train-Test Split**: Split the dataset into training and testing sets.
  
```python
from sklearn.model_selection import train_test_split
X = df.drop('Bank Account', axis=1)
y = df['Bank Account']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- **Train Classifier**: Train a machine learning classifier (e.g., Random Forest, Logistic Regression) on the training set.

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
```

- **Evaluate Model**: Test the model’s performance on the test set using metrics like accuracy, precision, recall, and F1-score.

```python
from sklearn.metrics import classification_report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 6. **Streamlit Application**
- **Create Streamlit App**: Build an interactive web application using Streamlit. The app will allow users to input demographic data, and the model will predict whether the individual is likely to have a bank account.

```bash
streamlit run app.py
```

- **App Features**:
  - Input fields for features such as age, gender, household size, education level, etc.
  - A validation button to run predictions.

### 7. **Deploy the Application**
- **GitHub Repository**: 
  - Create a new GitHub repository.
  - Push the code to GitHub.

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

- **Deploy on Streamlit Share**: 
  - Log in to your Streamlit account.
  - Deploy the app using the Streamlit sharing platform, linking it to your GitHub repository.

---

## Conclusion
This project showcases a comprehensive machine learning pipeline, from data cleaning and feature engineering to model training and deployment. The final product is a **Streamlit web application** that predicts the likelihood of an individual having a bank account based on their demographic features.

**Demo Link**: [[Your Streamlit App URL](https://beige-experts-grab.loca.lt/)]

Feel free to explore the dataset and the model's predictions by providing your own inputs in the Streamlit app.

## Future Improvements
- Implement advanced models like Gradient Boosting or XGBoost for better predictions.
- Add more features or external datasets to improve model accuracy.
- Enhance the web app's UI for a better user experience.

## Author
[Fadhili jakes]  
For questions or feedback, please feel free to reach out!
