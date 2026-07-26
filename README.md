# Experiment No. 1 – Introduction to Kaggle and Data Preprocessing

| **Name** | **Register Number** | **Experiment No.** | **Date** |
|-----------|---------------------|--------------------|----------|
| Surya Prakash B | 212224230281 | 1 | 24 - 07 - 2026 |


## AIM

To perform data preprocessing on a dataset downloaded from Kaggle.


## EQUIPMENTS REQUIRED

### Hardware
- PC or Laptop

### Software
- Python 3.7 or later
- Anaconda Distribution / Google Colab / Jupyter Notebook


## DATASET

**Dataset Name:** Titanic Dataset

**Source:** Kaggle

**Description:**  
The Titanic dataset contains passenger information such as age, gender, ticket class, fare, and survival status. It is one of the most widely used datasets for learning data preprocessing and machine learning concepts.


## RELATED THEORETICAL CONCEPT

### Kaggle

Kaggle, a subsidiary of Google LLC, is an online platform for data scientists and machine learning practitioners. It provides datasets, coding notebooks, competitions, and collaborative environments for building and sharing machine learning models. Kaggle also allows users to explore datasets, participate in competitions, and learn from the global data science community.

### Data Preprocessing

Data preprocessing is the process of cleaning and transforming raw data into a format suitable for machine learning algorithms. Real-world datasets often contain missing values, duplicate records, inconsistent formats, categorical variables, and outliers. These issues can reduce the performance of machine learning models if left untreated.

Common preprocessing techniques include:
- Handling missing values
- Removing duplicate records
- Encoding categorical variables
- Feature scaling
- Detecting and handling outliers
- Splitting the dataset into training and testing sets

Proper preprocessing improves the quality of the data and helps produce more accurate and reliable machine learning models.

### Need for Data Preprocessing

Data preprocessing is necessary because machine learning algorithms require clean and properly formatted data. Missing values, inconsistent formats, and different feature scales can negatively affect model performance.

The major objectives of preprocessing are:
- Improve data quality
- Handle missing values
- Convert categorical data into numerical form
- Normalize or standardize numerical features
- Detect and remove outliers
- Improve model accuracy and training efficiency

## ALGORITHM

1. Import the required libraries and load the Titanic dataset.
2. Perform data preprocessing by handling missing values, removing unnecessary columns, and encoding categorical data.
3. Split the dataset into training and testing sets and apply feature scaling using StandardScaler.
4. Train a Logistic Regression model using the training data and predict the output for the test data.
5. Calculate the model accuracy and display the results.

---

# PROGRAM

```python
import numpy as np
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("/content/train.csv")

# Display first five rows
print(df.head())

# Display dataset information
print(df.info())

# Display missing values
print(df.isnull().sum())

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Remove Cabin column
df = df.drop("Cabin", axis=1)

# Encode categorical column
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# One-Hot Encoding
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Feature Selection
X = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked_Q",
        "Embarked_S",
    ]
]

y = df["Survived"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Outlier Detection using Z-Score
age_mean = df["Age"].mean()
age_std = df["Age"].std()

outliers_age = df[
    np.abs(df["Age"] - age_mean) / age_std > 3
]

print("Number of outliers in Age:", len(outliers_age))
display(outliers_age.head())

# Feature Scaling
scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

X_train_std_df = pd.DataFrame(
    X_train_std,
    columns=X.columns
)

print("Scaled Training Features")
display(X_train_std_df.head())

# Model Training
model = LogisticRegression()

model.fit(X_train_std, y_train)

# Prediction
pred = model.predict(X_test_std)

# Accuracy
accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)
```

---

# OUTPUT

### First Five Rows

<img width="1478" height="250" alt="{D9435B51-889D-4359-89AF-EBA8FE349B0F}" src="https://github.com/user-attachments/assets/db9a6eef-0b5c-4908-acce-1be61a94a101" />

---

### Dataset Information

<img width="1460" height="472" alt="{5DF93133-A167-441B-A27D-8B071E5BDF84}" src="https://github.com/user-attachments/assets/f522badb-afce-44d4-b063-e7f89bd8c78f" />

---

### Missing Values

<img width="668" height="335" alt="{6CE4617F-9ACE-46F7-9399-AA9C6444DE00}" src="https://github.com/user-attachments/assets/8f92135e-86ce-46c3-9752-fca26e009259" />

---

### Outlier Detection

<img width="1374" height="284" alt="{AE7FBA20-4544-4AC5-85CC-226E577C82CA}" src="https://github.com/user-attachments/assets/57d82818-2599-4599-9c3a-83ba241ba294" />

---

### Standardized Features

<img width="851" height="267" alt="{2C559200-8467-4C69-B2C1-0E9BA0D33079}" src="https://github.com/user-attachments/assets/698e6077-6f3f-401b-a3f7-7adb272b596e" />

---

### Model Accuracy

<img width="278" height="30" alt="{2B291006-662A-4133-B77E-8F89B7C3C22E}" src="https://github.com/user-attachments/assets/95c59a5c-09a2-4056-8613-1ba3b71a6d0f" />

---

# RESULT

Thus, data preprocessing was successfully performed on the Titanic dataset downloaded from Kaggle using Python. Missing values were handled, categorical variables were encoded, outliers were identified, feature scaling was applied using the StandardScaler, and the dataset was split into training and testing sets. Finally, a Logistic Regression model was trained and achieved an accuracy of approximately **81%**.
