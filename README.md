# Student Mental Health Classification Using Machine Learning

---

## a. Problem Statement

Mental health issues among students have become a significant concern due to academic pressure, lifestyle habits, and stress. Early identification of students who are at risk of depression can help institutions take preventive and supportive measures.

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether a student is likely to experience depression based on academic, lifestyle, and behavioral attributes.

This is formulated as a binary classification problem, where the target variable is **Depression**.

---

## b. Dataset Description  [1 Mark]

The dataset consists of student demographic, academic, lifestyle, and mental-health-related features. Each record represents a single student.

### Dataset Attributes

| Column Name | Description | Data Type |
|------------|------------|----------|
| Student_ID | Unique identifier for each student | Integer |
| Age | Age of the student (18–24) | Integer |
| Gender | Gender of the student | Categorical |
| Department | Field of study | Categorical |
| CGPA | Cumulative Grade Point Average (0–4) | Float |
| Sleep_Duration | Average sleep hours per night | Float |
| Study_Hours | Average study hours per day | Float |
| Social_Media_Hours | Average social media usage per day | Float |
| Physical_Activity | Physical activity per week (minutes) | Integer |
| Stress_Level | Self-reported stress level (0–10) | Integer |
| Depression | Mental health status (1 = Probable Depression, 0 = Healthy) | Boolean |

- Total instances: ≥ 500  
- Target variable: **Depression**  
- Problem type: **Binary Classification**

---

## c. Models Used and Evaluation Metrics  [6 Marks]

The following machine learning models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

### Evaluation Metrics Used
- Accuracy  
- ROC-AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## Comparison Table of Model Performance

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|---------|-----|----------|--------|---------|-----|
| Logistic Regression | 0.61825 | 0.67931 | 0.16171 | 0.66799 | 0.26039 | 0.17132 |
| Decision Tree | 0.83385 | 0.53772 | 0.16944 | 0.16099 | 0.16821 | 0.07593 |
| kNN | 0.88995 | 0.59301 | 0.26076 | 0.05119 | 0.08558 | 0.07558 |
| Naive Bayes | 0.89700 | 0.68394 | 0.40977 | 0.05418 | 0.09570 | 0.11933 |
| Random Forest (Ensemble) | 0.89955 | 0.69694 | 0.52307 | 0.01690 | 0.03274 | 0.08020 |
| XGBoost (Ensemble) | 0.74105 | 0.68255 | 0.20822 | 0.56163 | 0.30380 | 0.21833 |

---

## Model Performance Observations  [3 Marks]

| ML Model Name | Observation about model performance |
|---------------|-----------------------------------|
| Logistic Regression | Achieved high recall after applying class balancing, making it effective at identifying depressed students, though precision remained low due to false positives. |
| Decision Tree | Produced moderate accuracy but showed weak generalization, resulting in lower AUC and MCC values. |
| kNN | Achieved high accuracy but very low recall, indicating poor detection of minority (depressed) class instances. |
| Naive Bayes | Provided better precision compared to kNN and Decision Tree but suffered from low recall due to independence assumptions. |
| Random Forest (Ensemble) | Obtained the highest accuracy but extremely low recall, indicating bias toward the majority class despite ensemble learning. |
| XGBoost (Ensemble) | Demonstrated the best balance among recall, F1 score, and MCC, making it the most robust model for this imbalanced mental-health dataset. |

---

## Conclusion

Accuracy alone was not sufficient due to class imbalance in the dataset. Metrics such as Recall, F1 Score, and Matthews Correlation Coefficient provided a more reliable evaluation. Among all models, **XGBoost showed the best overall balance**, making it the most suitable model for identifying students at risk of depression.
