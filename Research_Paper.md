# Bank Customer Churn Prediction - Research Paper

## Abstract
Customer churn is a critical challenge for retail banks, directly impacting revenue and customer lifetime value. This research presents a machine learning approach to predict customer churn using a dataset of 10,000 European bank customers. Four classification models were developed and compared: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. The Random Forest model achieved the best performance with 86.75% accuracy and an ROC-AUC of 0.91. Key churn drivers identified include age, number of products, and account activity status.

## 1. Introduction
Traditional churn analysis explains why churn happened after the fact. However, modern banking requires early identification of customers likely to churn, enabling proactive retention campaigns and personalized offers.

## 2. Problem Statement
Despite having rich customer-level data, banks often lack:
- Accurate churn prediction models
- Quantitative churn risk scores
- Explainable insights into churn drivers

## 3. Dataset Description
The dataset contains 10,000 customer records with 14 features including:
- Credit Score, Geography, Gender, Age
- Tenure, Balance, Number of Products
- Credit Card ownership, Active Member status
- Estimated Salary, Exited (target variable)

**Class Distribution:**
- Retained customers: 7,963 (79.6%)
- Churned customers: 2,037 (20.4%)

## 4. Methodology

### 4.1 Data Preprocessing
- Removed non-informative columns (CustomerId, Surname)
- Encoded categorical variables using one-hot encoding
- Scaled numerical features using StandardScaler

### 4.2 Feature Engineering
Created 8 derived features:
1. Balance_to_Salary - wealth distribution indicator
2. Age_Tenure - customer loyalty metric
3. Product_Density - products per tenure
4. Product_Engagement - product usage intensity
5. Wealth_Index - combined wealth measure
6. Account_Utilization - balance per product
7. High_Value - premium customer flag
8. Credit_Category - credit score tier

### 4.3 Models Developed
| Model | Type | Description |
|-------|------|-------------|
| Logistic Regression | Baseline | Linear classification |
| Decision Tree | Tree-based | Rule-based learning |
| Random Forest | Ensemble | Multiple decision trees |
| Gradient Boosting | Ensemble | Sequential learning |

## 5. Results

### 5.1 Model Performance Comparison
| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| Random Forest | 86.75% | 0.91 | 0.60 |
| Gradient Boosting | 86.00% | 0.89 | 0.58 |
| Decision Tree | 84.65% | 0.80 | 0.54 |
| Logistic Regression | 83.00% | 0.78 | 0.43 |

### 5.2 Top Churn Drivers (Feature Importance)
1. Number of Products
2. Is Active Member  
3. Age
4. Balance
5. Credit Score
6. Tenure
7. Geography_Germany
8. Estimated Salary

### 5.3 Key Insights from EDA
- **Churn Rate**: 20.4% overall
- **Geography**: Germany has highest churn (32.4%)
- **Gender**: Women churn more (25.1% vs 16.5%)
- **Activity**: Inactive members churn 2x more (26.9% vs 14.3%)
- **Products**: Single-product customers churn more (27.7%)
- **Age**: Strongest correlation with churn (0.285)

## 6. Discussion

### 6.1 Why Random Forest Performed Best
Random Forest handles non-linear relationships well and is robust to outliers. Its ensemble nature captures complex patterns in customer behavior.

### 6.2 Business Implications
- **Product Strategy**: Focus retention on single-product customers
- **Engagement Campaigns**: Target inactive members with personalized offers
- **Age-Based Retention**: Monitor customers over 50 years old
- **Geographic Focus**: Germany market needs special attention

## 7. Recommendations

1. **Immediate Actions**:
   - Launch retention campaigns for single-product customers
   - Engage inactive members with personalized offers
   - Implement monitoring system for customers over 50

2. **Medium-term Strategy**:
   - Develop Germany-specific retention programs
   - Create product bundling strategies
   - Implement customer health scores

3. **Long-term Initiatives**:
   - Build real-time churn prediction API
   - Integrate predictions into CRM system
   - Develop automated retention workflows

## 8. Conclusion
Machine learning effectively predicts customer churn with high accuracy. The Random Forest model provides reliable churn probability scores, enabling proactive retention strategies. Key churn drivers identified (age, product count, activity status) provide actionable insights for business decisions.

## 9. Future Work
- Incorporate more features (transaction history, customer service calls)
- Implement deep learning models for comparison
- Develop real-time prediction system
- A/B test retention strategies based on model predictions

## References
1. sklearn documentation - scikit-learn.org
2. Streamlit documentation - streamlit.io
3. European Bank dataset (provided)

---

**Author**: [Your Name]
**Date**: May 2026
**Project**: Bank Customer Churn Prediction