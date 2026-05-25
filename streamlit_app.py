"""
Bank Churn Prediction - Streamlit Dashboard
Interactive web application for churn prediction and analysis
"""

import sys
import traceback

# Error handling for imports
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import warnings
    warnings.filterwarnings('ignore')
except Exception as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.code(traceback.format_exc())
    st.stop()

# Page config
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #DC2626;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #F59E0B;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background-color: #10B981;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main app with error handling
try:
    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv('data/European_Bank.csv')
        return df

    # Train model
    @st.cache_resource
    def train_model():
        df = load_data()
        
        # Preprocessing
        df = df.drop(['CustomerId', 'Surname'], axis=1)
        
        # Feature engineering
        df['Balance_to_Salary'] = df['Balance'] / (df['EstimatedSalary'] + 1)
        df['Balance_to_Salary'] = df['Balance_to_Salary'].clip(0, 5)
        df['Age_Tenure'] = df['Age'] * df['Tenure']
        df['Product_Density'] = df['NumOfProducts'] / (df['Tenure'] + 1)
        df['Product_Engagement'] = df['NumOfProducts'] * df['IsActiveMember']
        df['Wealth_Index'] = df['Balance'] + (df['EstimatedSalary'] / 12)
        df['Account_Utilization'] = df['Balance'] / (df['NumOfProducts'] + 1)
        df['High_Value'] = ((df['Balance'] > 50000) | (df['EstimatedSalary'] > 100000)).astype(int)
        df['Credit_Category'] = pd.cut(df['CreditScore'], bins=[0, 580, 670, 740, 850], 
                                        labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        # Encode
        df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
        le = LabelEncoder()
        df['Credit_Category'] = le.fit_transform(df['Credit_Category'])
        
        # Features and target
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        
        # Scale
        numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                          'EstimatedSalary', 'Balance_to_Salary', 'Age_Tenure',
                          'Product_Density', 'Product_Engagement', 'Wealth_Index',
                          'Account_Utilization']
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        return model, scaler, le, X.columns.tolist()

    # Load data and model
    df = load_data()
    model, scaler, le, feature_cols = train_model()

except FileNotFoundError as e:
    st.error(f"❌ File Not Found Error: {str(e)}")
    st.info("Make sure the file 'data/European_Bank.csv' exists in your repository")
    st.code(traceback.format_exc())
    st.stop()
    
except Exception as e:
    st.error(f"❌ Data/Model Loading Error: {str(e)}")
    st.code(traceback.format_exc())
    st.stop()

# Main app body with error handling
try:
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/bank.png", width=80)
        st.title("🏦 Navigation")
        page = st.radio("Select Page", [
            "📊 Dashboard",
            "🎯 Risk Calculator",
            "📈 Feature Analysis",
            "📋 Model Info"
        ])

    # Dashboard Page
    if page == "📊 Dashboard":
        st.markdown('<p class="main-header">🏦 Bank Customer Churn Dashboard</p>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        churn_rate = df['Exited'].mean() * 100
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        with col3:
            st.metric("Avg Age", f"{df['Age'].mean():.0f}")
        with col4:
            st.metric("Avg Balance", f"€{df['Balance'].mean():,.0f}")
        
        st.markdown("---")
        
        # Geography chart
        col1, col2 = st.columns(2)
        with col1:
            geo_churn = df.groupby('Geography')['Exited'].mean()
            fig = px.bar(x=geo_churn.index, y=geo_churn.values, 
                         title="Churn Rate by Country",
                         color=geo_churn.values, color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            df['Age_Group'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 100], 
                                      labels=['18-30', '31-40', '41-50', '51-60', '60+'])
            age_churn = df.groupby('Age_Group')['Exited'].mean()
            fig = px.line(x=age_churn.index, y=age_churn.values, markers=True,
                          title="Churn Rate by Age")
            st.plotly_chart(fig, use_container_width=True)

    # Risk Calculator Page
    elif page == "🎯 Risk Calculator":
        st.markdown('<p class="main-header">🎯 Customer Risk Calculator</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            age = st.slider("Age", 18, 100, 35)
            tenure = st.slider("Tenure (years)", 0, 10, 5)
            balance = st.number_input("Balance (€)", 0, 500000, 50000)
        
        with col2:
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
            is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
            salary = st.number_input("Salary (€)", 0, 500000, 80000)
        
        col3, col4 = st.columns(2)
        with col3:
            geography = st.selectbox("Country", ["France", "Spain", "Germany"])
        with col4:
            gender = st.selectbox("Gender", ["Female", "Male"])
        
        # Convert inputs
        has_cr_card_val = 1 if has_cr_card == "Yes" else 0
        is_active_val = 1 if is_active == "Yes" else 0
        
        if st.button("Calculate Risk", type="primary", use_container_width=True):
            # Create customer data
            customer = {
                'CreditScore': credit_score, 'Age': age, 'Tenure': tenure,
                'Balance': balance, 'NumOfProducts': num_products,
                'HasCrCard': has_cr_card_val, 'IsActiveMember': is_active_val,
                'EstimatedSalary': salary
            }
            
            # Create DataFrame
            df_test = pd.DataFrame([customer])
            
            # Feature engineering
            df_test['Balance_to_Salary'] = df_test['Balance'] / (df_test['EstimatedSalary'] + 1)
            df_test['Balance_to_Salary'] = df_test['Balance_to_Salary'].clip(0, 5)
            df_test['Age_Tenure'] = df_test['Age'] * df_test['Tenure']
            df_test['Product_Density'] = df_test['NumOfProducts'] / (df_test['Tenure'] + 1)
            df_test['Product_Engagement'] = df_test['NumOfProducts'] * df_test['IsActiveMember']
            df_test['Wealth_Index'] = df_test['Balance'] + (df_test['EstimatedSalary'] / 12)
            df_test['Account_Utilization'] = df_test['Balance'] / (df_test['NumOfProducts'] + 1)
            df_test['High_Value'] = ((df_test['Balance'] > 50000) | (df_test['EstimatedSalary'] > 100000)).astype(int)
            
            # Credit category
            if credit_score < 580:
                cat = 0
            elif credit_score < 670:
                cat = 1
            elif credit_score < 740:
                cat = 2
            else:
                cat = 3
            df_test['Credit_Category'] = cat
            
            # One-hot encode
            df_test['Geography_Germany'] = 1 if geography == 'Germany' else 0
            df_test['Geography_Spain'] = 1 if geography == 'Spain' else 0
            df_test['Gender_Male'] = 1 if gender == 'Male' else 0
            
            # Ensure all features present
            for col in feature_cols:
                if col not in df_test.columns:
                    df_test[col] = 0
            
            df_test = df_test[feature_cols]
            
            # Predict
            prob = model.predict_proba(df_test)[0, 1]
            
            # Display result
            st.markdown("---")
            st.markdown("### 📊 Risk Assessment")
            
            col1, col2, col3 = st.columns(3)
            with col2:
                if prob >= 0.5:
                    st.markdown(f'<div class="risk-high">⚠️ HIGH RISK<br>Probability: {prob:.1%}</div>', unsafe_allow_html=True)
                elif prob >= 0.3:
                    st.markdown(f'<div class="risk-medium">⚠️ MEDIUM RISK<br>Probability: {prob:.1%}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low">✅ LOW RISK<br>Probability: {prob:.1%}</div>', unsafe_allow_html=True)

    # Feature Analysis Page
    elif page == "📈 Feature Analysis":
        st.markdown('<p class="main-header">📈 Feature Importance Analysis</p>', unsafe_allow_html=True)
        
        # Load feature importance from CSV (saved during model training)
        try:
            feature_imp = pd.read_csv('feature_importance_rf.csv')
            st.markdown("### 🔑 Top 15 Features Driving Churn")
            
            fig = px.bar(feature_imp.head(15), x='importance', y='feature', 
                         orientation='h', title="Feature Importance (Random Forest)",
                         color='importance', color_continuous_scale='Reds')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Feature importance file not found. Run model training first.")
        
        st.markdown("---")
        st.markdown("### 📊 How Features Impact Churn")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age impact
            age_churn = df.groupby(pd.cut(df['Age'], bins=[18,30,40,50,60,100]))['Exited'].mean()
            fig = px.line(x=[str(x) for x in age_churn.index], y=age_churn.values, 
                          markers=True, title="Churn Rate by Age Group")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Products impact
            prod_churn = df.groupby('NumOfProducts')['Exited'].mean()
            fig = px.bar(x=prod_churn.index, y=prod_churn.values, 
                         title="Churn Rate by Number of Products",
                         color=prod_churn.values, color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Activity impact
            active_churn = df.groupby('IsActiveMember')['Exited'].mean()
            fig = px.bar(x=['Inactive', 'Active'], y=active_churn.values,
                         title="Churn Rate by Activity Status",
                         color=active_churn.values, color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Geography impact
            geo_churn = df.groupby('Geography')['Exited'].mean()
            fig = px.bar(x=geo_churn.index, y=geo_churn.values,
                         title="Churn Rate by Country",
                         color=geo_churn.values, color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📈 Correlation Analysis")
        
        # Correlation heatmap
        corr_features = ['Age', 'NumOfProducts', 'IsActiveMember', 'Balance', 'CreditScore', 'Tenure']
        corr_data = df[corr_features + ['Exited']].corr()
        corr_with_target = corr_data['Exited'].drop('Exited').sort_values(ascending=False)
        
        fig = px.bar(x=corr_with_target.values, y=corr_with_target.index,
                     orientation='h', title="Correlation with Churn",
                     color=corr_with_target.values, color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - **Positive correlation** → Higher values increase churn risk
        - **Negative correlation** → Higher values decrease churn risk
        - **Age** has the strongest positive correlation with churn
        - **IsActiveMember** has strong negative correlation (active members churn less)
        """)

    # Model Info Page
    else:
        st.markdown('<p class="main-header">📋 Model Information</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ###  Models Used
        
        1. **Logistic Regression** - Baseline model
        2. **Decision Tree** - Simple tree-based model
        3. **Random Forest** - Ensemble of decision trees (Best)
        4. **Gradient Boosting** - Sequential learning
        
        ###  Key Features
        
        **Top Predictors of Churn:**
        - Number of Products
        - Is Active Member
        - Age
        - Balance
        - Credit Score
        
        ### Recommendations
        
        1. Focus on customers with single product
        2. Engage inactive members
        3. Monitor older customers (>50)
        4. Special attention to Germany market
        """)

    st.markdown("---")
    st.markdown("© 2024 Bank Churn Prediction System")

except Exception as e:
    st.error(f"❌ App Error: {str(e)}")
    st.code(traceback.format_exc())
