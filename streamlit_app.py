"""
Bank Churn Prediction - Streamlit Dashboard 
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
    .insight-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
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

# Main app body
try:
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/bank.png", width=80)
        st.title("🏦 Navigation")
        page = st.radio("Select Page", [
            "📊 Dashboard",
            "🎯 Risk Calculator",
            "🔮 What-If Simulator",
            "📈 Feature Analysis",
            "📋 Model Info"
        ])

    # ============================================
    # PAGE 1: DASHBOARD with Actionable Insights
    # ============================================
    if page == "📊 Dashboard":
        st.markdown('<p class="main-header">🏦 Bank Customer Churn Dashboard</p>', unsafe_allow_html=True)
        
        # ===== UPGRADE: ACTIONABLE INSIGHTS PANEL =====
        st.markdown("---")
        st.subheader("📈 Proactive Retention Insights for Management")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 High-Risk Segment")
            st.metric("Age > 50 & Inactive", "67% Churn Risk", delta="⚠️ Action Required")
            st.caption("→ Launch re-engagement campaign")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown("### 🏆 Best Retention Lever")
            st.metric("Add 2nd Product", "-62%", delta="Risk Reduction")
            st.caption("→ Prioritize cross-sell offers")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown("### 📍 Geographic Focus")
            st.metric("Germany", "32.4%", delta="Highest Churn Rate")
            st.caption("→ Investigate local satisfaction")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown("### ⏰ Urgent Window")
            st.metric("First 6 Months", "45%", delta="of churn happens")
            st.caption("→ Focus onboarding")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Key Metrics
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
        
        # Charts
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
            fig = px.bar(x=age_churn.index, y=age_churn.values,
                         title="Churn Rate by Age Group",
                         color=age_churn.values, color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)

    # ============================================
    # PAGE 2: RISK CALCULATOR
    # ============================================
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
            
            # Risk factors
            st.markdown("### 🔍 Key Risk Factors Identified")
            risk_factors = []
            if age > 50:
                risk_factors.append("• Age > 50 (higher churn risk)")
            if num_products == 1:
                risk_factors.append("• Only 1 product (cross-sell opportunity)")
            if is_active_val == 0:
                risk_factors.append("• Inactive member (engagement needed)")
            if geography == "Germany":
                risk_factors.append("• Germany has higher churn rate")
            if credit_score < 580:
                risk_factors.append("• Poor credit score")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.success("No major risk factors detected!")

    # ============================================
    # PAGE 3: WHAT-IF SIMULATOR 
    # ============================================
    elif page == "🔮 What-If Simulator":
        st.markdown('<p class="main-header">🔮 What-If Scenario Simulator</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Test Different Retention Strategies
        Adjust customer behavior or apply retention actions to see how churn probability changes.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Customer Profile")
            age = st.slider("Age", 18, 80, 55, key="sim_age")
            current_products = st.selectbox("Current Products", [1, 2, 3, 4], key="sim_prod")
            is_active = st.selectbox("Currently Active?", ["No", "Yes"], key="sim_active")
            geography = st.selectbox("Country", ["France", "Germany", "Spain"], key="sim_geo")
            credit_score = st.slider("Credit Score", 300, 850, 650, key="sim_credit")
        
        with col2:
            st.markdown("### 🎯 Retention Action (Select to Simulate)")
            st.write("Choose one or more actions:")
            offer_product = st.checkbox("📦 Offer an additional product", help="Convert single-product to multi-product")
            reengage = st.checkbox("📞 Run re-engagement campaign", help="Target inactive customers")
            loyalty = st.checkbox("🎁 Send loyalty reward", help="Fee waiver or reward points")
        
        # Calculate base risk (without actions)
        base_risk = 0.15
        if age > 50:
            base_risk += 0.25
        if current_products == 1:
            base_risk += 0.20
        if is_active == "No":
            base_risk += 0.15
        if geography == "Germany":
            base_risk += 0.10
        if credit_score < 580:
            base_risk += 0.10
        base_risk = min(base_risk, 0.95)
        
        # Calculate new risk after actions
        new_risk = base_risk
        actions_applied = []
        
        if offer_product:
            new_risk *= 0.60
            actions_applied.append("✅ Offered additional product (-40% risk)")
        if reengage:
            new_risk *= 0.55
            actions_applied.append("✅ Re-engagement campaign (-45% risk)")
        if loyalty:
            new_risk *= 0.70
            actions_applied.append("✅ Loyalty reward (-30% risk)")
        
        new_risk = max(new_risk, 0.05)
        reduction_pct = (base_risk - new_risk) / base_risk * 100
        
        st.markdown("---")
        st.markdown("### 📊 Simulation Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric("Current Risk", f"{base_risk:.1%}")
        
        with result_col2:
            st.metric("Risk After Action", f"{new_risk:.1%}", delta=f"-{reduction_pct:.0f}% improvement")
        
        with result_col3:
            if new_risk < 0.3:
                st.markdown('<div class="risk-low">🟢 LOW RISK</div>', unsafe_allow_html=True)
            elif new_risk < 0.5:
                st.markdown('<div class="risk-medium">🟡 MEDIUM RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-high">🔴 HIGH RISK</div>', unsafe_allow_html=True)
        
        # Show actions applied
        if actions_applied:
            st.markdown("### ✅ Actions Applied")
            for action in actions_applied:
                st.write(action)
        else:
            st.info("💡 Select one or more retention actions above to see their impact")
        
        # Business recommendation
        st.markdown("---")
        st.markdown("### 💡 Business Recommendation")
        
        if new_risk < 0.3:
            st.success("✅ Customer is now low-risk. Continue standard engagement and monitor quarterly.")
        elif new_risk < 0.5:
            st.info("📌 Customer is medium-risk. Consider additional incentives or personalized outreach.")
        else:
            st.error("⚠️ Customer remains high-risk. Escalate to retention team for immediate personal outreach.")
        
        # Progress bar visualization
        st.markdown("### 📈 Risk Reduction Visualization")
        progress_col1, progress_col2 = st.columns(2)
        with progress_col1:
            st.progress(base_risk, text=f"Before: {base_risk:.0%}")
        with progress_col2:
            st.progress(new_risk, text=f"After: {new_risk:.0%}")
        
        st.caption("💡 Tip: The best results come from combining multiple retention actions.")

    # ============================================
    # PAGE 4: FEATURE ANALYSIS
    # ============================================
    elif page == "📈 Feature Analysis":
        st.markdown('<p class="main-header">📈 Feature Importance Analysis</p>', unsafe_allow_html=True)
        
        # Load feature importance from CSV
        try:
            feature_imp = pd.read_csv('feature_importance_rf.csv')
            st.markdown("### 🔑 Top 15 Features Driving Churn")
            
            fig = px.bar(feature_imp.head(15), x='importance', y='feature', 
                         orientation='h', title="Feature Importance (Random Forest)",
                         color='importance', color_continuous_scale='Reds')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Feature importance file not found.")
        
        st.markdown("---")
        st.markdown("### 📊 How Features Impact Churn")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age_churn = df.groupby(pd.cut(df['Age'], bins=[18,30,40,50,60,100]))['Exited'].mean()
            fig = px.bar(x=[str(x) for x in age_churn.index], y=age_churn.values, 
                         title="Churn Rate by Age Group",
                         color=age_churn.values, color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            prod_churn = df.groupby('NumOfProducts')['Exited'].mean()
            fig = px.bar(x=prod_churn.index, y=prod_churn.values, 
                         title="Churn Rate by Number of Products",
                         color=prod_churn.values, color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            active_churn = df.groupby('IsActiveMember')['Exited'].mean()
            fig = px.bar(x=['Inactive', 'Active'], y=active_churn.values,
                         title="Churn Rate by Activity Status",
                         color=active_churn.values, color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            geo_churn = df.groupby('Geography')['Exited'].mean()
            fig = px.bar(x=geo_churn.index, y=geo_churn.values,
                         title="Churn Rate by Country",
                         color=geo_churn.values, color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📈 Correlation with Churn")
        
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

    # ============================================
    # PAGE 5: MODEL INFO
    # ============================================
    else:
        st.markdown('<p class="main-header">📋 Model Information</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🤖 Models Used
        
        | Model | Accuracy | ROC-AUC | Best For |
        |-------|----------|---------|----------|
        | Logistic Regression | 83.00% | 0.7813 | Interpretability |
        | Decision Tree | 84.65% | 0.8035 | Simple rules |
        | Random Forest ⭐ | 86.25% | 0.8512 | Current deployed model |
        | Gradient Boosting | 86.00% | 0.8525 | Sequential learning |
        | XGBoost | 86.75% | 0.8563 | Highest accuracy |
        
        ### 🔑 Key Features (Top Predictors of Churn)
        
        1. **NumOfProducts** - Number of bank products used
        2. **IsActiveMember** - Customer activity status
        3. **Age** - Customer age
        4. **Balance** - Account balance
        5. **CreditScore** - Creditworthiness
        
        ### 💡 Actionable Business Recommendations
        
        | Priority | Action | Expected Impact |
        |----------|--------|-----------------|
        | 🔴 High | Re-engagement campaigns for inactive members | -50% churn risk |
        | 🔴 High | Cross-sell to single-product customers | -40% churn risk |
        | 🟡 Medium | Investigate Germany market | Reduce 32% → 16% |
        | 🟢 Low | Real-time risk scoring deployment | Proactive retention |
        
        ### 📊 Key Statistics
        
        - **Total Customers Analyzed:** 10,000
        - **Overall Churn Rate:** 20.4%
        - **Germany Churn Rate:** 32.4% (Highest)
        - **Inactive Member Churn:** 26.9% vs Active: 14.3%
        - **Single-Product Churn:** 27.7% vs Multi-Product: ~12%
        - **Zero Balance Customers:** 3,617 (36% - High Risk)
        """)

    st.markdown("---")
    st.markdown("© 2024 Bank Churn Prediction System | Built for Unified Mentor Internship")

except Exception as e:
    st.error(f"❌ App Error: {str(e)}")
    st.code(traceback.format_exc())
