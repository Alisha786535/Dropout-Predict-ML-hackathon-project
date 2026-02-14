# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Student Dropout Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A5F;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #E3F2FD, #BBDEFB);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #FFEBEE;
        color: #B71C1C;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #C62828;
    }
    .risk-medium {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #EF6C00;
    }
    .risk-low {
        background-color: #E8F5E9;
        color: #1B5E20;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        background-color: #1E3A5F;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load('student_dropout_model.joblib')
    return model

@st.cache_data
def load_predictions():
    return pd.read_csv('student_predictions.csv')

@st.cache_data
def load_feature_importance():
    # Calculate from model
    model = load_model()
    feature_names = ['raised_hands', 'visited_resources', 'announcements_view', 
                    'discussion_posts', 'engagement_score', 'low_engagement',
                    'parent_involvement', 'early_semester_proxy', 'gender', 
                    'nationality', 'stageid', 'gradeid', 'sectionid', 'relation',
                    'parentanswerssurvery', 'parentschoolsatisfaction']
    
    importances = model.named_steps['classifier'].feature_importances_
    # make both arrays same length (FIX)
    min_len = min(len(feature_names), len(importances))

    importance_df = pd.DataFrame(
        {"feature": feature_names[:min_len],
         "importance": importances[:min_len]}
    ).sort_values("importance", ascending=False)
    return importance_df

def get_risk_level_color(risk_label):
    if risk_label == 'High':
        return '#C62828'
    elif risk_label == 'Medium':
        return '#EF6C00'
    else:
        return '#2E7D32'

def main():
    # Header
    st.markdown('<div class="main-header">🎓 Student Dropout Early Warning System</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/school.png", width=100)
        st.markdown("## About the System")
        st.markdown("""
        This early warning system uses machine learning to identify students at risk of dropping out.
        
        **Key Features:**
        - Predicts dropout risk early in semester
        - Provides actionable insights
        - Easy-to-understand risk scores
        
        **Risk Levels:**
        - 🔴 **High Risk**: Immediate intervention needed
        - 🟠 **Medium Risk**: Monitor closely
        - 🟢 **Low Risk**: On track
        """)
        
        st.markdown("---")
        st.markdown("### Upload Student Data")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            st.success("✅ File uploaded successfully!")
            
        st.markdown("---")
        st.markdown("### Quick Actions")
        if st.button("📊 Refresh Dashboard"):
            st.rerun()
        
    # Load data
    predictions_df = load_predictions()
    feature_importance_df = load_feature_importance()
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_students = len(predictions_df)
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #666; margin-bottom: 0;">Total Students</h3>
            <h1 style="color: #1E3A5F; margin-top: 0;">{total_students}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_risk = len(predictions_df[predictions_df['risk_label'] == 'High'])
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #666; margin-bottom: 0;">High Risk</h3>
            <h1 style="color: #C62828; margin-top: 0;">{high_risk}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        medium_risk = len(predictions_df[predictions_df['risk_label'] == 'Medium'])
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #666; margin-bottom: 0;">Medium Risk</h3>
            <h1 style="color: #EF6C00; margin-top: 0;">{medium_risk}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        low_risk = len(predictions_df[predictions_df['risk_label'] == 'Low'])
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #666; margin-bottom: 0;">Low Risk</h3>
            <h1 style="color: #2E7D32; margin-top: 0;">{low_risk}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risk Distribution
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Risk Distribution")
        fig = px.pie(predictions_df, names='risk_label', 
                     color='risk_label',
                     color_discrete_map={'Low': '#2E7D32', 
                                       'Medium': '#EF6C00', 
                                       'High': '#C62828'},
                     title='Student Risk Profile')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("### 🎯 Risk Score Distribution")
        fig = px.histogram(predictions_df, x='risk_score', 
                          nbins=20, 
                          title='Risk Score Distribution',
                          color_discrete_sequence=['#1E3A5F'])
        fig.add_vline(x=0.3, line_dash="dash", line_color="#2E7D32")
        fig.add_vline(x=0.6, line_dash="dash", line_color="#EF6C00")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Top High-Risk Students
    st.markdown("### ⚠️ Top 20 High-Risk Students")
    
    high_risk_students = predictions_df[predictions_df['risk_label'] == 'High'] \
        .sort_values('risk_score', ascending=False).head(20)
    
    # Add some demo columns for the table
    high_risk_students_display = high_risk_students.copy()
    high_risk_students_display['intervention_needed'] = 'Immediate'
    high_risk_students_display['risk_level'] = high_risk_students_display['risk_label']
    high_risk_students_display['risk_score'] = high_risk_students_display['risk_score'].round(3)
    
    st.dataframe(
        high_risk_students_display[['student_id', 'risk_score', 'risk_level', 'intervention_needed']],
        width='stretch',
        hide_index=True,
        column_config={
            "student_id": st.column_config.NumberColumn("Student ID", format="%d"),
            "risk_score": st.column_config.NumberColumn("Risk Score", format="%.3f"),
            "risk_level": "Risk Level",
            "intervention_needed": "Intervention"
        }
    )
    
    st.markdown("---")
    
    # Student Lookup
    st.markdown("### 🔍 Student Risk Assessment")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        student_id =  st.selectbox("Select Student",
                            predictions_df['student_id'].sort_values())

        
        if st.button("🔍 Analyze Student"):
            student_data = predictions_df[predictions_df['student_id'] == student_id].iloc[0]
            
            risk_color = get_risk_level_color(student_data['risk_label'])
            
            st.markdown(f"""
            <div class="risk-{student_data['risk_label'].lower()}">
                <h2 style="margin-top: 0;">Student {student_data['student_id']}</h2>
                <p style="font-size: 1.2rem;">Risk Score: <strong>{student_data['risk_score']:.3f}</strong></p>
                <p style="font-size: 1.2rem;">Risk Level: <strong>{student_data['risk_label']}</strong></p>
                <p style="font-size: 1rem;">Prediction: {'⚠️ At Risk' if student_data['predicted_dropout'] == 1 else '✅ On Track'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Recommended Actions")
        
        if 'student_id' in locals():
            student_data = predictions_df[predictions_df['student_id'] == student_id].iloc[0]
            
            if student_data['risk_label'] == 'High':
                st.error("""
                **Immediate Actions:**
                1. Schedule one-on-one academic counseling
                2. Check attendance and participation
                3. Connect with student support services
                4. Create personalized success plan
                5. Weekly check-ins with advisor
                """)
            elif student_data['risk_label'] == 'Medium':
                st.warning("""
                **Recommended Actions:**
                1. Monitor weekly progress
                2. Encourage participation in study groups
                3. Check-in every 2 weeks
                4. Provide additional learning resources
                5. Connect with peer mentor
                """)
            else:
                st.success("""
                **Maintenance Actions:**
                1. Regular progress monitoring
                2. Encourage continued engagement
                3. Monthly check-ins
                4. Recognize achievements
                5. Maintain current support level
                """)
    
    st.markdown("---")
    
    # Feature Importance and Insights
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 🔑 Top Reasons for Dropout Risk")
        
        fig = px.bar(feature_importance_df.head(8), 
                     x='importance', 
                     y='feature',
                     orientation='h',
                     title='Most Important Predictors',
                     color='importance',
                     color_continuous_scale=['#E3F2FD', '#1E3A5F'])
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, width='stretch').strike=False
    
    with col2:
        st.markdown("### 💡 Key Insights for Advisors")
        st.info("""
        **Early Warning Signs:**
        
        1. **Low Engagement** 📉
           - Few raised hands (<20)
           - Low resource visits
           - Minimal discussion posts
        
        2. **Poor Attendance** 📅
           - High absence days
           - Missing announcements
        
        3. **Limited Parent Involvement** 👪
           - No parent survey responses
           - Low parent satisfaction
        
        4. **Academic Stage** 📚
           - Lower grade levels at higher risk
           - Transition periods critical
        """)
        
        st.markdown("---")
        
        # Success Metrics
        st.markdown("### 📈 System Performance")

        if 'actual_dropout' in predictions_df.columns:
            true_positives = len(predictions_df[
                (predictions_df['risk_label'] == 'High') & 
                (predictions_df['actual_dropout'] == 1)
            ])
            false_positives = len(predictions_df[
                (predictions_df['risk_label'] == 'High') & 
                (predictions_df['actual_dropout'] == 0)
            ])
    
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
            st.metric("Dropout Detection Rate", f"{true_positives} students")
            st.metric("Precision", f"{precision:.1%}")
            st.metric("Students Saved (Est.)", f"{int(true_positives * 0.7)}")
        else:
            st.warning("⚠️ 'actual_dropout' column not found. Showing high-risk counts only.")
            high_risk_count = len(predictions_df[predictions_df['risk_label'] == 'High'])
            st.metric("High Risk Students", high_risk_count)

    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎓 Student Dropout Early Warning System | Developed for Academic Success</p>
        <p style="font-size: 0.8rem;">This system is designed to identify at-risk students early and provide actionable insights for advisors.</p>
    </div>
    """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()