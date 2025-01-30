import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
from datetime import datetime, timedelta

from aws_cost_analyzer import (
    AnalysisConfig,
    AWSCostAnalyzer,
    SnowflakeConnection,
    CostDataLoader
)

def load_config():
    """Load configuration from YAML file"""
    with open("config/config.yaml") as f:
        return AnalysisConfig(**yaml.safe_load(f))

def create_cost_forecast_chart(forecast_results):
    """Create interactive cost forecast chart"""
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=forecast_results['forecast']['ds'],
            y=forecast_results['forecast']['y'],
            name='Actual Cost',
            line=dict(color='#2563eb')
        )
    )
    
    # Add forecast values
    fig.add_trace(
        go.Scatter(
            x=forecast_results['forecast']['ds'],
            y=forecast_results['forecast']['yhat'],
            name='Forecast',
            line=dict(color='#9333ea', dash='dash')
        )
    )
    
    fig.update_layout(
        title='Cost Forecast Analysis',
        xaxis_title='Date',
        yaxis_title='Cost ($)',
        height=500
    )
    
    return fig

def create_anomaly_chart(anomaly_results):
    """Create interactive anomaly detection chart"""
    fig = go.Figure()
    
    # Add traces for each severity level
    colors = {'normal': '#22c55e', 'medium': '#eab308', 'high': '#dc2626'}
    
    for severity in ['normal', 'medium', 'high']:
        mask = anomaly_results['anomalies']['severity'] == severity
        data = anomaly_results['anomalies'][mask]
        
        fig.add_trace(
            go.Scatter(
                x=data['UsageStartDate'],
                y=data['UnblendedCost'],
                name=f'{severity.title()} Cost',
                mode='markers',
                marker=dict(color=colors[severity], size=8)
            )
        )
    
    fig.update_layout(
        title='Cost Anomalies',
        xaxis_title='Date',
        yaxis_title='Cost ($)',
        height=500
    )
    
    return fig

def create_optimization_chart(optimization_results):
    """Create interactive optimization opportunities chart"""
    recommendations = optimization_results['recommendations']
    
    fig = go.Figure(
        go.Bar(
            x=[rec['category'] for rec in recommendations],
            y=[rec['annual_savings'] for rec in recommendations],
            text=[f"${savings:,.0f}" for savings in [rec['annual_savings'] for rec in recommendations]],
            textposition='auto',
            marker_color='#2563eb'
        )
    )
    
    fig.update_layout(
        title='Cost Optimization Opportunities',
        xaxis_title='Category',
        yaxis_title='Potential Annual Savings ($)',
        height=500
    )
    
    return fig

def main():
    st.set_page_config(page_title="AWS Cost Analyzer", layout="wide")
    
    # Header
    st.title("AWS Cost Analysis Dashboard")
    st.markdown("Enterprise-Grade AWS Cost Management and Optimization Platform")
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Sidebar configuration
    st.sidebar.title("Analysis Configuration")
    
    forecast_periods = st.sidebar.slider(
        "Forecast Periods (Days)",
        min_value=7,
        max_value=90,
        value=30
    )
    
    anomaly_sensitivity = st.sidebar.slider(
        "Anomaly Detection Sensitivity",
        min_value=0.01,
        max_value=0.2,
        value=0.1,
        step=0.01
    )
    
    min_saving_threshold = st.sidebar.number_input(
        "Minimum Savings Threshold ($)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )
    
    # Run Analysis Button
    if st.sidebar.button("Run Analysis"):
        with st.spinner("Running cost analysis..."):
            try:
                # Create configuration
                config = AnalysisConfig(
                    forecast_periods=forecast_periods,
                    anomaly_sensitivity=anomaly_sensitivity,
                    min_saving_threshold=min_saving_threshold
                )
                
                # Initialize and run analyzer
                analyzer = AWSCostAnalyzer(config)
                results = analyzer.run_analysis()
                st.session_state.analysis_results = results
                st.success("Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                return
    
    # Display Results
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Key Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Forecast Accuracy (MAPE)",
                f"{results['forecast_results']['metrics']['mape']:.1%}"
            )
            
        with col2:
            st.metric(
                "Total Anomalies",
                results['anomaly_results']['metrics']['total_anomalies']
            )
            
        with col3:
            st.metric(
                "Potential Annual Savings",
                f"${results['optimization_results']['total_savings']:,.2f}"
            )
        
        # Tabs for detailed analysis
        tab1, tab2, tab3 = st.tabs(["Cost Forecast", "Anomalies", "Optimization"])
        
        with tab1:
            st.plotly_chart(
                create_cost_forecast_chart(results['forecast_results']),
                use_container_width=True
            )
            
        with tab2:
            st.plotly_chart(
                create_anomaly_chart(results['anomaly_results']),
                use_container_width=True
            )
            
        with tab3:
            st.plotly_chart(
                create_optimization_chart(results['optimization_results']),
                use_container_width=True
            )
            
            # Optimization Recommendations
            st.subheader("Optimization Recommendations")
            for rec in results['optimization_results']['recommendations']:
                with st.expander(f"{rec['category']} - ${rec['annual_savings']:,.2f} potential savings"):
                    st.write(f"**Description:** {rec['description']}")
                    st.write(f"**Priority:** {rec['priority']}")
                    st.write(f"**Implementation Complexity:** {rec['implementation_complexity']}")
                    st.write("**Action Items:**")
                    for item in rec['action_items']:
                        st.write(f"- {item}")
        
        # Download Report
        if st.download_button(
            "Download PDF Report",
            data=open(results['report_path'], 'rb').read(),
            file_name=f"aws_cost_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        ):
            st.success("Report downloaded successfully!")

if __name__ == "__main__":
    main()
