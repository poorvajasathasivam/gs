"""
AWS Cost Analysis and Optimization Platform
-----------------------------------------
Enterprise-Grade Solution for AWS Cost Management
Version: 1.0.0
Last Updated: 2025-01-30

This platform provides comprehensive AWS cost analysis, forecasting, and optimization 
using advanced machine learning techniques and FinOps best practices.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import snowflake.connector
from fpdf import FPDF
import yaml
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configuration and Constants
CONFIG_PATH = Path("config/config.yaml")
REPORT_TEMPLATE_PATH = Path("templates/report_template.html")
LOG_PATH = Path("logs/aws_cost_analysis.log")

@dataclass
class AnalysisConfig:
    """Configuration parameters for the analysis"""
    forecast_periods: int = 30
    anomaly_sensitivity: float = 0.1
    min_saving_threshold: float = 1000.0
    report_format: str = "pdf"

class DataValidator:
    """Validates input data quality and structure"""
    
    @staticmethod
    def validate_cost_data(df: pd.DataFrame) -> bool:
        """
        Validates AWS cost data format and quality
        
        Args:
            df: Input DataFrame containing AWS cost data
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        required_columns = [
            'UsageStartDate', 'UsageAccountId', 'ProductCode',
            'UnblendedCost', 'UsageAmount'
        ]
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Validate data types
        if not pd.api.types.is_datetime64_any_dtype(df['UsageStartDate']):
            raise ValueError("UsageStartDate must be datetime type")
            
        # Check for negative costs
        if (df['UnblendedCost'] < 0).any():
            raise ValueError("Found negative cost values")
            
        # Validate date range
        date_range = df['UsageStartDate'].max() - df['UsageStartDate'].min()
        if date_range.days < 30:
            raise ValueError("Insufficient historical data for analysis")
            
        return True

class SnowflakeConnection:
    """Manages Snowflake database connections with enterprise security"""
    
    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)['snowflake']
        
        self.connection_params = {
            'user': 'GUEST_USER',
            'role': 'GUEST_USER_ROLE',
            'account': self.config['account'],
            'warehouse': 'XSMALL_WH',
            'database': self.config['database']
        }
        
    def __enter__(self):
        try:
            self.conn = snowflake.connector.connect(**self.connection_params)
            return self.conn
        except Exception as e:
            logging.error(f"Snowflake connection failed: {str(e)}")
            raise
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'conn'):
            self.conn.close()

class CostDataLoader:
    """Handles data loading and preprocessing"""
    
    def __init__(self, connection: SnowflakeConnection):
        self.connection = connection
        self.validator = DataValidator()
        
    def load_cost_data(self) -> pd.DataFrame:
        """
        Loads AWS cost data from Snowflake
        
        Returns:
            pd.DataFrame: Processed cost data ready for analysis
        """
        query = """
        SELECT 
            identity.LineItemId,
            identity.TimeInterval,
            lineItem.UsageStartDate,
            lineItem.UsageEndDate,
            lineItem.UsageAccountId,
            lineItem.ProductCode,
            lineItem.UsageType,
            lineItem.Operation,
            lineItem.AvailabilityZone,
            lineItem.ResourceId,
            lineItem.UsageAmount,
            lineItem.UnblendedRate,
            lineItem.UnblendedCost,
            product.ProductName,
            product.PurchaseOption,
            pricing.unit,
            reservation.EffectiveCost,
            savings.SavingsPlanARN,
            discount.TotalDiscount
        FROM aws_cur_table
        WHERE lineItem.UsageStartDate >= DATEADD(month, -12, CURRENT_DATE())
        """
        
        with self.connection as conn:
            df = pd.read_sql(query, conn)
            
        df['UsageStartDate'] = pd.to_datetime(df['UsageStartDate'])
        df['UsageEndDate'] = pd.to_datetime(df['UsageEndDate'])
        
        self.validator.validate_cost_data(df)
        return df

class CostAnalyzer(ABC):
    """Abstract base class for cost analysis components"""
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Performs cost analysis"""
        pass

class CostForecaster(CostAnalyzer):
    """Handles cost forecasting using Prophet"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Generates cost forecasts using Prophet
        
        Args:
            df: Input DataFrame with cost data
            
        Returns:
            Dict containing forecast results and metrics
        """
        # Prepare data for Prophet
        forecast_df = df.groupby('UsageStartDate')['UnblendedCost'].sum().reset_index()
        forecast_df.columns = ['ds', 'y']
        
        # Train test split
        train_size = int(len(forecast_df) * 0.8)
        train = forecast_df[:train_size]
        test = forecast_df[train_size:]
        
        # Train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(train)
        
        # Generate predictions
        future_dates = model.make_future_dataframe(
            periods=self.config.forecast_periods
        )
        forecast = model.predict(future_dates)
        
        # Calculate metrics
        test_predictions = forecast.tail(len(test))
        mape = mean_absolute_percentage_error(test['y'], test_predictions['yhat'])
        mae = mean_absolute_error(test['y'], test_predictions['yhat'])
        
        return {
            'forecast': forecast,
            'model': model,
            'metrics': {
                'mape': mape,
                'mae': mae
            }
        }

# This is part 1 of the solution. Would you like me to continue with the remaining components including:
1. Anomaly Detection
2. Cost Optimization Recommendations
3. Visualization Engine
4. Report Generation
5. Main Application Class
6. CLI Interface
7. Configuration Management
8. Testing Framework


# Anomaly Detection Component
class AnomalyDetector(CostAnalyzer):
    """Enterprise-grade anomaly detection for AWS costs"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.isolation_forest = IsolationForest(
            contamination=self.config.anomaly_sensitivity,
            random_state=42
        )
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Detects cost anomalies using multiple detection methods
        
        Args:
            df: Input DataFrame with cost data
            
        Returns:
            Dict containing detected anomalies and their severity
        """
        # Prepare daily cost data
        daily_costs = df.groupby('UsageStartDate')['UnblendedCost'].sum().reset_index()
        
        # Statistical detection (3-sigma rule)
        mean_cost = daily_costs['UnblendedCost'].mean()
        std_cost = daily_costs['UnblendedCost'].std()
        stat_upper = mean_cost + 3 * std_cost
        stat_lower = mean_cost - 3 * std_cost
        
        # Isolation Forest detection
        isolation_scores = self.isolation_forest.fit_predict(
            daily_costs[['UnblendedCost']]
        )
        
        # Combine detection methods
        anomalies = daily_costs.copy()
        anomalies['statistical_anomaly'] = (
            (daily_costs['UnblendedCost'] > stat_upper) |
            (daily_costs['UnblendedCost'] < stat_lower)
        )
        anomalies['isolation_anomaly'] = isolation_scores == -1
        
        # Determine severity
        anomalies['severity'] = 'normal'
        anomalies.loc[
            anomalies['statistical_anomaly'] | anomalies['isolation_anomaly'],
            'severity'
        ] = 'medium'
        anomalies.loc[
            anomalies['statistical_anomaly'] & anomalies['isolation_anomaly'],
            'severity'
        ] = 'high'
        
        return {
            'anomalies': anomalies,
            'thresholds': {
                'upper': stat_upper,
                'lower': stat_lower
            },
            'metrics': {
                'total_anomalies': (anomalies['severity'] != 'normal').sum(),
                'high_severity': (anomalies['severity'] == 'high').sum()
            }
        }

# Cost Optimization Component
class CostOptimizer(CostAnalyzer):
    """Identifies and quantifies cost optimization opportunities"""
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyzes cost data for optimization opportunities
        
        Args:
            df: Input DataFrame with cost data
            
        Returns:
            Dict containing optimization recommendations and potential savings
        """
        recommendations = []
        
        # EC2 Instance Optimization
        ec2_recommendations = self._analyze_ec2_optimization(df)
        recommendations.extend(ec2_recommendations)
        
        # Storage Optimization
        storage_recommendations = self._analyze_storage_optimization(df)
        recommendations.extend(storage_recommendations)
        
        # Reserved Instance Opportunities
        ri_recommendations = self._analyze_ri_opportunities(df)
        recommendations.extend(ri_recommendations)
        
        return {
            'recommendations': recommendations,
            'total_savings': sum(rec['annual_savings'] for rec in recommendations)
        }
    
    def _analyze_ec2_optimization(self, df: pd.DataFrame) -> List[Dict]:
        """Analyzes EC2 instance usage for optimization opportunities"""
        ec2_data = df[df['ProductCode'] == 'AmazonEC2']
        recommendations = []
        
        # Instance right-sizing
        instance_usage = ec2_data.groupby('ResourceId').agg({
            'UsageAmount': 'mean',
            'UnblendedCost': 'sum'
        })
        
        underutilized = instance_usage[
            instance_usage['UsageAmount'] < 0.3  # Less than 30% utilization
        ]
        
        if not underutilized.empty:
            recommendations.append({
                'category': 'Instance Rightsizing',
                'description': 'Instances with low CPU utilization',
                'affected_resources': underutilized.index.tolist(),
                'annual_savings': underutilized['UnblendedCost'].sum() * 0.4,
                'priority': 'High',
                'implementation_complexity': 'Medium',
                'action_items': [
                    'Review instance sizing',
                    'Consider downsizing instances',
                    'Implement auto-scaling'
                ]
            })
            
        return recommendations
    
    def _analyze_storage_optimization(self, df: pd.DataFrame) -> List[Dict]:
        """Analyzes storage usage for optimization opportunities"""
        storage_data = df[df['ProductCode'].isin(['AmazonS3', 'AmazonEBS'])]
        recommendations = []
        
        # Analyze S3 storage classes
        s3_data = storage_data[storage_data['ProductCode'] == 'AmazonS3']
        standard_storage = s3_data[
            s3_data['StorageClass'] == 'Standard'
        ]['UnblendedCost'].sum()
        
        if standard_storage > 0:
            recommendations.append({
                'category': 'S3 Storage Optimization',
                'description': 'Optimize S3 storage classes',
                'annual_savings': standard_storage * 0.3,
                'priority': 'Medium',
                'implementation_complexity': 'Low',
                'action_items': [
                    'Implement lifecycle policies',
                    'Move infrequently accessed data to IA storage',
                    'Archive old data to Glacier'
                ]
            })
            
        return recommendations
    
    def _analyze_ri_opportunities(self, df: pd.DataFrame) -> List[Dict]:
        """Analyzes Reserved Instance purchase opportunities"""
        on_demand_usage = df[df['PurchaseOption'] == 'OnDemand']
        recommendations = []
        
        if not on_demand_usage.empty:
            total_od_cost = on_demand_usage['UnblendedCost'].sum()
            recommendations.append({
                'category': 'Reserved Instance Opportunities',
                'description': 'Convert On-Demand to Reserved Instances',
                'annual_savings': total_od_cost * 0.4,
                'priority': 'High',
                'implementation_complexity': 'Medium',
                'action_items': [
                    'Review steady-state workloads',
                    'Calculate optimal RI purchase',
                    'Consider Savings Plans'
                ]
            })
            
        return recommendations

# Visualization Engine
class VisualizationEngine:
    """Generates interactive visualizations for cost analysis"""
    
    def create_dashboard(self, 
                        forecast_results: Dict,
                        anomaly_results: Dict,
                        optimization_results: Dict) -> go.Figure:
        """
        Creates comprehensive interactive dashboard
        
        Args:
            forecast_results: Results from cost forecasting
            anomaly_results: Results from anomaly detection
            optimization_results: Results from cost optimization
            
        Returns:
            go.Figure: Interactive Plotly dashboard
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Cost Forecast',
                'Cost Anomalies',
                'Service Cost Distribution',
                'Optimization Opportunities',
                'Daily Cost Trends',
                'Savings Summary'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # Add forecast plot
        forecast = forecast_results['forecast']
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name='Forecast',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add anomalies plot
        anomalies = anomaly_results['anomalies']
        for severity in ['normal', 'medium', 'high']:
            mask = anomalies['severity'] == severity
            fig.add_trace(
                go.Scatter(
                    x=anomalies[mask]['UsageStartDate'],
                    y=anomalies[mask]['UnblendedCost'],
                    name=f'{severity.title()} Cost',
                    mode='markers'
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1600,
            showlegend=True,
            title_text="AWS Cost Analysis Dashboard",
            template="plotly_white"
        )
        
        return fig

# Report Generator
class ReportGenerator:
    """Generates comprehensive cost analysis reports"""
    
    def __init__(self, template_path: Path = REPORT_TEMPLATE_PATH):
        self.template_path = template_path
        
    def generate_report(self,
                       forecast_results: Dict,
                       anomaly_results: Dict,
                       optimization_results: Dict) -> str:
        """
        Generates detailed PDF report with analysis findings
        
        Args:
            forecast_results: Results from cost forecasting
            anomaly_results: Results from anomaly detection
            optimization_results: Results from cost optimization
            
        Returns:
            str: Path to generated PDF report
        """
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(190, 10, 'AWS Cost Analysis Report', ln=True, align='C')
        
        # Executive Summary
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(190, 10, '1. Executive Summary', ln=True)
        pdf.set_font('Arial', '', 12)
        
        summary_text = f"""
        Analysis Period: {datetime.now().strftime('%Y-%m-%d')}
        Forecast Accuracy (MAPE): {forecast_results['metrics']['mape']:.2%}
        Total Anomalies Detected: {anomaly_results['metrics']['total_anomalies']}
        Potential Annual Savings: ${optimization_results['total_savings']:,.2f}
        """
        
        pdf.multi_cell(190, 10, summary_text)
        
        # Save report
        report_path = f"reports/aws_cost_analysis_{datetime.now().strftime('%Y%m%d')}.pdf"
        pdf.output(report_path)
        
        return report_path

# Main Application Class
class AWSCostAnalyzer:
    """Main application class orchestrating the analysis workflow"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.data_loader = CostDataLoader(SnowflakeConnection())
        self.forecaster = CostForecaster(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.optimizer = CostOptimizer()
        self.visualization = VisualizationEngine()
        self.report_generator = ReportGenerator()
        
    def run_analysis(self) -> Dict:
        """
        Executes complete analysis workflow
        
        Returns:
            Dict containing all analysis results
        """
        try:
            # Load data
            df = self.data_loader.load_cost_data()
            
            # Run analysis components
            forecast_results = self.forecaster.analyze(df)
            anomaly_results = self.anomaly_detector.analyze(df)
            optimization_results = self.optimizer.analyze(df)
            
            # Generate visualizations
            dashboard = self.visualization.create_dashboard(
                forecast_results,
                anomaly_results,
                optimization_results
            )
            
            # Generate report
            report_path = self.report_generator.generate_report(
                forecast_results,
                anomaly_results,
                optimization_results
            )
            
            return {
                'forecast_results': forecast_results,
                'anomaly_results': anomaly_results,
                'optimization_results': optimization_results,
                'dashboard': dashboard,
                'report_path': report_path
            }
            
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            raise

# CLI Interface
def main():
    """Command-line interface for the AWS Cost Analyzer"""
    try:
        # Load configuration
        with open(CONFIG_PATH) as f:
            config = AnalysisConfig(**yaml.safe_load(f))
            
        # Initialize analyzer
        analyzer = AWSCostAnalyzer(config)
        
        # Run analysis
        results = analyzer.run_analysis()
        
        # Print summary
        print("\nAWS Cost Analysis Complete")
        print("--------------------------")
        print(f"Report generated: {results['report_path']}")
        print(f"Total savings identified: ${results['optimization_results']['total_savings']:,.2f}")
        print(f"Forecast accuracy (MAPE): {results['forecast_results']['metrics']['mape']:.2%}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

