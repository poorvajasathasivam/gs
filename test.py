# config/config.yaml
"""
snowflake:
  account: your_account
  database: aws_cur_db
  schema: public
  warehouse: XSMALL_WH

analysis:
  forecast_periods: 30
  anomaly_sensitivity: 0.1
  min_saving_threshold: 1000.0
  report_format: pdf

logging:
  level: INFO
  file: logs/aws_cost_analysis.log
"""

# tests/test_analyzer.py
import pytest
import pandas as pd
from datetime import datetime, timedelta
from aws_cost_analyzer import (
    AWSCostAnalyzer,
    AnalysisConfig,
    DataValidator,
    CostForecaster,
    AnomalyDetector,
    CostOptimizer
)

@pytest.fixture
def sample_data():
    """Creates sample AWS cost data for testing"""
    dates = pd.date_range(
        start='2024-01-01',
        end='2024-12-31',
        freq='D'
    )
    
    return pd.DataFrame({
        'UsageStartDate': dates,
        'UsageAccountId': 'test-account',
        'ProductCode': 'AmazonEC2',
        'UnblendedCost': np.random.uniform(100, 1000, len(dates)),
        'UsageAmount': np.random.uniform(0, 100, len(dates))
    })

def test_data_validator():
    """Tests data validation functionality"""
    validator = DataValidator()
    
    # Test valid data
    valid_data = sample_data()
    assert validator.validate_cost_data(valid_data) == True
    
    # Test invalid data
    invalid_data = valid_data.drop('UnblendedCost', axis=1)
    with pytest.raises(ValueError):
        validator.validate_cost_data(invalid_data)

def test_cost_forecaster():
    """Tests cost forecasting functionality"""
    config = AnalysisConfig()
    forecaster = CostForecaster(config)
