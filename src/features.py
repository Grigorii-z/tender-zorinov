"""
Константы проекта: списки признаков для модели.
Автогенерация из notebooks/03_feature_engineering.ipynb
"""

FEATURE_COLS = ['start_price', 'log_start_price', 'tender_security', 'security_ratio', 'has_security', 'advance_money_pct', 'has_advance', 'year', 'month', 'quarter', 'day_of_week', 'day_of_month', 'is_end_of_quarter', 'is_end_of_year', 'customer_total_tenders', 'customer_success_rate', 'region_total_tenders', 'region_success_rate', 'procedure_success_rate', 'procedure', 'legislation', 'for_small_business', 'customer_region_code', 'customer_region']

CAT_FEATURES = ['procedure', 'legislation', 'for_small_business', 'customer_region_code', 'customer_region']

TARGET = 'y_success'
