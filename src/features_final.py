"""Финальный список признаков для модели."""

FEATURE_COLS = ['start_price', 'log_start_price', 'tender_security', 'security_ratio', 'has_security', 'advance_money_pct', 'has_advance', 'year', 'month', 'quarter', 'day_of_week', 'day_of_month', 'is_end_of_quarter', 'is_end_of_year', 'customer_total_tenders', 'customer_success_rate', 'region_total_tenders', 'region_success_rate', 'procedure_success_rate', 'procedure', 'legislation', 'for_small_business', 'customer_region_code', 'customer_region', 'cust_proc_success_rate', 'cust_proc_count', 'price_vs_cust_avg']

CAT_FEATURES = ['procedure', 'legislation', 'for_small_business', 'customer_region_code', 'customer_region']

TARGET = "y_success"
