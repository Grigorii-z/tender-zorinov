"""
Вспомогательные функции для Streamlit-приложения.
Содержит логику для расчёта производных признаков из ввода пользователя.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
HISTORICAL_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "tender_data.csv"


def load_historical_data() -> pd.DataFrame:
    """Загружает исторические данные для расчёта истории заказчиков."""
    df = pd.read_csv(HISTORICAL_DATA_PATH)
    df['publication_date'] = pd.to_datetime(df['publication_date'])
    df['y_success'] = (df['selection_phase'] == 'Завершена').astype(int)
    return df


def get_customer_history(historical_df: pd.DataFrame, customer_inn: str) -> dict:
    """Возвращает историю заказчика по ИНН."""
    customer_df = historical_df[historical_df['customer_inn'] == customer_inn]

    if len(customer_df) == 0:
        # Нет истории — возвращаем нейтральные значения
        return {
            'customer_total_tenders': 0,
            'customer_success_rate': 0.5,
            'cust_avg_price': None,
        }

    return {
        'customer_total_tenders': len(customer_df),
        'customer_success_rate': customer_df['y_success'].mean(),
        'cust_avg_price': customer_df['start_price'].mean(),
    }


def get_region_history(historical_df: pd.DataFrame, region_code: str) -> dict:
    """Возвращает историю региона."""
    region_df = historical_df[historical_df['customer_region_code'] == region_code]

    if len(region_df) == 0:
        return {
            'region_total_tenders': 0,
            'region_success_rate': 0.5,
        }

    return {
        'region_total_tenders': len(region_df),
        'region_success_rate': region_df['y_success'].mean(),
    }


def get_procedure_history(historical_df: pd.DataFrame, procedure: str) -> float:
    """Возвращает историческую успешность процедуры."""
    proc_df = historical_df[historical_df['procedure'] == procedure]

    if len(proc_df) == 0:
        return 0.5

    return proc_df['y_success'].mean()


def get_cust_proc_history(historical_df: pd.DataFrame, customer_inn: str, procedure: str) -> dict:
    """Возвращает кросс-историю заказчик × процедура."""
    cp_df = historical_df[
        (historical_df['customer_inn'] == customer_inn) &
        (historical_df['procedure'] == procedure)
        ]

    if len(cp_df) == 0:
        return {
            'cust_proc_success_rate': 0.5,
            'cust_proc_count': 0,
        }

    return {
        'cust_proc_success_rate': cp_df['y_success'].mean(),
        'cust_proc_count': len(cp_df),
    }


def get_unique_values(historical_df: pd.DataFrame, column: str) -> list:
    """Возвращает список уникальных значений колонки для выпадающего списка."""
    return sorted(historical_df[column].dropna().unique().tolist())


def build_full_features(user_input: dict, historical_df: pd.DataFrame) -> dict:
    """
    Превращает ввод пользователя в полный набор признаков для модели.

    Args:
        user_input: словарь с базовыми полями от пользователя
        historical_df: исторический датасет для расчёта истории

    Returns:
        Полный словарь со всеми признаками модели
    """
    pub_date = pd.Timestamp(user_input['publication_date'])

    # История заказчика
    cust_hist = get_customer_history(historical_df, user_input['customer_inn'])

    # История региона
    reg_hist = get_region_history(historical_df, user_input['customer_region_code'])

    # История процедуры
    proc_success = get_procedure_history(historical_df, user_input['procedure'])

    # Кросс-история
    cp_hist = get_cust_proc_history(
        historical_df, user_input['customer_inn'], user_input['procedure']
    )

    # Соотношение цены к средней по заказчику
    if cust_hist['cust_avg_price'] is not None and cust_hist['cust_avg_price'] > 0:
        price_vs_avg = user_input['start_price'] / cust_hist['cust_avg_price']
        price_vs_avg = min(price_vs_avg, 100)  # ограничиваем выбросы
    else:
        price_vs_avg = 1.0

    # Все признаки модели
    features = {
        # Финансовые
        'start_price': user_input['start_price'],
        'log_start_price': np.log10(max(user_input['start_price'], 1)),
        'tender_security': user_input['tender_security'],
        'security_ratio': user_input['tender_security'] / user_input['start_price'] if user_input[
                                                                                           'start_price'] > 0 else 0,
        'has_security': int(user_input['tender_security'] > 0),
        'advance_money_pct': user_input['advance_money_pct'],
        'has_advance': int(user_input['advance_money_pct'] > 0),

        # Временные
        'year': pub_date.year,
        'month': pub_date.month,
        'quarter': pub_date.quarter,
        'day_of_week': pub_date.dayofweek,
        'day_of_month': pub_date.day,
        'is_end_of_quarter': int(pub_date.month in [3, 6, 9, 12]),
        'is_end_of_year': int(pub_date.month == 12),

        # History
        'customer_total_tenders': cust_hist['customer_total_tenders'],
        'customer_success_rate': cust_hist['customer_success_rate'],
        'region_total_tenders': reg_hist['region_total_tenders'],
        'region_success_rate': reg_hist['region_success_rate'],
        'procedure_success_rate': proc_success,
        'cust_proc_success_rate': cp_hist['cust_proc_success_rate'],
        'cust_proc_count': cp_hist['cust_proc_count'],
        'price_vs_cust_avg': price_vs_avg,

        # Категориальные — все приводим к строкам, как было при обучении
        'procedure': str(user_input['procedure']),
        'legislation': str(user_input['legislation']),
        'for_small_business': str(user_input['for_small_business']),
        'customer_region_code': str(user_input['customer_region_code']),
        'customer_region': str(user_input['customer_region']),

        # Текстовое
        'tender_name': user_input['tender_name'],
    }

    return features