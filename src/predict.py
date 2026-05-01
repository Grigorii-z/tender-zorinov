"""
Модуль предсказания: загрузка модели и инференс на одном тендере.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from pathlib import Path

# Пути относительно корня проекта
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "catboost_final.cbm"


def load_model():
    """Загружает обученную модель CatBoost."""
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def predict_one(model, tender_data: dict) -> dict:
    df = pd.DataFrame([tender_data])

    df = df[model.feature_names_]
    # Получаем настоящий список категориальных от модели
    cat_indices = model.get_cat_feature_indices()
    cat_features_in_model = [model.feature_names_[i] for i in cat_indices]
    text_indices = model.get_text_feature_indices()
    text_features_in_model = [model.feature_names_[i] for i in text_indices]

    print(f"Модель ожидает категориальные: {cat_features_in_model}")
    print(f"Модель ожидает текстовые: {text_features_in_model}")

    # Приводим к строкам ТОЛЬКО те, что модель действительно считает категориальными
    for col in cat_features_in_model:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in text_features_in_model:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Получаем вероятность
    probability = float(model.predict_proba(df)[:, 1][0])
    # ... остальное
    prediction = int(probability >= 0.5)

    # SHAP-значения для объяснения (топ-5 признаков)
    # CatBoost требует Pool-объект для get_feature_importance с ShapValues
    from catboost import Pool

    shap_pool = Pool(
        df,
        cat_features=cat_features_in_model,
        text_features=text_features_in_model
    )
    shap_values = model.get_feature_importance(
        data=shap_pool, type='ShapValues'
    )
    
    feature_contributions = shap_values[0, :-1]  # убираем bias
    feature_names = model.feature_names_

    # Сортируем по абсолютному вкладу, берём топ-5
    sorted_idx = np.argsort(np.abs(feature_contributions))[::-1][:5]
    top_features = [
        {
            'feature': feature_names[i],
            'contribution': float(feature_contributions[i]),
            'value': df.iloc[0][feature_names[i]]
        }
        for i in sorted_idx
    ]

    return {
        'probability': probability,
        'prediction': prediction,
        'top_features': top_features
    }