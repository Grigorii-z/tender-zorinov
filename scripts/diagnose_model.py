"""
Диагностика обученной CatBoost-модели: сравнивает признаки внутри
.cbm-файла со списками из src/features_final.py и печатает расхождения.

Используется для отладки ошибки вида:
    _catboost.CatBoostError: bad object for id: 0.5
которая возникает, когда признак подаётся как float, а модель ждёт его
как категориальный/текстовый.
"""
import sys
from pathlib import Path

from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features_final import FEATURE_COLS, CAT_FEATURES, TEXT_FEATURES

MODEL_PATH = PROJECT_ROOT / "models" / "catboost_final.cbm"


def main() -> None:
    print(f"Загружаю модель: {MODEL_PATH}")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    feature_names = list(model.feature_names_)
    cat_indices = list(model.get_cat_feature_indices())
    text_indices = list(model.get_text_feature_indices())

    cat_in_model = [feature_names[i] for i in cat_indices]
    text_in_model = [feature_names[i] for i in text_indices]

    print("\n" + "=" * 70)
    print("ВСЕ ПРИЗНАКИ МОДЕЛИ (model.feature_names_)")
    print("=" * 70)
    for i, name in enumerate(feature_names):
        marker = ""
        if i in cat_indices:
            marker = " [CAT]"
        elif i in text_indices:
            marker = " [TEXT]"
        print(f"  {i:>3}. {name}{marker}")

    print("\n" + "=" * 70)
    print("КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ В МОДЕЛИ")
    print("=" * 70)
    print(f"  индексы: {cat_indices}")
    print(f"  имена:   {cat_in_model}")

    print("\n" + "=" * 70)
    print("ТЕКСТОВЫЕ ПРИЗНАКИ В МОДЕЛИ")
    print("=" * 70)
    print(f"  индексы: {text_indices}")
    print(f"  имена:   {text_in_model}")

    print("\n" + "=" * 70)
    print("СОДЕРЖИМОЕ src/features_final.py")
    print("=" * 70)
    print(f"  FEATURE_COLS  ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"  CAT_FEATURES  ({len(CAT_FEATURES)}): {CAT_FEATURES}")
    print(f"  TEXT_FEATURES ({len(TEXT_FEATURES)}): {TEXT_FEATURES}")

    print("\n" + "=" * 70)
    print("РАСХОЖДЕНИЯ")
    print("=" * 70)

    cat_in_model_set = set(cat_in_model)
    cat_in_file_set = set(CAT_FEATURES)
    text_in_model_set = set(text_in_model)
    text_in_file_set = set(TEXT_FEATURES)
    all_features_in_model_set = set(feature_names)
    all_features_in_file_set = set(FEATURE_COLS)

    only_in_model_cat = cat_in_model_set - cat_in_file_set
    only_in_file_cat = cat_in_file_set - cat_in_model_set

    only_in_model_text = text_in_model_set - text_in_file_set
    only_in_file_text = text_in_file_set - text_in_model_set

    only_in_model_all = all_features_in_model_set - all_features_in_file_set
    only_in_file_all = all_features_in_file_set - all_features_in_model_set

    print("\n[CAT_FEATURES]")
    if not only_in_model_cat and not only_in_file_cat:
        print("  OK — списки совпадают")
    else:
        if only_in_model_cat:
            print(f"  ⚠ модель считает КАТЕГОРИАЛЬНЫМИ, но их НЕТ в CAT_FEATURES:")
            for n in sorted(only_in_model_cat):
                print(f"      - {n}")
        if only_in_file_cat:
            print(f"  ⚠ в CAT_FEATURES записано, но модель НЕ считает категориальными:")
            for n in sorted(only_in_file_cat):
                print(f"      - {n}")

    print("\n[TEXT_FEATURES]")
    if not only_in_model_text and not only_in_file_text:
        print("  OK — списки совпадают")
    else:
        if only_in_model_text:
            print(f"  ⚠ модель считает ТЕКСТОВЫМИ, но их НЕТ в TEXT_FEATURES:")
            for n in sorted(only_in_model_text):
                print(f"      - {n}")
        if only_in_file_text:
            print(f"  ⚠ в TEXT_FEATURES записано, но модель НЕ считает текстовыми:")
            for n in sorted(only_in_file_text):
                print(f"      - {n}")

    print("\n[FEATURE_COLS — общий набор]")
    if not only_in_model_all and not only_in_file_all:
        print("  OK — наборы совпадают")
    else:
        if only_in_model_all:
            print(f"  ⚠ есть в модели, но НЕТ в FEATURE_COLS:")
            for n in sorted(only_in_model_all):
                print(f"      - {n}")
        if only_in_file_all:
            print(f"  ⚠ есть в FEATURE_COLS, но НЕТ в модели:")
            for n in sorted(only_in_file_all):
                print(f"      - {n}")

    print("\n" + "=" * 70)
    print("ИСТОЧНИК ПРАВДЫ ДЛЯ predict.py — модель")
    print("=" * 70)
    print(f"CAT_FEATURES (из модели)  = {cat_in_model}")
    print(f"TEXT_FEATURES (из модели) = {text_in_model}")


if __name__ == "__main__":
    main()
