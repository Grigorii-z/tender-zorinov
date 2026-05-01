"""
Streamlit-приложение: Информационная система анализа тендерных закупок.
Дипломный проект Зоринова Г., СПбГУПТД, 2026.
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import date
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.predict import load_model, predict_one
from src.streamlit_helpers import (
    load_historical_data,
    get_unique_values,
    build_full_features,
)

# ============ НАСТРОЙКА СТРАНИЦЫ ============

st.set_page_config(
    page_title="Анализ тендеров",
    page_icon="📊",
    layout="wide"
)
# ============ БОКОВАЯ ПАНЕЛЬ ============

with st.sidebar:
    st.markdown("# 📊 Анализ тендеров")
    st.markdown("---")

    st.markdown("### О приложении")
    st.markdown(
        "Информационная система прогнозирования успешности "
        "тендерных закупок на основе методов искусственного интеллекта."
    )
    st.markdown(
        "Дипломный проект 2026 года.\n\n"
        "**СПбГУПТД**, направление «Прикладная информатика в экономике»"
    )

    st.markdown("---")
    st.markdown("### О модели")
    st.markdown(
        """
- **Алгоритм:** CatBoost (градиентный бустинг)
- **Признаков:** 28
- **AUC-ROC:** 0.71
- **F1-score:** 0.60
- **Recall:** 0.65
- **Кросс-валидация:** AUC 0.728 ± 0.044
- **Период обучения:** 2017–2023
        """
    )

    st.markdown("---")
    st.markdown("### Источник данных")
    st.markdown(
        "Открытые данные о тендерных закупках по 44-ФЗ и 223-ФЗ. "
        "Объём: 4 519 тендеров за 5,5 лет."
    )

    st.markdown("---")
    st.markdown("### Автор")
    st.markdown(
        "**Зоринов Григорий**\n\n"
        "СПбГУПТД, 4 курс, 2026"
    )
# ============ КЕШИРОВАНИЕ ============

@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_historical_data():
    return load_historical_data()


model = get_model()
historical_df = get_historical_data()

# ============ ШАПКА ============

st.title("📊 Анализ тендерных закупок")
st.markdown(
    "Информационная система прогнозирования успешности тендеров "
    "на основе методов искусственного интеллекта CatBoost"
)
st.markdown("---")

# ============ ФОРМА ВВОДА ============

st.header("Параметры тендера")

# Списки уникальных значений из исторических данных
procedures = get_unique_values(historical_df, 'procedure')
regions = get_unique_values(historical_df, 'customer_region')
legislations = get_unique_values(historical_df, 'legislation')

# Раскладка в две колонки
col1, col2 = st.columns(2)

with col1:
    st.subheader("Основные параметры")

    tender_name = st.text_input(
        "Название тендера",
        value="Поставка медицинского оборудования"
    )

    procedure = st.selectbox("Способ отбора", procedures)
    legislation = st.selectbox("Законодательство", legislations)
    customer_region = st.selectbox("Регион заказчика", regions)

    # Маппинг названия региона в код (берём первый попавшийся)
    region_code_raw = historical_df[
        historical_df['customer_region'] == customer_region
        ]['customer_region_code'].iloc[0]

    # Приведение к строке без ".0" в конце (78.0 → "78")
    if pd.isna(region_code_raw):
        region_code = "Unknown"
    else:
        region_code = str(int(region_code_raw)) if isinstance(region_code_raw, float) else str(region_code_raw)

    customer_inn = st.text_input(
        "ИНН заказчика",
        value="7707083893",
        help="10 или 12 цифр. Если ИНН не было в данных — будут нейтральные значения истории."
    )

    pub_date = st.date_input(
        "Дата публикации",
        value=date.today()
    )

with col2:
    st.subheader("Финансовые параметры")

    start_price = st.number_input(
        "Начальная цена контракта (руб)",
        min_value=0,
        max_value=10_000_000_000_000,  # 10 трлн как верхняя планка
        value=10_000_000,
        step=100_000,
        format="%d"
    )

    tender_security = st.number_input(
        "Сумма обеспечения заявки (руб)",
        min_value=0,
        max_value=int(start_price),
        value=0,
        step=10_000,
        format="%d"
    )

    advance_money_pct = st.number_input(
        "Размер аванса (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=5.0,
        format="%.1f"
    ) / 100  # переводим в долю

    for_small_business_label = st.selectbox(
        "Только для малого бизнеса (СМП)?",
        ["Нет", "Да"]
    )
    # Маппинг к формату обучения: модель видела строки "True"/"False"
    for_small_business = "True" if for_small_business_label == "Да" else "False"

# ============ КНОПКА И РЕЗУЛЬТАТ ============

st.markdown("---")

if st.button("🔮 Получить рекомендацию", type="primary", use_container_width=True):

    # Собираем ввод пользователя
    user_input = {
        'tender_name': tender_name,
        'procedure': procedure,
        'legislation': legislation,
        'customer_region': customer_region,
        'customer_region_code': str(region_code),
        'customer_inn': customer_inn,
        'publication_date': pub_date,
        'start_price': start_price,
        'tender_security': tender_security,
        'advance_money_pct': advance_money_pct,
        'for_small_business': for_small_business,
    }

    # Строим полный набор признаков
    features = build_full_features(user_input, historical_df)

    # Делаем предсказание
    with st.spinner("Анализ..."):
        result = predict_one(model, features)

    # Показываем результат — пока в простом виде
        # Цветной баннер с рекомендацией
        if result['probability'] >= 0.7:
            st.success(
                f"### ✅ Рекомендуется участвовать "
                f"(вероятность успеха {result['probability']:.1%})"
            )
        elif result['probability'] >= 0.4:
            st.warning(
                f"### ⚠️ Пограничный случай "
                f"(вероятность успеха {result['probability']:.1%})"
            )
        else:
            st.error(
                f"### ❌ Высокий риск несостоявшегося тендера "
                f"(вероятность успеха {result['probability']:.1%})"
            )

        # Визуальный прогресс-бар
        st.progress(result['probability'])

        # Метрики в три колонки
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(
                label="Вероятность успеха",
                value=f"{result['probability']:.1%}"
            )
        with col_b:
            st.metric(
                label="Прогноз модели",
                value="Состоится" if result['prediction'] == 1 else "Не состоится"
            )
        with col_c:
            # Уровень доверия по типу процедуры
            confidence_map = {
                'Электронный аукцион': 'Высокий',
                'Открытый конкурс в электронной форме': 'Средний',
                'Аукцион в электронной форме': 'Высокий',
            }
            confidence = confidence_map.get(procedure, 'Средний')
            st.metric(
                label="Уровень доверия",
                value=confidence
            )

    st.subheader("Топ-5 факторов, повлиявших на решение")
    for i, feat in enumerate(result['top_features'], 1):
        direction = "↑ повышает" if feat['contribution'] > 0 else "↓ понижает"
        st.write(
            f"**{i}. {feat['feature']}** = {feat['value']} "
            f"({direction} вероятность на {abs(feat['contribution']):.3f})"
        )

st.markdown("---")

with st.expander("ℹ️ Как читать результат?"):
    st.markdown(
        """
**Вероятность успеха** — оценка модели CatBoost вероятности того, что 
тендер будет успешно завершён (т.е. найдётся победитель и контракт будет заключён).

**Топ-5 факторов** — это признаки, которые сильнее всего повлияли на решение 
модели именно для этого тендера. Стрелка ↑ означает, что признак повысил 
вероятность успеха, ↓ — понизил.

**Уровень доверия модели:**
- На «Электронный аукцион» — высокая надёжность (ошибка 24,5%)
- На «Открытый конкурс в электронной форме» — средняя (ошибка 38,9%)
- Чем стандартнее процедура, тем точнее прогноз

**Ограничения модели:**
- Модель обучена на данных 2017–2023 годов и не учитывает изменения, 
произошедшие позже.
- В периоды экономической турбулентности качество прогнозов снижается.
- Окончательное решение об участии должен принимать специалист 
с учётом всех аспектов конкретной закупки.
        """
    )