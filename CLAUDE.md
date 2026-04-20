# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
# Activate virtual environment (all commands assume this is active)
source .venv/Scripts/activate
```

## Common Commands

```bash
# Run a Streamlit app
streamlit run <app.py>

# Launch Jupyter Lab
jupyter lab

# Run a Python script
python <script.py>

# Install/update dependencies
pip install -r requirements.txt
```

## Tech Stack

- **Python 3.10** with a pre-configured `.venv`
- **Data/ML:** Pandas, NumPy, Scikit-learn, CatBoost, Optuna, SHAP, Numba
- **Visualization:** Matplotlib, Plotly, Altair, Seaborn
- **Web UI:** Streamlit
- **Interactive analysis:** Jupyter/JupyterLab
- **Database:** SQLAlchemy (ORM) + Alembic (migrations)
- **CLI:** Click
- **HTTP:** Requests, HTTPx

## Architecture Notes

- `requirements.txt` is UTF-16 little-endian encoded — use a tool that handles this encoding when reading/modifying it.
- No source code exists yet; the project is in initial setup phase with dependencies pre-installed.

# Дипломный проект: Информационная система анализа тендерных закупок

## Контекст
Студент 4 курса СПбГУПТД, направление "Прикладная информатика в экономике".
Дипломная работа: "Информационная система анализа тендерных закупок на основе методов искусственного интеллекта".

Цель: разработать локальное приложение, которое на основе методов машинного обучения (CatBoost) прогнозирует вероятность победы в тендере и риск расторжения контракта, на основе чего даёт рекомендацию поставщику об участии.

## Сроки
- До 8 мая 2026 — рабочий прототип для показа научному руководителю
- До 31 мая 2026 — финальная версия + 2 глава диплома

## Технологический стек
- Python 3.11 (виртуальное окружение в .venv)
- CatBoost — основная ML-библиотека
- pandas, numpy — обработка данных
- scikit-learn — метрики и вспомогательные функции
- SHAP — интерпретация модели
- Streamlit — локальный веб-интерфейс
- Jupyter — ноутбуки для EDA

## Структура проекта
- data/raw/ — исходные датасеты (не коммитить в git)
- data/processed/ — очищенные данные
- notebooks/ — исследовательские Jupyter-ноутбуки
- src/ — модули production-кода (data_loader, preprocessing, model, predict)
- models/ — сохранённые CatBoost-модели (.cbm файлы, не коммитить)
- reports/ — графики, метрики, скриншоты
- app.py — Streamlit-приложение
- journal.md — дневник разработки для 2 главы диплома

## Модели
Две бинарные классификации на CatBoost:
1. y_win — вероятность победы поставщика в тендере
2. y_broken — вероятность расторжения контракта

Итоговая рекомендация формируется по обеим моделям.

## Правила для кода
- Все функции снабжай type hints
- Docstrings на русском в Google-стиле
- Комментарии в местах, где логика неочевидна
- Категориальные признаки передавай через параметр cat_features CatBoost
- Разделение train/test — строго по времени (TimeSeriesSplit), не случайно
- Все случайности фиксируй через random_state=42 для воспроизводимости

## Важный контекст
Проект делается одновременно с прохождением практики, поэтому автор не всегда имеет полный рабочий день для кодинга. Приоритет — работающий код, даже если не идеальный, который можно дорабатывать итеративно.