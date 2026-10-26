# Taxi Orders Forecast: Next-Hour Demand at Airports

Hourly demand forecasting for a taxi company, built as part of the Yandex Practicum Data Science program.
The notebook is written in Russian; this README summarizes it in English.

## Business task

A taxi company wants to attract more drivers to airports during peak hours.
To do that, it needs a forecast of the number of orders **for the next hour**.
Requirement: **RMSE ≤ 48** on the test set (the last 10% of the data).

## Data

`taxi.csv` — timestamps of taxi orders at airports, March–August 2018.
After resampling to one hour: 4,416 hourly observations.

## What I did

1. **Preparation.** Parsed dates, checked the index order, gaps and duplicates, resampled to 1-hour intervals.
2. **Time-series analysis.** Seasonal decomposition, rolling mean and standard deviation.
   - Upward trend: the weekly rolling mean grows from ~55 orders/hour in March to ~150 in late August.
   - Strong daily seasonality: peak at midnight (~145 orders/hour), minimum at 6–7 a.m. (~25–30).
   - Weaker weekly seasonality; the series is non-stationary, with large spikes in August.
3. **Feature engineering.** Hour, day of week, lags 1–168 (168 = same hour one week earlier) and a 24-hour rolling mean that excludes the current value, so there is no leakage from the target.
4. **Modeling.** Linear Regression, Random Forest, LightGBM and CatBoost, tuned with `GridSearchCV` on `TimeSeriesSplit` (5 folds, no shuffling).
5. **Testing.** Compared the best model with a naive baseline that predicts the previous hour's value.

## Results

| Model | RMSE (cross-validation) |
|---|---|
| **CatBoost** | **22.40** |
| Linear Regression | 22.55 |
| LightGBM | 22.87 |
| Random Forest | 22.95 |

| Test set | RMSE |
|---|---|
| **CatBoost** | **36.95** |
| Naive baseline (previous hour) | 58.86 |

- The requirement is met: **RMSE 36.95 ≤ 48**, about **37% lower error** than the naive baseline.
- Test error is higher than on cross-validation because the test period (13–31 August) is the busiest and most volatile part of the series.
- The model follows the daily cycle well but underestimates rare extreme spikes (400+ orders/hour). For the business, this means the forecast is reliable in ordinary hours, while unusual peaks still need a safety margin of drivers.

## Tech stack

Python · pandas · NumPy · Matplotlib · statsmodels · scikit-learn · LightGBM · CatBoost

## How to run

```bash
pip install -r requirements.txt
jupyter notebook taxi_orders_forecast.ipynb
```

The notebook downloads the dataset from a public Yandex Practicum link, so no local data files are needed.

---

**Кратко по-русски.** Прогноз числа заказов такси в аэропортах на следующий час (учебный проект Яндекс Практикума). Проанализированы тренд и суточная и недельная сезонность, созданы лаговые и календарные признаки, сравнены четыре модели на `TimeSeriesSplit`. Лучшая модель — CatBoost: RMSE на тесте 36,95 при требовании не больше 48, это на 37% точнее наивного прогноза.
