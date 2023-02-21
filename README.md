# datathon-cajamar-malbecs

Repositorio de codigo para competencia de datos Cajamar 2023

## Environment Setup

```bash
$ python -m venv venv
$ source venv/Scripts/activate
$(venv) pip install -r requirements.txt
```

## Fechas Importantes

-   **Primera Fase del Concurso**: Del 23 de febrero al 16 de marzo de 2023

-   **Fallo del Jurado Local:** 28 de marzo de 2023

-   Las otras no las pongo para no mufar.

## Notes

https://docs.google.com/document/d/1IhF1DT96C64qPPKVZsh3_CnXA-rrEE36HifjB9VuyiU/edit?usp=sharing
Paso lo que esta en el doc:

### Features

-   Feaures de Fecha: dia de semana, mes, número de semana del año
-   Feriados: hay que ver con que nivel de detalle los podemos encontrar y si nos afecta. La producion (lo que genere la vid) va a cambiar mucho en funcion de los feriados.
-   Consumo de vino: hay que ver de donde lo sacamos o si hay algun proxy.
-   Cosechas de otros productos agro: muy buena idea.

### Modelos

-   Autoregressive (AR)
-   Autoregressive Integrated Moving Average (ARIMA)
-   Seasonal Autoregressive Integrated Moving Average (SARIMA)
-   Exponential Smoothing (ES)
-   XGBoost
-   Prophet
-   LSTM (Deep Learning)
-   DeepAR
-   N-BEATS
-   Temporal Fusion Transformer (Google)

Creo que todos estan en implementados en Darts.

### Evaluacion

-   [x] RMSE
-   En el test, agrupar por dia (si es que el consumo viene por hora) y ver el promedio, asi vemos que dia tiene el peor error. Esto puede ir por dia fecha o por dia nombre, mes, etc. Para que no se caguen el negativo con el positivo, hacer ABS de la diferencia entre pred y real, como una columna nueva.

## General

-   Hay que ir pensando como mierda vamos a tratar los datos de pandemia si nos llegan a venir.

## Links

-   [The Best Deep Learning Models for Time Series Forecasting](https://towardsdatascience.com/the-best-deep-learning-models-for-time-series-forecasting-690767bc63f0)
-   [An End-to-End Project on Time Series Analysis and Forecasting with Python](https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b)
-   [Multiple Series? Forecast Them together with any Sklearn Model ](https://towardsdatascience.com/multiple-series-forecast-them-together-with-any-sklearn-model-96319d46269)
-   [TBATS Python: Tutorial & Examples](https://www.ikigailabs.io/multivariate-time-series-forecasting-python/tbats-python)
-   [VAR](https://www.ikigailabs.io/multivariate-time-series-forecasting-python/vector-autoregression-python)

## Kaggle notebooks

Dejo algunos notebooks que encontre en kaggle para robar datita.

-   [EDA and Plotly](https://www.kaggle.com/code/kashishrastogi/store-sales-analysis-time-serie?scriptVersionId=81112640)

-   [Eda y moving average](https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide#11.-Exponential-Moving-Average)

-   [Darts](https://www.kaggle.com/code/ferdinandberr/darts-forecasting-deep-learning-global-models)

-   [Custom Forecast (small dataset)](https://www.kaggle.com/code/cdeotte/seasonal-model-with-validation-lb-1-091#kln-111)

## Libraries & Tools

Dejo algunas liberias que me parecen interesantes para probar.

-   [Darts](https://unit8co.github.io/darts/index.html) : para forecast con un monton de modelos. Tiene su propia api y no es tan facil/intuitiva.
-   [Upgini](https://upgini.com/) : busqueda de nuevas features (externas). Puede que encontremos algo. Tiene mas que nada datos de clima, de "mapas", y dias festivos.
-   [Optuna](https://optuna.org/) : busqueda de hyperparametros. Para tenerla en cuenta aunque no es tan importante al principio creo.
-   [skforecast](https://joaquinamatrodrigo.github.io/skforecast/0.6.0/index.html) : para forecast con "regresores" (sklearn, xbgoost, catboost, etc.) muy facil de usar
-   [sktime](https://www.sktime.net/en/latest/index.html)

## TODOs

-   [ ] Anotarnos cosas pa no olvidar...
