# datathon-cajamar-malbecs

Repositorio de codigo para competencia de datos Cajamar 2023

## Contenido

El repositorio cuenta con una serie de notebooks de exploracion, feature engineering y
modelado, en los cuales se pueden encontrar las diferentes pruebas y evaluaciones
realizadas por el equipo.

## Environment Setup

1.  Clonar repositorio.

```bash
$ git clone https://github.com/vitostamatti/datathon-cajamar-malbecs.git
```

2. Crear entorno de desarrollo

```bash
$ python -m venv venv
$ source venv/Scripts/activate
$ (venv) pip install -r requirements.txt
```

3. Instalar librearia `malbecs`

```
$ (venv) pip install -e ./src/
```

## Entrega

En el directorio de [`./notebooks`]("./notebooks") se encuentra la entega en formato `.ipynb` tanto para
la exploracion y analisis de datos como para el entrenamiento y prediccion del modelo
para esta primera fase. Por otro lado, en el directorio [`./scripts`]("./scripts") se encuentran los
scripts necesarios para la ejecucion de las diferentes transformaciones de datos, el entrenamiento
del modelo final seleccionado para esta fase, y la generacion de predicciones.

### Notebooks

-   [01-Exploracion-Malbecs]("../notebooks/01-Exploracion-Malbecs.ipynb")
-   [02-Prediccion-Malbecs]("../notebooks/02-Prediccion-Malbecs.ipynb")

### Scripts

-   [run_prep.py]("../../scripts/run_prep.py")
-   [run_train.py]("../../scripts/run_train.py")
-   [run_pred_py]("../../scripts/run_pred.py")
-   [run_all_py]("../../scripts/run_all.py")
