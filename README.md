<p align="center">
  <a href="" rel="noopener">
 <img src="https://pbs.twimg.com/media/FommiJ9WIAEBPI0.jpg" alt="Project logo"></a>
</p>
<h1 align="center">Malbecs - Datathon Cajamar 2023 - Fase Local</h1>

<p align="center">
    Repositorio de codigo para competencia de datos Cajamar 2023.
    <br> 
</p>

## Contenido

-   [Contenido](#contenido)
-   [Problema ](#problema-)
-   [Idea / Solucion ](#idea--solucion-)
    -   [Limitaciones ](#limitaciones-)
    -   [Trabajo Futuro ](#trabajo-futuro-)
-   [Setup ](#setup-)
    -   [1. Clonar Repositorio](#1-clonar-repositorio)
    -   [2. Crear entorno de desarrollo](#2-crear-entorno-de-desarrollo)
    -   [3. Instalar `malbecs`](#3-instalar-malbecs)
    -   [4. Datos Iniciales](#4-datos-iniciales)
-   [Entrega ](#entrega-)
    -   [Notebooks](#notebooks)
    -   [Scripts](#scripts)
-   [Autores ](#autores-)

## Problema <a name = "problema"></a>

España es el tercer productor mundial de vino. Disponer de una previsión precisa de la producción en una campaña agrícola es cada vez más necesario de cara a optimizar todos los procesos de la cadena: recolección, traslado, procesado, almacenamiento y distribución.

Dado lo anterior y partiendo de amplios datasets con histórico de producciones de los viñedos que conforman la cooperativa La Viña, así como histórico de la climatología de los mismos, intentaremos crear el mejor modelo de predicción de producción de una campaña en base al cual se pueda estimar la cosecha que dispondrá la cooperativa meses antes de la recolección.

## Idea / Solucion <a name = "idea"></a>

Para la presente entrega, el equipo se enfoco principalmente en la
generacion de nuevas variables a partir de los datos diponibles. Se
lograron incoporar al dataset final para engtrenamiento, un aproxiamdo de
300 variables. Sin embargo, no todas fueron utilizadas para el modelo final.

Se llevo a cabo una destilacion minuciosa de variables siguiendo un principio
de menor a mayor complejidad. Lo que logramos notar, es que los mejores
resultados no se obtuvieron al incoporar gran cantidad de variables, sino que,
con una cantidad justa y las transformaciones necesarias, fue suficiente para
la seleccion final de esta primera fase.

En cuanto a los modelos analizados y seleccionados, se hizo una prueba de multiples algoritmos. Se analizaron modelos lineales, vecinos cercanos, redes neuronales, arboles, maquinas de vector soporte, boosting trees, entre otros. Creemos que la principal dificultad del problema es la cantidad limitada de datos, que dificulta la explotacion de algoritmos mas complejos.

Como resultado, se opto por utilizar un algoritmo de Random Forest clasico, con una busqueda de parametros utilizando periodos de 2019, 2020 y 2021 para evaluar los resultados. Para esta tarea se crearon objectos customizados de validacion cruzada y de transformacion.

### Limitaciones <a name = "limitaciones"></a>

La limitacion principal de la presente entrega es el tiempo. Debido a
restricciones de tiempo el equipo se vio en la necesidad de tomar decisiones
de manera agil y, en algunas situaciones, sin la posibilidad de evaluar
diferentes escenarios.

Nuestro mayor punto de dolor es la falta de explotacion de las variables
relacionadas al clima. Por motivos ya mencionados, no llegaron a ser
analizadas en profundidad y, por ende, no fueron seleccionadas para
el modelo de esta entrega.

### Trabajo Futuro <a name = "futuro"></a>

Dentro de las ideas futuras y posibles cambios, podriamos mencionar dos principales.

La incorporacion de variables climaticas al modelo: Es conocida la relacion y el impacto que tiene el factor climatico sobre la produccion de vino. El mayor desfio es la incorporacion de una variable con alta granularidad temporal a un dataset con granularidad anual. El equipo sostiene que la correcta incorporacion de estas variables, deberia ayudar a una mejora de los resultados.

La imputacion de la variable superficie con estrategias mas apropiada: la variable superficie juega un rol escencial en la produccion final de una finca. Dada su relevancia, la imputacion por promedios agrupados puede arrojar resultados erroneos en aquellos casos donde la finca haya modificado su superficie o su distribcion entre las distintas variedades. Se tuvo muy en cuenta este punto, y se intentaron llevar a cabo otras imputaciones mas complejas, a partir de modelos de regresion. Sin embargo los resultados no llegaron a ser lo sufcientemente explorados para su adoptacion. Sostenemos que, con mas tiempo y dedicacion, la mejora en la imputacion de esta variable deberai arrojar una mejora en los resultados.

Ambos puntos fueron explorados durante esta primer fase, aunque sin arrojar
resultados notablemente mejores, por lo que se opto por no incorporarlos.

## Setup <a name = "setup"></a>

A continuacion se detallan los pasos que se deben realizar para la ejecucion correcta del codigo en el presente repositorio.

### 1. Clonar Repositorio

```bash
$ git clone https://github.com/vitostamatti/datathon-cajamar-malbecs.git
```

### 2. Crear entorno de desarrollo

```bash
$ python -m venv venv
$ source venv/Scripts/activate
$ (venv) pip install -r requirements.txt
```

### 3. Instalar `malbecs`

```
$ (venv) pip install -e ./src/
```

### 4. Datos Iniciales

Colocar en el directorio [`/data/raw/`](./data/raw) los tres ficheros de datos originales

-   UH_2023_TRAIN.TXT
-   DATOS_ETO.TXT
-   DATOS_METO.TXT

Es necesario para la ejecucion del codigo, contar con los datos de origen.

## Entrega <a name="entrega"></a>

En el directorio de [`./notebooks`](./notebooks) se encuentra la entega en formato `.ipynb` tanto para
la exploracion y analisis de datos como para el entrenamiento y prediccion del modelo
para esta primera fase. Por otro lado, en el directorio [`./scripts`](./scripts) se encuentran los
scripts necesarios para la ejecucion de las diferentes transformaciones de datos, el entrenamiento
del modelo final seleccionado para esta fase, y la generacion de predicciones.

### Notebooks

-   [01-Exploracion-Malbecs](/notebooks/01-Exploracion-Malbecs.ipynb): se detalla el propceso de exploracion y visualizacion inicial de los datos, como asi tambien las primeras decisiones sobre tratamiento de variables, generacion y utilizacion de las mismas.
-   [02-Prediccion-Malbecs](/notebooks/02-Prediccion-Malbecs.ipynb): se detalla el paso a paso del preprocesamiento y la ingenieria de variables realizada, como asi tambien el entrenamiento y evaluacion del modelo seleccionado.

### Scripts

-   [run_prep.py](/scripts/run_prep.py)
-   [run_train.py](/scripts/run_train.py)
-   [run_pred_py](/scripts/run_pred.py)
-   [run_all_py](/scripts/run_all.py)

## Autores <a name = "autores"></a>

-   [@vitostamatti](https://github.com/vitostamatti)
-   [@DenisTros](https://github.com/DenisTros)
-   [@sumitkumarjethani](https://github.com/sumitkumarjethani)
