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

- [Contenido](#contenido)
- [Problema ](#problema-)
- [Idea / Solucion ](#idea--solucion-)
  - [Trabajo Futuro ](#trabajo-futuro-)
- [Setup ](#setup-)
  - [1. Pre-Requisitos](#1-pre-requisitos)
  - [2. Crear entorno de python](#2-crear-entorno-de-python)
  - [3. Instalar Dependencias](#3-instalar-dependencias)
  - [4. Datos Iniciales](#4-datos-iniciales)
- [Entrega ](#entrega-)
  - [Presentacion](#presentacion)
  - [Notebooks](#notebooks)
  - [Scripts](#scripts)
- [Agradecimientos ](#agradecimientos-)
- [Autores ](#autores-)

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

### Trabajo Futuro <a name = "futuro"></a>

Dentro de las ideas futuras y posibles cambios, podriamos mencionar uno principal.

La imputacion de la variable superficie con estrategias mas apropiada: la variable superficie juega un rol escencial en la produccion final de una finca. Dada su relevancia, la imputacion por promedios agrupados puede arrojar resultados erroneos en aquellos casos donde la finca haya modificado su superficie o su distribcion entre las distintas variedades. Se tuvo muy en cuenta este punto, y se intentaron llevar a cabo otras imputaciones mas complejas, a partir de modelos de regresion. Sin embargo los resultados no llegaron a ser lo sufcientemente explorados para su adoptacion. Sostenemos que, con mas tiempo y dedicacion, la mejora en la imputacion de esta variable deberai arrojar una mejora en los resultados.

## Setup <a name = "setup"></a>

A continuacion se detallan los pasos que se deben realizar para la ejecucion correcta del codigo en el presente repositorio.

### 1. Pre-Requisitos

Se requiere contar con `python` instalado. Se puede descargar el instalador desde la [pagina oficial](https://www.python.org/downloads/).

### 2. Crear entorno de python

Crear un entorno de python si se desea aislar las dependencias del projecto.

```bash
$ python -m venv venv
$ source venv/Scripts/activate
```

### 3. Instalar Dependencias

Instalar librerias de python.

```bash
$ (venv) pip install -r requirements.txt
```

Instalar `malbecs` (libreria propia)

```
$ (venv) pip install -e ./src/
```

Para mas detalle sobre la libreria malbecs, puede acceder a la documentacion disponible en [`pydocs`](http://malbecs-docs-cajamar-2023.s3-website-us-east-1.amazonaws.com/)

### 4. Datos Iniciales

Colocar en el directorio [`/data/raw/`](./data/raw) los tres ficheros de datos originales

-   UH_2023_TRAIN.TXT
-   DATOS_ETO.TXT
-   DATOS_METO.TXT

Es necesario para la ejecucion del codigo, contar con los datos de origen.

## Entrega <a name="entrega"></a>

La entrega cuenta con dos _scripts_ y con dos formatos de ejecution. El _script_ de exploracion, y el _script_ de prediccion. Los formatos de ejecucion son: `notebooks` y `.py`.

En el directorio de [`./notebooks`](./notebooks) se encuentra la entrega en formato `.ipynb` tanto para la exploracion y analisis de datos como para el entrenamiento y prediccion del modelo para esta primera fase.

En el directorio [`./scripts`](./scripts) se encuentran los scripts necesarios para la ejecucion de las diferentes transformaciones de datos, el entrenamiento del modelo final seleccionado para esta fase, y la generacion de predicciones.

Ademas, en el directorio [/docs](./docs/) se encuentra una presentacion resumen de la entrega realizada resumiendo los puntos importantes del proceso llevado a cabo por el equipo.

### Presentacion

La presentacion [Datathon Cajamar 2023 - Malbecs - Fase Nacional](./docs/Datathon%20Cajamar%202023%20-%20Malbecs%20-%20Fase%20Nacional.pdf) es un resumen de la presente entrega, donde se comentan algunas de las tecnicas utilizadas, como asi tambien los resultados obtenidos en terminos generales.

### Notebooks

-   [01-Exploracion-Malbecs](/notebooks/01-Exploracion-Malbecs.ipynb): se detalla el propceso de exploracion y visualizacion inicial de los datos, como asi tambien las primeras decisiones sobre tratamiento de variables, generacion y utilizacion de las mismas.

Dentro de este, se encuentran comentados y analizados los nuevos conceptos a tener en cuenta en la entrega: **Transparencia, Explicabilidad, Sostenibilidad y Justicia**
Para justicia y explicabilidad hemos creado apartados en este script de prediccion, mientras que para sostenibilidad hemos decorado los archivos .py ejecutables con memorias que entregan el consumo energetico y computacional. Por último, para asegurar una transparencia en nuestro código y modelo, generamos la documentación pertinente para todas las funciones creadas, generamos los comentarios necesarios en los notebooks, y creamos un dashboard adjunto en PowerBi para analizar las producciones y errores en un mayor grado de detalle.

-   [02-Prediccion-Malbecs](/notebooks/02-Prediccion-Malbecs.ipynb): se detalla el paso a paso del preprocesamiento y la ingenieria de variables realizada, como asi tambien el entrenamiento y evaluacion del modelo seleccionado.

### Scripts

Para la ejecucion de los script, lo mas conveniente es utilizar los parametros por defecto utilizando el siguiente comando.

```bash
$ (venv) python ./scripts/<script.py>
```

Si quiere mas informacion sobre los parametros utilizados, ejecutar el siguiente comando.

```bash
$ (venv) python ./scripts/<script.py> --help
```

Si quiere ejecutar el script obteniendo informacion de la utilizaicon de memoria, ejecutar el siguiente comando.

```bash
$ (venv) python -m memory_profiler ./scripts/<script.py>
```

Cada script cuenta con un log de ejecucion que muestra por consola los
pasos realizados.

-   [run_prep.py](/scripts/run_prep.py) : preproceso e ingenieria de variables de todos los datos de entrada.
-   [run_train.py](/scripts/run_train.py): validacion y entrenamiento del modelo final seleccionado.
-   [run_pred_py](/scripts/run_pred.py): generacion de predicciones para 2022 a partir del modelo entrenado.
-   [run_all_py](/scripts/run_all.py): ejecucion end-to-end de todos los pasos.

## Agradecimientos <a name = "agradecimientos"></a>

Aprovechamos para agradecer a todos los encargados organizacion y realizacion del evento por la oportunidad de poder participar, y para felicitarlos por
la excelente ejecucion del mismo. Fue una experiencia enriquecedora para el equipo y una competencia realmente interesante y desafiante en todos los aspectos.

## Autores <a name = "autores"></a>

-   [@vitostamatti](https://github.com/vitostamatti)
-   [@DenisTros](https://github.com/DenisTros)
-   [@sumitkumarjethani](https://github.com/sumitkumarjethani)
