# imports
import os
import numpy as np
from malbecs.modeling import train as tr
from malbecs.modeling import models as mm
import logging
import logging.config
import argparse

from memory_profiler import profile

parser = argparse.ArgumentParser(
    prog='Predicciones',
    description="""
    Script para predecir utilizando el modelo propuesto por Malbecs. Se debe especificar la ruta de los ficheros procesados, 
    como asi tambien la ruta la modelo final entrenado y la ruta destino de las predicciones.
    """
)

parser.add_argument(
    '-wi', '--wine-final',
    default="./data/final/wine_final.csv",
    help='Ruta a los datos procesados de UH_2023_TRAIN.txt'
)
parser.add_argument(
    '-ei', '--eto-final',
    default="./data/final/eto_final.csv",
    help='Ruta a los datos procesados de DATOS_ETO.txt'
)
parser.add_argument(
    '-mi', '--meteo-final',
    default="./data/final/meteo_final.csv",
    help='Ruta a los datos de DATOS_METEO.txt'
)
parser.add_argument(
    '-m', '--model',
    default="./data/models/model_final.pkl",
    help='Ruta origen del modelo entrenado'
)
parser.add_argument(
    '-po', '--preds-out',
    default="./data/final/Malbecs.txt",
    help='Ruta destino de las predicciones.'
)
args = parser.parse_args()

@profile
def run_pred(wine_path, eto_path, meteo_path, model_path, preds_path):

    logger = logging.getLogger(os.path.basename(__file__))

    final_wine_path = wine_path
    final_eto_path = eto_path
    final_meteo_path = meteo_path

    logger.info(
        f'Loading files:\n \t\t\t{wine_path}\n \t\t\t{eto_path}\n \t\t\t{meteo_path}')

    X_final, _ = tr.load_xy(
        wine_path=final_wine_path,
        eto_path=final_eto_path,
        meteo_path=final_meteo_path,
        min_camp=22,
        max_camp=22
    )

    logger.info(f'Loading model {model_path}')
    
    model = mm.load_trained_model(model_path)

    y_pred_final = model.predict(X_final)

    preds_final = X_final[['id_finca', 'variedad',
                              'modo', 'tipo', 'color', 'superficie']].copy()

    preds_final['produccion'] = y_pred_final

    preds_final = preds_final.sort_values(
        ['id_finca', 'variedad', 'modo', 'tipo', 'color', 'superficie'], ascending=True)

    logger.info(f'Saving predictions to {preds_path}')
    preds_final.to_csv(preds_path, header=None, index=None, sep='|', mode='w')


if __name__ == "__main__":
    if os.path.exists('./config/log.conf'):
        logging.config.fileConfig('./config/log.conf')

    run_pred(args.wine_final, args.eto_final,
             args.meteo_final, args.model, args.preds_out)
