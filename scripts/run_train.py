# imports
import os
import numpy as np
from sklearn.model_selection import cross_validate

from malbecs.modeling import train as tr
from malbecs.modeling import models as mm
import logging
import logging.config
import os

import argparse


parser = argparse.ArgumentParser(
    prog='Entrenar Modelo',
    description="""
    Script para entrenar el modelo propuesto por Malbecs. Se debe especificar la ruta de los ficheros procesados.
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
    '-mo', '--model',
    default="./data/models/model_final.pkl",
    help='Ruta destino del modelo entrenado'
)

args = parser.parse_args()


def run_train(wine_final, eto_final, meteo_final, model_final):

    logger = logging.getLogger(os.path.basename(__file__))

    final_wine_path = wine_final
    final_eto_path = eto_final
    final_meteo_path = meteo_final

    logger.info(
        f'Loading files:\n \t\t\t{wine_final}\n \t\t\t{eto_final}\n \t\t\t{meteo_final}\n')
    data = tr.load_final_data(
        wine_path=final_wine_path,
        eto_path=final_eto_path,
        meteo_path=final_meteo_path
    )

    data_train = tr.filter_camp(data.copy(), min_camp=15, max_camp=21)

    X, y = tr.xy_split(data_train)

    cat_cols = [
        'id_finca',
        'id_zona',
        'id_estacion',
        'variedad',
        "modo",
        "tipo",
        "color",
        "prod_shift1_gt_shift2"
    ]

    X[cat_cols] = X[cat_cols].astype('category')

    train_idxs, test_idxs = tr.CampKFold.get_train_test(
        X['campaña'], from_camp=19, to_camp=21
    )

    cv = tr.CampKFold(train_idxs, test_idxs)

    m = mm.get_final_model()

    logger.info(f'Cross Validating model')
    res = cross_validate(
        estimator=m,
        X=X,
        y=y,
        cv=cv,
        n_jobs=-1,
        scoring=tr.rmse_scorer,
        return_train_score=True,
        return_estimator=True
    )

    logger.info(f"Train RMSE: {res.get('train_score')}")
    logger.info(f"Test RMSE: {res.get('test_score')}")
    logger.info(f"Train Mean RMSE: {np.mean(res.get('train_score'))}")
    logger.info(f"Test Mean RMSE: {np.mean(res.get('test_score'))}")

    logger.info(f"Training on full dataset")
    models = []
    for i in range(10):
        m = mm.get_final_model()
        m.set_params(randomforestregressor__random_state=mm.seed*(1+i))
        m.fit(X, y)
        models.append(m)

    logger.info(f"Saving model to {model_final}")
    mm.save_trained_model(models, model_final)


if __name__ == "__main__":
    if os.path.exists('./config/log.conf'):
        logging.config.fileConfig('./config/log.conf')

    run_train(args.wine_final, args.eto_final, args.meteo_final, args.model)
