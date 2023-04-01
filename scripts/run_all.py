# imports
import argparse
import os
import numpy as np


from sklearn.model_selection import cross_validate

from malbecs.modeling import train as tr
from malbecs.modeling import models as mm
from malbecs.preprocess import wine as wine_pr
from malbecs.feateng import wine as wine_fe
from malbecs.preprocess import eto as eto_pr
from malbecs.feateng import eto as eto_fe
from malbecs.preprocess import meteo as meteo_pr

import logging
import logging.config

seed = 42

parser = argparse.ArgumentParser(
    prog='Pipeline Completo',
    description="""
    Script para ejecutar el pipeline completo que incluye desde el preprocesado de los datos, el entrenamiento del modelo 
    y la prediccion sobre el periodo 2022. No se guardaran ficheros intermedios.
    """
)


parser.add_argument(
    '-wi', '--wine-raw',
    default="./data/raw/UH_2023_TRAIN.txt",
    help='Ruta a los datos de UH_2023_TRAIN.txt'
)
parser.add_argument(
    '-ei', '--eto-raw',
    default="./data/raw/DATOS_ETO.TXT",
    help='Ruta a los datos de DATOS_ETO.txt'
)
parser.add_argument(
    '-mi', '--meteo-raw',
    default="./data/raw/DATOS_METEO.TXT",
    help='Ruta a los datos de DATOS_METEO.txt'
)

parser.add_argument(
    '-po', '--preds-out',
    default="./data/final/Malbecs.txt",
    help='Ruta destino de las predicciones.'
)

args = parser.parse_args()


def run_all(wine_raw, eto_raw, meteo_raw, preds_path):

    logger = logging.getLogger(os.path.basename(__file__))

    logger.info('Starting preprocessing...')

    logger.info(f'Loading {wine_raw}')
    wine_data = wine_pr.load_wine_dataset(wine_raw)

    logger.info(f'Preprocessing {wine_raw}')
    wine_data = wine_pr.preproces_wine_data(
        wine_data
    )

    logger.info(f'Generating Features for {os.path.basename(wine_raw)}')
    wine_data = wine_fe.feateng_wine_data(
        wine_data)

    logger.info(f'Loading {eto_raw}')
    eto_data = eto_pr.load_eto_dataset(eto_raw)

    logger.info(f'Preprocessing {eto_raw}')
    eto_data = eto_pr.preprocess_eto_dataset(
        eto_data, eto_pr.cols_mean, eto_pr.cols_sum)

    logger.info(f'Generating Features for {eto_raw}')
    eto_data = eto_fe.feateng_eto(eto_data)

    logger.info(f'Loading {meteo_raw}')
    meteo_data = meteo_pr.load_meteo_data(meteo_raw)

    logger.info(f'Preprocessing {meteo_raw}')
    meteo_data = meteo_pr.preproces_meteo_data(meteo_data)

    data = tr.merge_data(wine_data, eto_data=eto_data, meteo_data=meteo_data)

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

    data_final = tr.filter_camp(data, min_camp=22, max_camp=22)

    X_final, _ = tr.xy_split(data_final)

    X_final[cat_cols] = X_final[cat_cols].astype('category')

    preds_final = []
    for model in models:
        preds_final.append(model.predict(X_final))

    y_pred_final = np.mean(preds_final, 0)

    preds_final = data_final[['id_finca', 'variedad',
                              'modo', 'tipo', 'color', 'superficie']].copy()

    preds_final['produccion'] = y_pred_final

    preds_final = preds_final.sort_values(
        ['id_finca', 'variedad', 'modo', 'tipo', 'color', 'superficie'], ascending=True)

    logger.info(f'Saving predictions to {preds_path}')
    preds_final.to_csv(preds_path, header=None, index=None, sep='|', mode='w')


if __name__ == "__main__":
    if os.path.exists('./config/log.conf'):
        logging.config.fileConfig('./config/log.conf')
    run_all(args.wine_raw, args.eto_raw, args.meteo_raw, args.preds_out)
