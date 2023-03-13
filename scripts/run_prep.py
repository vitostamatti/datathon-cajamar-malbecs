# imports
from malbecs.preprocess import wine as wine_pr
from malbecs.feateng import wine as wine_fe
from malbecs.preprocess import eto as eto_pr
from malbecs.feateng import eto as eto_fe
from malbecs.preprocess import meteo as meteo_pr
import argparse
import logging
import logging.config
import os


def run_preprocessing(wine_raw, wine_final, eto_raw, eto_final, meteo_raw, meteo_final):

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
        wine_data, output_path=wine_final
    )
    logger.info(
        f'Processed {wine_raw} saved to {wine_final}.')

    logger.info(f'Loading {eto_raw}')
    eto_data = eto_pr.load_eto_dataset(eto_raw)

    logger.info(f'Preprocessing {eto_raw}')
    eto_data = eto_pr.preprocess_eto_dataset(
        eto_data, eto_pr.cols_mean, eto_pr.cols_sum)

    logger.info(f'Generating Features for {eto_raw}')
    eto_data = eto_fe.feateng_eto(eto_data, output_path=eto_final)
    logger.info(
        f'Processed {eto_raw} saved to {eto_final}')

    logger.info(f'Loading {meteo_raw}')
    meteo_data = meteo_pr.load_meteo_data(meteo_raw)

    logger.info(f'Preprocessing {meteo_raw}')
    meteo_data = meteo_pr.preproces_meteo_data(meteo_data, meteo_final)
    logger.info(
        f'Processed {meteo_raw} saved to {meteo_final}')

    return meteo_data


parser = argparse.ArgumentParser(
    prog='Preprocesar Data',
    description="""
    Script para transformar los datos crudos en datos preparados para el entrenamiento y prediccion.\n
    Es necesario especificar las rutas a los tres ficheros crudos originales: UH_2023_TRAIN.txt,
    DATOS_ETO.txt, y DATOS_METEO.txt. Ademas es necesario especificar las rutas a los ficheros de 
    salida correspondientes a cada uno.
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
    '-wo', '--wine-final',
    default="./data/final/wine_final.csv",
    help='Ruta de salida de los datos procesados de UH_2023_TRAIN.txt')

parser.add_argument(
    '-eo', '--eto-final',
    default="./data/final/eto_final.csv",
    help='Ruta de salida de los datos procesados de DATOS_ETO.txt')

parser.add_argument(
    '-mo', '--meteo-final',
    default="./data/final/meteo_final.csv",
    help='Ruta de salida de los datos procesados de DATOS_METEO.txt')

args = parser.parse_args()

if __name__ == "__main__":
    if os.path.exists('./config/log.conf'):
        logging.config.fileConfig('./config/log.conf')

    meteo_data = run_preprocessing(args.wine_raw, args.wine_final,
                                   args.eto_raw, args.eto_final, args.meteo_raw, args.meteo_final)
