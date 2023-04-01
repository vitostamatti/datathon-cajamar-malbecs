import pandas as pd
from malbecs.utils import fillna_by_group, fillna_by_value, replace_zeros_with_na
import os


def get_sup_tot_camp_finca(data):
    tot_sup_camp_finca = (
        data.groupby(['id_finca', 'campaña'])
        .agg({"superficie": 'sum'})
        .rename(columns={"superficie": "sup_tot_camp_finca"})
        .reset_index()
    )
    return data.merge(
        tot_sup_camp_finca,
        left_on=['id_finca', 'campaña'],
        right_on=['id_finca', 'campaña'],
        how='left'
    )


def get_sup_tot_finca(data):
    tot_sup_finca = (
        data.groupby(['id_finca', 'campaña'])
        .agg({"superficie": 'sum'})
        .rename(columns={"superficie": "superficie_total"})
        .groupby('id_finca')
        .agg({"superficie_total": "mean"})
        .reset_index()
    )
    return data.merge(
        tot_sup_finca,
        left_on=['id_finca'],
        right_on=['id_finca'],
        how='left'
    )


def get_n_var_finca_camp(data):
    n_variedad_finca = (
        data.groupby(['id_finca', 'campaña'])
        .agg({"variedad": 'nunique'})
        .rename(columns={"variedad": "n_var_camp_finca"})
    )
    return data.merge(
        n_variedad_finca,
        left_on=['id_finca', 'campaña'],
        right_on=['id_finca', 'campaña'],
        how='left'
    )


def get_shifted_production(wine_data: pd.DataFrame):

    wine_data['prod_shift1'] = wine_data.groupby(['id_finca', 'variedad', 'modo'])[
        'produccion'].shift()
    wine_data['prod_shift2'] = wine_data.groupby(['id_finca', 'variedad', 'modo'])[
        'prod_shift1'].shift()
    wine_data['prod_shift1'] = wine_data['prod_shift1'].fillna(-1)
    wine_data['prod_shift2'] = wine_data['prod_shift2'].fillna(-1)

    return wine_data


def get_production_changes(wine_data: pd.DataFrame):

    wine_data['prod_shift1_gt_shift2'] = [1 if x > y else 0 for x,
                                          y in zip(wine_data.prod_shift1, wine_data.prod_shift2)]

    wine_data['prod_shift_max'] = [x if x > y else y for x,
                                   y in zip(wine_data.prod_shift1, wine_data.prod_shift2)]

    wine_data['prod_shift_change'] = [x-y for x,
                                      y in zip(wine_data.prod_shift1, wine_data.prod_shift2)]

    wine_data['prod_shift_avg'] = [
        (x+y)/2 if y != -1 else x for x, y in zip(wine_data.prod_shift1, wine_data.prod_shift2)]

    return wine_data


def get_production_change_by_estacion(wine_data: pd.DataFrame):
    prod_change_by_estacion = pd.DataFrame(wine_data.groupby(['campaña', 'id_estacion']).agg(
        prod_est_mean_change=('prod_shift1_gt_shift2', 'mean'))).reset_index()

    wine_data = wine_data.merge(prod_change_by_estacion, how='left', on=[
                                'campaña', 'id_estacion'])

    return wine_data


def get_shifted_production_he(wine_data):

    wine_data['prod_he_shift1'] = (
        wine_data
        [['campaña', 'id_finca', 'variedad', 'modo', 'superficie', 'produccion']]
        .assign(prod_he=lambda df: df['produccion']/df['superficie'])
        .groupby(['id_finca', 'variedad', 'modo'])['prod_he'].shift()
    )
    wine_data['prod_he_shift2'] = (
        wine_data
        [['campaña', 'id_finca', 'variedad', 'modo', 'superficie', 'produccion']]
        .assign(prod_he=lambda df: df['produccion']/df['superficie'])
        .groupby(['id_finca', 'variedad', 'modo'])['prod_he'].shift(2)
    )

    wine_data['prod_he_shift1'] = wine_data['prod_he_shift1'].fillna(-1)
    wine_data['prod_he_shift2'] = wine_data['prod_he_shift2'].fillna(-1)

    return wine_data


def get_shifted_production_he_changes(wine_data: pd.DataFrame):

    wine_data['prod_he_shift_change'] = [x-y for x,
                                         y in zip(wine_data.prod_he_shift1, wine_data.prod_he_shift2)]
    wine_data['prod_he_shift_avg'] = [
        (x+y)/2 if y != -1 else x for x, y in zip(wine_data.prod_he_shift1, wine_data.prod_he_shift2)]

    return wine_data


def _historic_prod_he_by_var_and_zone(wine_data, period: int):
    return (
        wine_data[wine_data['campaña'] <= period]
        [['campaña', 'id_zona', 'variedad', 'modo', 'prod_he_shift1']]
        .groupby(['variedad', 'modo', 'id_zona'])
        .agg(
            prod_he_var_zone_mean_hist=('prod_he_shift1', 'mean'),
            prod_he_var_zone_std_hist=('prod_he_shift1', 'std')
        ).fillna(-1).reset_index().assign(campaña=period)
    )


def get_historic_prod_he_by_var_and_zone(wine_data):
    dfs = []
    from_period, to_period = wine_data['campaña'].min(
    )+1, wine_data['campaña'].max()+1
    for i in range(from_period, to_period):
        dfs.append(_historic_prod_he_by_var_and_zone(wine_data, period=i))

    # return dfs
    wine_data = wine_data.merge(
        pd.concat(dfs),
        left_on=['variedad', 'modo', 'id_zona', 'campaña'],
        right_on=['variedad', 'modo', 'id_zona', 'campaña'],
        how='left'
    ).fillna(-1)
    return wine_data


def _historic_prod_he_by_var(wine_data, period: int):
    return (
        wine_data[wine_data['campaña'] <= period]
        [['variedad', 'modo', 'prod_he_shift1']]
        .groupby(['variedad', 'modo'])
        .agg(
            prod_he_var_mean_hist=('prod_he_shift1', 'mean'),
            prod_he_var_std_hist=('prod_he_shift1', 'std')
        ).fillna(-1).reset_index().assign(campaña=period)
    )


def get_historic_prod_he_by_var(wine_data):
    dfs = []
    from_period, to_period = wine_data['campaña'].min(
    )+1, wine_data['campaña'].max()+1
    for i in range(from_period, to_period):
        dfs.append(_historic_prod_he_by_var(wine_data, period=i))

    wine_data = wine_data.merge(
        pd.concat(dfs),
        left_on=['variedad', 'modo', 'campaña'],
        right_on=['variedad', 'modo', 'campaña'],
        how='left'
    ).fillna(-1)
    return wine_data


def get_shifted_production_by_var(wine_data):
    prod_by_finca_var = wine_data.groupby(['campaña', 'id_finca', 'variedad'])[
        'produccion'].sum().reset_index()
    prod_by_finca_var['prod_var_shift_1'] = prod_by_finca_var.groupby(
        ['id_finca', 'variedad'])['produccion'].shift(1).fillna(-1)
    prod_by_finca_var['prod_var_shift_2'] = prod_by_finca_var.groupby(
        ['id_finca', 'variedad'])['produccion'].shift(2).fillna(-1)

    return wine_data.merge(
        prod_by_finca_var[['campaña', 'id_finca', 'variedad',
                           'prod_var_shift_1', 'prod_var_shift_2']],
        left_on=['campaña', 'id_finca', 'variedad'],
        right_on=['campaña', 'id_finca', 'variedad'],
    )


def get_shifted_production_by_finca(wine_data):

    prod_by_finca = wine_data.groupby(['campaña', 'id_finca'])[
        'produccion'].sum().reset_index()
    prod_by_finca['prod_finca_shift_1'] = prod_by_finca.groupby(
        ['id_finca'])['produccion'].shift(1).fillna(-1)
    prod_by_finca['prod_finca_shift_2'] = prod_by_finca.groupby(
        ['id_finca'])['produccion'].shift(2).fillna(-1)

    return wine_data.merge(
        prod_by_finca[['campaña', 'id_finca',
                       'prod_finca_shift_1', 'prod_finca_shift_1']],
        left_on=['campaña', 'id_finca'],
        right_on=['campaña', 'id_finca'],
    )


def get_shifted_production_he_by_var(wine_data):

    prod_he_by_var = wine_data.groupby(['campaña', 'variedad', 'modo']).agg(
        prod_he_var_mean=('prod_he_shift1', 'mean'),
        prod_he_var_std=('prod_he_shift1', 'std')
    ).fillna(-1).reset_index()

    return wine_data.merge(
        prod_he_by_var[['campaña', 'variedad', 'modo',
                        'prod_he_var_mean', 'prod_he_var_std']],
        left_on=['campaña', 'variedad', 'modo'],
        right_on=['campaña', 'variedad', 'modo'],
    )


def get_shifted_production_he_by_zone(wine_data):
    prod_he_by_zone = wine_data.groupby(['campaña', 'variedad', 'modo', 'id_zona']).agg(
        prod_he_zone_mean=('prod_he_shift1', 'mean'),
        prod_he_zone_std=('prod_he_shift1', 'std')
    ).fillna(-1).reset_index()

    return wine_data.merge(
        prod_he_by_zone[['campaña', 'variedad', 'modo',
                         'id_zona', 'prod_he_zone_mean', 'prod_he_zone_std']],
        left_on=['campaña', 'variedad', 'modo', 'id_zona'],
        right_on=['campaña', 'variedad', 'modo', 'id_zona'],
    )


def get_shifted_superficie(wine_data: pd.DataFrame):

    wine_data['sup_shift1'] = wine_data.groupby(['id_finca', 'variedad', 'modo'])[
        'superficie'].shift()
    wine_data['sup_shift2'] = wine_data.groupby(['id_finca', 'variedad', 'modo'])[
        'sup_shift1'].shift()

    wine_data['sup_shift1'] = wine_data['sup_shift1'].fillna(-1)
    wine_data['sup_shift2'] = wine_data['sup_shift2'].fillna(-1)

    return wine_data


def get_total_prod_from_he(wine_data):
    for c in [c for c in wine_data.columns if "_he" in c]:
        wine_data[f"{c}_total"] = wine_data[c]*wine_data['superficie']
    return wine_data


def feateng_wine_data(wine_data, output_path=None):

    wine_data = get_sup_tot_camp_finca(wine_data)

    wine_data = get_sup_tot_finca(wine_data)

    wine_data = get_n_var_finca_camp(wine_data)

    wine_data = get_shifted_production(wine_data)

    wine_data = get_shifted_superficie(wine_data)

    wine_data = get_production_changes(wine_data)

    wine_data = get_production_change_by_estacion(wine_data)

    wine_data = get_shifted_production_he(wine_data)

    wine_data = get_shifted_production_he_changes(wine_data)

    wine_data = get_historic_prod_he_by_var_and_zone(wine_data)

    wine_data = get_historic_prod_he_by_var(wine_data)

    wine_data = get_shifted_production_by_var(wine_data)

    wine_data = get_shifted_production_he_by_zone(wine_data)

    wine_data = get_shifted_production_he_by_var(wine_data)

    wine_data = get_shifted_production_by_finca(wine_data)

    wine_data = get_total_prod_from_he(wine_data)

    # save
    if output_path:
        wine_data.to_csv(output_path, index=False)

        dirname = os.path.dirname(output_path)

        with open(os.path.join(dirname, "wine_features.txt"), "w") as f:
            wine_cols = wine_data.columns.to_list()
            wine_cols.remove('produccion')
            f.write("\n".join(wine_cols))

    return wine_data
