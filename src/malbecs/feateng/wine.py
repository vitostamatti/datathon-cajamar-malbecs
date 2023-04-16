import pandas as pd
from malbecs.utils import fillna_by_group, fillna_by_value, replace_zeros_with_na
import os


def get_sup_tot_camp_finca(data):
    """
    Calculates the total surface area for each farm and campaign based on a DataFrame containing farm data.

    Args:
        data (pandas.DataFrame): The DataFrame containing farm data.

    Returns:
        pandas.DataFrame: The original DataFrame with an additional column representing the total surface area
        for each farm and campaign, and merged with a separate DataFrame containing the same information.
    """
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
    """
    Calculates the total surface area for each farm based on a DataFrame containing farm data.

    Args:
        data (pandas.DataFrame): The DataFrame containing farm data.

    Returns:
        pandas.DataFrame: The original DataFrame with an additional column representing the mean total surface area
        for each farm, and merged with a separate DataFrame containing the same information.
    """
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
    """
    Calculates the number of unique varieties for each farm and campaign based on a DataFrame containing farm data.

    Args:
        data (pandas.DataFrame): The DataFrame containing farm data.

    Returns:
        pandas.DataFrame: The original DataFrame with an additional column representing the number of unique crop varieties
        for each farm and campaign, and merged with a separate DataFrame containing the same information.
    """
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
    """
    Shifts the production column of the input wine_data DataFrame for each combination of id_finca, variedad, and modo,
    by two periods, and adds two new columns, prod_shift1 and prod_shift2, with the shifted values. If there is no
    value to shift, the corresponding value in prod_shift1 or prod_shift2 is replaced with -1.

    Args:
        wine_data (pd.DataFrame): DataFrame containing the data to shift.

    Returns:
        pd.DataFrame: The input wine_data DataFrame with two new columns, prod_shift1 and prod_shift2.
    """
    wine_data['prod_shift1'] = wine_data.groupby(['id_finca', 'variedad', 'modo'])[
        'produccion'].shift()
    wine_data['prod_shift2'] = wine_data.groupby(['id_finca', 'variedad', 'modo'])[
        'prod_shift1'].shift()
    wine_data['prod_shift1'] = wine_data['prod_shift1'].fillna(-1)
    wine_data['prod_shift2'] = wine_data['prod_shift2'].fillna(-1)

    return wine_data


def get_production_changes(wine_data: pd.DataFrame):
    """
    Calculates several production changes based on shifted values in the provided wine data DataFrame.

    Args:
        wine_data (pd.DataFrame): The wine data DataFrame.

    Returns:
        pd.DataFrame: The updated wine data DataFrame, with the following new columns:
            - prod_shift1_gt_shift2: A binary column (1 or 0) indicating whether the value in the prod_shift1 column is
              greater than the value in the prod_shift2 column.
            - prod_shift_max: A column indicating the maximum value between prod_shift1 and prod_shift2.
            - prod_shift_change: A column indicating the difference between prod_shift1 and prod_shift2.
            - prod_shift_avg: A column indicating the average between prod_shift1 and prod_shift2, or prod_shift1 if
              prod_shift2 is missing (-1).
    """

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
    """
    Calculates the mean production change by season for each station.

    Args:
        wine_data (pd.DataFrame): A pandas DataFrame containing wine production data.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the mean production change by season
                      for each station.
    """

    prod_change_by_estacion = pd.DataFrame(wine_data.groupby(['campaña', 'id_estacion']).agg(
        prod_est_mean_change=('prod_shift1_gt_shift2', 'mean'))).reset_index()

    wine_data = wine_data.merge(prod_change_by_estacion, how='left', on=[
                                'campaña', 'id_estacion'])

    return wine_data


def get_shifted_production_he(wine_data):
    """
    Calculates the production per hectare shift columns for the input wine_data DataFrame.

    Args:
        wine_data (pd.DataFrame): The input DataFrame containing the wine production data.

    Returns:
        pd.DataFrame: The input wine_data DataFrame with the production per hectare shift columns
                      added (prod_he_shift1 and prod_he_shift2).
    """

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
    """
    Calculates the changes in the hectare yield of the wine data by comparing 
    the hectare yields of the current and previous two years. Adds two columns 
    to the input DataFrame: 'prod_he_shift_change' with the difference in the 
    hectare yields, and 'prod_he_shift_avg' with the average hectare yield between 
    the current and previous year.

    Args:
        wine_data (pd.DataFrame): DataFrame with the following columns: 
            'campaña', 'id_finca', 'variedad', 'modo', 'superficie', and 'produccion'. 
            'superficie' and 'produccion' are used to calculate the hectare yield.

    Returns:
        pd.DataFrame: The input DataFrame with two additional columns: 
            'prod_he_shift_change' and 'prod_he_shift_avg'.
    """
    wine_data['prod_he_shift_change'] = [x-y for x,
                                         y in zip(wine_data.prod_he_shift1, wine_data.prod_he_shift2)]
    wine_data['prod_he_shift_avg'] = [
        (x+y)/2 if y != -1 else x for x, y in zip(wine_data.prod_he_shift1, wine_data.prod_he_shift2)]

    return wine_data


def _historic_prod_he_by_var_modo_zona(wine_data, period: int):
    """
    Calculates the historic average and standard deviation of the production per hectare (prod_he) by variety, mode, 
    and zone up to a certain period.

    Args:
        wine_data (pd.DataFrame): The data frame containing the data.
        period (int): The period up to which the historic average and standard deviation will be calculated.

    Returns:
        pd.DataFrame: A data frame with the historic average and standard deviation of prod_he by variety, mode, and zone.
    """
    return (
        wine_data[wine_data['campaña'] <= period]
        [['campaña', 'id_zona', 'variedad', 'modo', 'prod_he_shift1']]
        .groupby(['variedad', 'modo', 'id_zona'])
        .agg(
            prod_he_var_zone_mean_hist=('prod_he_shift1', 'mean'),
            prod_he_var_zone_std_hist=('prod_he_shift1', 'std')
        ).fillna(-1).reset_index().assign(campaña=period)
    )


def get_historic_prod_he_by_var_modo_modo_zona(wine_data):
    """
    Returns a DataFrame with the historic mean production of `prod_he` grouped by
    `variedad`, `modo`, `modo_zona`, and `campaña` for each year between the minimum and maximum
    years in `wine_data`.

    Parameters:
        wine_data (pd.DataFrame): The input DataFrame containing wine production data.

    Returns:
        pd.DataFrame: A DataFrame with the historic mean production of `prod_he` grouped by
            `variedad`, `modo`, `modo_zona`, and `campaña` for each year between the minimum and maximum
            years in `wine_data`.

    """
    dfs = []
    from_period, to_period = wine_data['campaña'].min(
    )+1, wine_data['campaña'].max()+1
    for i in range(from_period, to_period):
        dfs.append(_historic_prod_he_by_var_modo_zona(wine_data, period=i))

    # return dfs
    wine_data = wine_data.merge(
        pd.concat(dfs),
        left_on=['variedad', 'modo', 'id_zona', 'campaña'],
        right_on=['variedad', 'modo', 'id_zona', 'campaña'],
        how='left'
    ).fillna(-1)
    return wine_data


def _historic_prod_he_by_var_modo(wine_data, period: int):
    """
    Compute the historic mean and standard deviation of prod_he_shift1 by variety and mode up to a given period.

    Args:
        wine_data: DataFrame containing the wine data
        period: The period up to which the historic mean and standard deviation of prod_he_shift1 will be computed

    Returns:
        DataFrame containing the historic mean and standard deviation of prod_he_shift1 by variety and mode up to the given period
    """
    return (
        wine_data[wine_data['campaña'] <= period]
        [['variedad', 'modo', 'prod_he_shift1']]
        .groupby(['variedad', 'modo'])
        .agg(
            prod_he_var_mean_hist=('prod_he_shift1', 'mean'),
            prod_he_var_std_hist=('prod_he_shift1', 'std')
        ).fillna(-1).reset_index().assign(campaña=period)
    )


def get_historic_prod_he_by_var_modo(wine_data):
    """
    Compute the historic mean and standard deviation of prod_he_shift1 by variety and mode for all periods in the dataset.

    Args:
        wine_data: DataFrame containing the wine data

    Returns:
        DataFrame containing the historic mean and standard deviation of prod_he_shift1 by variety and mode for all periods in the dataset
    """
    dfs = []
    from_period, to_period = wine_data['campaña'].min(
    )+1, wine_data['campaña'].max()+1
    for i in range(from_period, to_period):
        dfs.append(_historic_prod_he_by_var_modo(wine_data, period=i))

    wine_data = wine_data.merge(
        pd.concat(dfs),
        left_on=['variedad', 'modo', 'campaña'],
        right_on=['variedad', 'modo', 'campaña'],
        how='left'
    ).fillna(-1)

    return wine_data


def get_shifted_production_by_var(wine_data):
    """
    Given a pandas DataFrame with wine production data, returns a new DataFrame with additional columns for
    shifted production values.

    Args:
        wine_data (pandas DataFrame): The input DataFrame with columns 'campaña', 'id_finca', 'variedad', and 'produccion'.

    Returns:
        pandas DataFrame: A new DataFrame with the same columns as the input DataFrame plus two additional columns,
            'prod_var_shift_1' and 'prod_var_shift_2', which contain the production values for each finca and variety
            shifted by 1 and 2 rows, respectively. If there is no data available for a given finca and variety combination
            for the shifted rows, the corresponding value will be -1.
    """
    prod_by_finca_var = wine_data.groupby(['campaña', 'id_finca', 'variedad'])[
        'produccion'].sum().reset_index()
    
    prod_by_finca_var['prod_var_shift_1'] = prod_by_finca_var.groupby(
        ['id_finca', 'variedad'])['produccion'].shift(1).fillna(-1)
    
    prod_by_finca_var['prod_var_shift_2'] = prod_by_finca_var.groupby(
        ['id_finca', 'variedad'])['produccion'].shift(2).fillna(-1)

    return wine_data.merge(
        prod_by_finca_var[['campaña', 'id_finca', 'variedad','prod_var_shift_1', 'prod_var_shift_2']],
        left_on=['campaña', 'id_finca', 'variedad'],
        right_on=['campaña', 'id_finca', 'variedad'],
    )


def get_shifted_production_by_finca(wine_data):
    """
    Given a pandas DataFrame with wine production data, returns a new DataFrame with additional columns for
    shifted production values.

    Args:
        wine_data (pandas DataFrame): The input DataFrame with columns 'campaña', 'id_finca', and 'produccion'.

    Returns:
        pandas DataFrame: A new DataFrame with the same columns as the input DataFrame plus two additional columns,
            'prod_finca_shift_1' and 'prod_finca_shift_2', which contain the production values for each 'id_finca'
            shifted by 1 and 2 rows, respectively. If there is no data available for a given 'id_finca' for the shifted
            rows, the corresponding value will be -1.
    """
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
    """
    Compute the mean and standard deviation of prod_he_shift1 and the difference between the means of prod_he_shift1 and prod_he_shift2
    by variety and campaign.

    Args:
        wine_data: DataFrame containing the wine data

    Returns:
        DataFrame containing the mean and standard deviation of prod_he_shift1 and the difference between the means of prod_he_shift1 and prod_he_shift2
            by variety and campaign
    """
    prod_he_by_var = wine_data.groupby(['campaña', 'variedad']).agg(
        prod_he_var_mean_shift1=('prod_he_shift1', 'mean'),
        prod_he_var_mean_shift2=('prod_he_shift2', 'mean'),
        prod_he_var_std_shift1=('prod_he_shift1', 'std')
    ).fillna(-1).reset_index()

    prod_he_by_var['prod_he_var_change'] = [
        x-y for x,y in 
        zip(prod_he_by_var.prod_he_var_mean_shift1, prod_he_by_var.prod_he_var_mean_shift2)
        ]

    return wine_data.merge(
        prod_he_by_var[[
            'campaña', 'variedad', 
            'prod_he_var_mean_shift1', 
            'prod_he_var_std_shift1', 
            'prod_he_var_change'
        ]],
        left_on=['campaña', 'variedad'],
        right_on=['campaña', 'variedad'],
    )


def get_shifted_production_he_by_zone(wine_data):
    """
    Compute the mean and standard deviation of prod_he_shift1 and the difference between the means of prod_he_shift1 and prod_he_shift2
    by zone and campaign.

    Args:
        wine_data: DataFrame containing the wine data

    Returns:
        DataFrame containing the mean and standard deviation of prod_he_shift1 and the difference between the means of prod_he_shift1 and prod_he_shift2
            by zone and campaign
    """
    prod_he_by_var = wine_data.groupby(['campaña', 'id_zona']).agg(
        prod_he_zona_mean_shift1=('prod_he_shift1', 'mean'),
        prod_he_zona_mean_shift2=('prod_he_shift2', 'mean'),
        prod_he_zona_std_shift1=('prod_he_shift1', 'std')
    ).fillna(-1).reset_index()

    prod_he_by_var['prod_he_zona_change'] = [
        x-y for x,y in 
        zip(prod_he_by_var.prod_he_zona_mean_shift1, prod_he_by_var.prod_he_zona_mean_shift2)
        ]

    return wine_data.merge(
        prod_he_by_var[[
            'campaña', 'id_zona','prod_he_zona_mean_shift1', 
            'prod_he_zona_std_shift1','prod_he_zona_change'
        ]],
        left_on=['campaña', 'id_zona'],
        right_on=['campaña', 'id_zona'],
    )


def get_shifted_production_he_by_var_modo(wine_data):
    """
    Calculate shifted production HE by variety and mode and return the updated wine_data 
    DataFrame with new columns.

    Args:
        wine_data: a Pandas DataFrame with columns 'campaña', 'variedad', 'modo', 'prod_he_shift1', and 'prod_he_shift2'

    Returns:
        A Pandas DataFrame with the same columns as wine_data plus additional columns 'prod_he_var_modo_mean_shift1',
            'prod_he_var_modo_std_shift1', and 'prod_he_var_modo_change'. The 'prod_he_var_modo_mean_shift1' column represents 
            the mean production HE for each variety and mode in the previous year (i.e., 'prod_he_shift1' column).
            The 'prod_he_var_modo_std_shift1' column represents the standard deviation of the production HE for each variety 
            and mode in the previous year. The 'prod_he_var_modo_change' column represents the difference in mean 
            production HE between the current year and the previous year for each variety and mode.
    """
    prod_he_by_var_modo = wine_data.groupby(['campaña', 'variedad', 'modo']).agg(
        prod_he_var_modo_mean_shift1=('prod_he_shift1', 'mean'),
        prod_he_var_modo_mean_shift2=('prod_he_shift2', 'mean'),
        prod_he_var_modo_std_shift1=('prod_he_shift1', 'std')
    ).fillna(-1).reset_index()

    prod_he_by_var_modo['prod_he_var_modo_change'] = [
        x-y for x,y in 
        zip(prod_he_by_var_modo.prod_he_var_modo_mean_shift1, prod_he_by_var_modo.prod_he_var_modo_mean_shift2)
        ]

    return wine_data.merge(
        prod_he_by_var_modo[[
            'campaña', 'variedad', 'modo',
            'prod_he_var_modo_mean_shift1', 
            'prod_he_var_modo_std_shift1', 
            'prod_he_var_modo_change'
        ]],
        left_on=['campaña', 'variedad', 'modo'],
        right_on=['campaña', 'variedad', 'modo'],
    )


def get_shifted_production_he_by_var_modo_zona(wine_data):
    """
    Calculates the mean and standard deviation of the prod_he_shift1 column grouped by the campaign,
    variety, mode and zone, calculates the difference between the means of prod_he_shift1 and prod_he_shift2, 
    and merges the results into the original wine_data DataFrame.
    
    Args:
        wine_data (pandas.DataFrame): The input DataFrame containing wine production data.
    
    Returns:
        pandas.DataFrame: The merged DataFrame containing the mean and standard deviation of the prod_he_shift1 
            column, the difference between the means of prod_he_shift1 and prod_he_shift2, grouped by the campaign,
            variety, mode and zone, and the original wine production data.
    """
    prod_he_by_zone = wine_data.groupby(['campaña', 'variedad', 'modo', 'id_zona']).agg(
        prod_he_var_modo_zona_mean_shift1=('prod_he_shift1', 'mean'),
        prod_he_var_modo_zona_std_shift1=('prod_he_shift1', 'std'),
        prod_he_var_modo_zona_mean_shift2=('prod_he_shift2', 'mean'),
    ).fillna(-1).reset_index()

    prod_he_by_zone['prod_he_var_modo_zona_change'] = [
        x-y for x,y in 
        zip(prod_he_by_zone.prod_he_var_modo_zona_mean_shift1, prod_he_by_zone.prod_he_var_modo_zona_mean_shift2)
        ]

    return wine_data.merge(
        prod_he_by_zone[[
            'campaña', 'variedad', 'modo', 'id_zona', 
            'prod_he_var_modo_zona_mean_shift1', 
            'prod_he_var_modo_zona_std_shift1', 
            'prod_he_var_modo_zona_change'
        ]],
        left_on=['campaña', 'variedad', 'modo', 'id_zona'],
        right_on=['campaña', 'variedad', 'modo', 'id_zona'],
    )


def get_shifted_superficie(wine_data: pd.DataFrame):
    """
    Returns a pandas DataFrame with two new columns indicating the shifted values of the "superficie" column of the input
    DataFrame. The shifting is performed based on the values of the "id_finca", "variedad", and "modo" columns.

    Args:
        wine_data (pd.DataFrame): Input pandas DataFrame with columns "id_finca", "variedad", "modo", and "superficie".

    Returns:
        pd.DataFrame: Output pandas DataFrame with two new columns "sup_shift1" and "sup_shift2" indicating the shifted
            values of the "superficie" column. The NaN values in the shifted columns are filled with -1.
"""
    wine_data['sup_shift1'] = wine_data.groupby(['id_finca', 'variedad', 'modo'])[
        'superficie'].shift()
    wine_data['sup_shift2'] = wine_data.groupby(['id_finca', 'variedad', 'modo'])[
        'sup_shift1'].shift()

    wine_data['sup_shift1'] = wine_data['sup_shift1'].fillna(-1)
    wine_data['sup_shift2'] = wine_data['sup_shift2'].fillna(-1)

    return wine_data


def get_total_prod_from_he(wine_data):
    """
    Returns a pandas DataFrame with new columns indicating the total production of each column ending with "_he",
    calculated by multiplying it by the "superficie" column.

    Args:
        wine_data (pd.DataFrame): Input pandas DataFrame with columns ending with "_he" and "superficie".

    Returns:
        pd.DataFrame: Output pandas DataFrame with new columns ending with "_he_total" indicating the total production
            of each column ending with "_he" after multiplying it by the "superficie" column.
    """
    for c in [c for c in wine_data.columns if "_he" in c]:
        wine_data[f"{c}_total"] = wine_data[c]*wine_data['superficie']
    return wine_data


def feateng_wine_data(wine_data, output_path=None):
    """
    Applies a series of feature engineering transformations to the wine_data dataframe and returns the resulting
    transformed dataframe.

    Args:
        wine_data: A pandas dataframe containing wine production data.
        output_path: Optional string indicating the path where the transformed dataframe should be saved.

    Returns:
        A pandas dataframe with the transformed wine production data.
    """

    wine_data = get_sup_tot_camp_finca(wine_data)

    wine_data = get_sup_tot_finca(wine_data)

    wine_data = get_n_var_finca_camp(wine_data)

    wine_data = get_shifted_production(wine_data)

    wine_data = get_shifted_superficie(wine_data)

    wine_data = get_production_changes(wine_data)

    wine_data = get_production_change_by_estacion(wine_data)

    wine_data = get_shifted_production_he(wine_data)

    wine_data = get_shifted_production_he_changes(wine_data)

    wine_data = get_historic_prod_he_by_var_modo_modo_zona(wine_data)

    wine_data = get_historic_prod_he_by_var_modo(wine_data)

    wine_data = get_shifted_production_by_var(wine_data)

    wine_data = get_shifted_production_he_by_var_modo_zona(wine_data)

    wine_data = get_shifted_production_he_by_var_modo(wine_data)

    wine_data = get_shifted_production_he_by_var(wine_data)

    wine_data = get_shifted_production_he_by_zone(wine_data)

    wine_data = get_shifted_production_by_finca(wine_data)

    wine_data = get_total_prod_from_he(wine_data)

    # save
    if output_path:
        wine_data.to_csv(output_path, index=False)

        dirname = os.path.dirname(output_path)

        with open(os.path.join(dirname, "wine_features.txt"), "w", encoding="utf-8") as f:
            wine_cols = wine_data.columns.to_list()
            wine_cols.remove('produccion')
            f.write("\n".join(wine_cols))


    return wine_data
