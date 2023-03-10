import pandas as pd



def get_mean_features(eto_data, features, name):
    eto_data[name] = eto_data[list(features)].mean(axis=1)
    return eto_data

def get_total_features(eto_data, features, name):
    eto_data[name] = eto_data[list(features)].sum(axis=1)
    return eto_data

def get_std_features(eto_data, features, name):
    eto_data[name] = eto_data[list(features)].std(axis=1)
    return eto_data  


def feateng_eto(eto_data, output_path=None):

    precip_feats = eto_data.filter(like="SumTotalPrecip").columns
    eto_data = get_mean_features(eto_data,precip_feats,name="MeanPrecip")
    eto_data = get_total_features(eto_data,precip_feats,name="TotalPrecip")
    eto_data = get_std_features(eto_data,precip_feats,name="StdlPrecip")

    snow_feats = eto_data.filter(like="SumTotalSnow").columns
    eto_data = get_mean_features(eto_data,snow_feats,name="MeanSnow")
    eto_data = get_total_features(eto_data,snow_feats,name="TotalSnow")

    if output_path:
        eto_data.to_csv(output_path, index=False)

    return eto_data