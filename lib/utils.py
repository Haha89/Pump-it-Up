# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pickle import load, dump, HIGHEST_PROTOCOL
from datetime import datetime

PATH_PREPRO = "../data/preprocessing/"


def generate_report(dataframe, name="PumpItUp-EDA"):
    """Generate a report using Pandas profiling"""
    import pandas_profiling as pdp
    profile_train_df = pdp.ProfileReport(dataframe,
                                         title="Pandas Profiling Report",
                                         explorative=True)
    profile_train_df.to_file(output_file=f"../{name}.html")


def preprocessing_data(data, test=False):
    """Preprocesses the input. If test is true, loads existing data"""
    data.drop_duplicates(inplace=True)

    # Drop columns
    data.drop(["recorded_by", "lga",
               "num_private", "region_code", "district_code", "id",
               "ward", "scheme_name", "wpt_name", "extraction_type_class",
               "extraction_type_group", "payment_type", "management",
               "water_quality", "quantity_group", "source", "source_class",
               "waterpoint_type_group", "management_group", "population"],
              axis=1, inplace=True)

    # Replace missing values
    values = {'funder': "Unknown", 'installer': "DWE",
              'public_meeting': True, 'scheme_management': "VWC",
              "permit": True}
    data.fillna(value=values)

    vill = pd.read_csv(PATH_PREPRO + "villages.csv")
    regions = pd.read_csv(PATH_PREPRO + "regions.csv")
    constructions_years = pd.read_csv(PATH_PREPRO + "construction_year.csv")

    data['key'] = data.subvillage + data.region

    data['gps_height'] = np.where((data['latitude'] > -.5) & (data['gps_height'] < 1),
                                  data["key"].map(
                                      vill.set_index("key")['gps_height']),
                                  data['gps_height'])

    data['gps_height'] = np.where(data['gps_height'].isnull(),
                                  data["region"].map(
                                      regions.set_index("region")['gps_height']),
                                  data['gps_height'])

    data['longitude'] = np.where(data['longitude'] < 1,
                                 data["key"].map(
                                     vill.set_index("key")['longitude']),
                                 data['longitude'])

    data['longitude'] = np.where(data['longitude'].isnull(),
                                 data["region"].map(
                                     regions.set_index("region")['longitude']),
                                 data['longitude'])

    data['latitude'] = np.where(data['latitude'] > -.5,
                                data["key"].map(
                                    vill.set_index("key")['latitude']),
                                data['latitude'])

    data['latitude'] = np.where(data['latitude'].isnull(),
                                data["region"].map(
                                    regions.set_index("region")['latitude']),
                                data['latitude'])

    data['construction_year'] = np.where(data['construction_year'] < 1,
                                         data["key"].map(
                                             constructions_years.set_index("key")['construction_year']),
                                         data['construction_year'])

    data.at[data.construction_year.isnull(), "construction_year"] = 1994

    data['funder'] = data.apply(lambda row: funder_wrangler(row), axis=1)
    data['installer'] = data.apply(lambda row: installer_wrangler(row), axis=1)
    data['extraction_type'] = data.apply(lambda row:
                                         extraction_type_wrangler(row), axis=1)

    # One hot encoding
    data = pd.get_dummies(data, columns=["source_type", "scheme_management",
                                         "payment", "extraction_type", "basin",
                                         "waterpoint_type", "quality_group",
                                         "quantity", 'funder', "installer"])

    data["year_recorded"] = pd.DatetimeIndex(data.date_recorded).year
    data["month_recorded"] = pd.DatetimeIndex(data.date_recorded).month
    data['days_since_recorded'] = datetime(2014, 1, 1) - \
        pd.to_datetime(data.date_recorded)
    data['days_since_recorded'] = data['days_since_recorded']\
        .astype('timedelta64[D]').astype(int)

    dic_tf = {True: 1, False: 0}
    data.permit = data.permit.astype(bool).map(dic_tf)
    data.public_meeting = data.public_meeting.astype(bool).map(dic_tf)
    data.amount_tsh = data.amount_tsh.map(lambda x: 1 if x >= 2e4 else 0)

    # Numerical values
    if not test:  # Training, save values
        dic = {}
        for cl in ["gps_height", "longitude", "latitude", "construction_year"]:
            min_col = data[cl].min()
            max_col = data[cl].max()
            dic[cl] = {"min": min_col, "max": max_col}
            data[cl] = (data[cl] - min_col)/(max_col - min_col)

        with open(PATH_PREPRO + 'prepro.pickle', 'wb') as handle:
            dump(dic, handle, protocol=HIGHEST_PROTOCOL)

    else:  # Testing, load values
        with open(PATH_PREPRO + 'prepro.pickle', 'rb') as handle:
            dic = load(handle)

        for cl in ["gps_height", "longitude", "latitude", "construction_year"]:
            min_col = dic[cl]["min"]
            max_col = dic[cl]["max"]
            data[cl] = (data[cl] - min_col)/(max_col - min_col)

    if "status_group" in data.columns:
        dic = {"functional": 2,
               "functional needs repair": 1,
               "non functional": 0}
        data.status_group = data.status_group.map(dic)

    col_drop = ["subvillage", "key", "region", "date_recorded"]
    data.drop(col_drop, axis=1, inplace=True)
    return data


def create_village_region_files(path_inputs):
    """Opens train_values + test_values, find all villages/region and
    calculates average GPS locations"""
    train_val = pd.read_csv(path_inputs + 'train_values.csv')
    test_val = pd.read_csv(path_inputs + 'test_values.csv')
    data = pd.concat([train_val, test_val])

    vill = data[data.longitude > 0][["region", "subvillage",
                                     "longitude", "latitude", "gps_height"]]
    vill = vill.groupby(["region", "subvillage"], as_index=False).mean()
    vill["key"] = vill.subvillage + vill.region
    vill.to_csv(PATH_PREPRO + "villages.csv", index=False)

    regions = vill.groupby(["region"], as_index=False).mean()
    regions.to_csv(PATH_PREPRO + "regions.csv", index=False)

    const_dates = data[data.construction_year > 0][["region", "subvillage",
                                                    "construction_year"]]
    const_dates = const_dates.groupby(["region",
                                       "subvillage"], as_index=False).mean()
    const_dates["key"] = const_dates.subvillage + const_dates.region
    const_dates.to_csv(PATH_PREPRO + "construction_year.csv", index=False)


def funder_wrangler(row):
    '''Keep top 8 values and set the rest to 'other'''
    dic = {'Government Of Tanzania': 'gov', 'Danida': 'danida', "Tasaf": "tas",
           'Rwssp': 'rwssp', 'World Bank': 'world_bank', 'Hesawa': 'hesawa',
           'Kkkt': 'kkkt', 'World Vision': 'world_vision', 'Unicef': 'unicef'}
    return dic[row["funder"]] if row['funder'] in dic.keys() else 'other'


def installer_wrangler(row):
    '''Keep top 8 values and set the rest to 'other'''
    dic = {'DWE': 'dwe', 'Government': 'gov', 'Commu': 'commu', 'TCRS': 'tcrs',
           'DANIDA': 'danida', 'KKKT': 'kkkt', 'RWE': 'rwe', "Gover": "gov",
           'Hesawa': 'hesawa', "Central government": "gov", "DANID": "danida"}
    return dic[row["installer"]] if row['installer'] in dic.keys() else 'other'


def extraction_type_wrangler(row):
    '''Keep top 8 values and set the rest to 'other'''
    dic = {'cemo': "other motorpump", 'climax': "other motorpump",
           "other - mkulima/shinyanga": "other handpump", "swn 80": "swn",
           "other - play pump": "other handpump",
           "walimi": "other handpump", "other - swn 81": "swn",
           "india mark ii": "india mark", "india mark iii": "india mark"}
    if row['extraction_type'] in dic.keys():
        return dic[row["extraction_type"]]
    else:
        return row["extraction_type"]


def lr_decay(current_iter):
    return max(1e-3, 0.1 * np.power(.995, current_iter))
