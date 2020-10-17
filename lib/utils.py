# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pickle import load, dump, HIGHEST_PROTOCOL

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
    data.drop(["recorded_by", "installer", "lga",
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
    constructions_years_regions = pd.read_csv(PATH_PREPRO + "construction_year_regions.csv")

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
    # One hot encoding
    data = pd.get_dummies(data, columns=["source_type", "scheme_management",
                                         "payment", "extraction_type",
                                         "waterpoint_type", "quantity",
                                         "quality_group", "basin", 'funder'])

    data["year_recorded"] = pd.DatetimeIndex(data.date_recorded).year
    data.date_recorded = pd.DatetimeIndex(data.date_recorded).month
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
    data.drop(["subvillage", "key", "region"], axis=1, inplace=True)

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
    regions = const_dates.groupby(["region"], as_index=False).mean()
    regions.to_csv(PATH_PREPRO + "construction_year_regions.csv", index=False)


def funder_wrangler(row):
    '''Keep top 8 values and set the rest to 'other'''
    if row['funder'] == 'Government Of Tanzania':
        return 'gov'
    elif row['funder'] == 'Danida':
        return 'danida'
    elif row['funder'] == 'Hesawa':
        return 'hesawa'
    elif row['funder'] == 'Rwssp':
        return 'rwssp'
    elif row['funder'] == 'World Bank':
        return 'world_bank'
    elif row['funder'] == 'Kkkt':
        return 'kkkt'
    elif row['funder'] == 'World Vision':
        return 'world_vision'
    elif row['funder'] == 'Unicef':
        return 'unicef'
    else:
        return 'other'


# Creation du reseau
class Network(nn.Module):
    def __init__(self, input_dim, size):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_dim, size)
        self.l2 = nn.Linear(size, size)
        self.l3 = nn.Linear(size, size)
        self.l4 = nn.Linear(size, size)
        self.l5 = nn.Linear(size, size)
        self.lout = nn.Linear(size, 3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.45)
        self.bn1 = nn.BatchNorm1d(num_features=size)
        self.bn2 = nn.BatchNorm1d(num_features=size)
        self.bn3 = nn.BatchNorm1d(num_features=size)
        self.bn4 = nn.BatchNorm1d(num_features=size)
        self.bn5 = nn.BatchNorm1d(num_features=size)
        self.sigmoid = nn.Sigmoid()

        # init to normal
        nn.init.normal_(self.l1.weight, mean=0, std=1.0)
        nn.init.normal_(self.l2.weight, mean=0, std=1.0)
        nn.init.normal_(self.l3.weight, mean=0, std=1.0)
        nn.init.normal_(self.l4.weight, mean=0, std=1.0)
        nn.init.normal_(self.l5.weight, mean=0, std=1.0)
        nn.init.normal_(self.lout.weight, mean=0, std=1.0)

    def forward(self, x):
        x = self.bn1(F.relu(self.l1(x)))
        x = self.bn2(F.relu(self.l2(x)))
        x = self.dropout1(x)
        x = self.bn3(F.relu(self.l3(x)))
        x = self.dropout2(x)
        x = self.bn4(F.relu(self.l4(x)))
        x = self.bn5(F.relu(self.l5(x)))
        x = self.dropout2(x)
        x = self.lout(x)
        return x
