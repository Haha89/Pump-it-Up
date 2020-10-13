# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pickle import load, dump, HIGHEST_PROTOCOL

PATH_PREPRO = "../data/preprocessing/"


def generate_report(dataframe, name="PumpItUp-EDA"):
    import pandas_profiling as pdp
    profile_train_df = pdp.ProfileReport(dataframe,
                                         title="Pandas Profiling Report",
                                         explorative=True)
    profile_train_df.to_file(output_file=f"../{name}.html")


def preprocessing_data(data, test=False):
    """input"""
    data.drop_duplicates(inplace=True)

    # Drop columns
    data.drop(["amount_tsh", "recorded_by", "funder", "installer",
               "num_private", "region_code", "district_code",
               "ward", "scheme_name", "wpt_name", "lga", "extraction_type",
               "extraction_type_group", "payment_type", "management",
               "water_quality", "quantity_group", "source", "source_class",
               "waterpoint_type",  "population",
               "public_meeting", "management_group", "id"],
              axis=1, inplace=True)

    # Replace missing values
    values = {'funder': "Unknown", 'installer': "DWE",
              'public_meeting': True, 'scheme_management': "VWC",
              "construction_year": 1994, "permit": True}
    data.fillna(value=values)

    data.permit = data.permit.astype(bool)
    data.at[data.construction_year == 0, "construction_year"] = 1994

    if not test:
        # Longitude
        vill = data[data.longitude > 0][["region", "subvillage",
                                         "longitude", "latitude",
                                         "gps_height"]]
        vill = vill.groupby(["region", "subvillage"], as_index=False).mean()
        vill["key"] = vill.subvillage + vill.region
        vill.to_csv(PATH_PREPRO + "villages.csv", index=False)
        regions = vill.groupby(["region"], as_index=False).mean()
        regions.to_csv(PATH_PREPRO + "regions.csv", index=False)
    else:
        vill.read_csv(PATH_PREPRO + "villages.csv")
        regions.read_csv(PATH_PREPRO + "regions.csv")

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

    data.drop(["subvillage", "key"], axis=1, inplace=True)

    data.to_csv("../postpro.csv")
    # One hot encoding
    data = pd.get_dummies(data, columns=["source_type", "scheme_management",
                                         "payment", "extraction_type_class",
                                         "waterpoint_type_group", "quantity",
                                         "quality_group", "basin", "region",
                                         "permit"])
    data.date_recorded = pd.DatetimeIndex(data.date_recorded).month

    # Numerical values
    if not test:  # Training, save values
        dic = {}
        for cl in ["gps_height", "longitude", "latitude", "construction_year"]:
            min_col = data[cl].min()
            max_col = data[cl].max()
            dic[cl] = {"min": min_col, "max": max_col}
            data[cl] = (data[cl] - min_col)/(max_col - min_col)

        with open('./prepro.pickle', 'wb') as handle:
            dump(dic, handle, protocol=HIGHEST_PROTOCOL)

    else:  # Testing, load values
        with open('./prepro.pickle', 'rb') as handle:
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

    return data


# Creation du reseau
class Network(nn.Module):
    def __init__(self, input_dim, size):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_dim, size)
        self.l2 = nn.Linear(size, size)
        self.l3 = nn.Linear(size, size)
        self.l4 = nn.Linear(size, size)
        self.lout = nn.Linear(size, 3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.45)
        self.sigmoid = nn.Sigmoid()

        # init to normal
        nn.init.normal_(self.l1.weight, mean=0, std=1.0)
        nn.init.normal_(self.l2.weight, mean=0, std=1.0)
        nn.init.normal_(self.l3.weight, mean=0, std=1.0)
        nn.init.normal_(self.l4.weight, mean=0, std=1.0)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.sigmoid(x)
        x = self.dropout1(x)
        x = self.l3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.l4(x)
        x = self.sigmoid(x)
        x = self.lout(x)
        return x
