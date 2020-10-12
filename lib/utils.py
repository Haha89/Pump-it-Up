# -*- coding: utf-8 -*-
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pickle import load, dump, HIGHEST_PROTOCOL


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
               "num_private", "subvillage", "region_code", "district_code",
               "ward", "scheme_name", "wpt_name", "lga", "extraction_type",
               "extraction_type_group", "payment_type", "management",
               "water_quality", "quantity_group", "source", "source_class",
               "waterpoint_type",  "population",
               "public_meeting", "management_group", "permit"],
              axis=1, inplace=True)

    # Replace missing values
    values = {'funder': "Unknown", 'installer': "DWE",
              'public_meeting': True, 'scheme_management': "VWC",
              "construction_year": 1980} #, "permit": True
    data.fillna(value=values)
    
    # data.permit = data.permit.astype(bool) 
    
    data.at[data.construction_year == 0, "construction_year"] = 1994

    # One hot encoding
    data = pd.get_dummies(data, columns=["source_type", "scheme_management",
                                         "payment", "extraction_type_class",
                                         "basin", "waterpoint_type_group",
                                         "quality_group", "quantity", "region"])
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
    def __init__(self, input_dim):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_dim, 40)
        self.l2 = nn.Linear(40, 40)
        self.l4 = nn.Linear(40, 3)
        self.dropout1 = nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.l4(x)
        return x
