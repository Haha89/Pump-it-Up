# -*- coding: utf-8 -*-

import pandas as pd
from utils import preprocessing_data, Network
from torch import tensor, cuda, max, load, stack
import torch.utils.data as data_utils


DEVICE = "cuda" if cuda.is_available() else "cpu"
DEVICE = "cpu"
PATH_DATA = "../data/"

dic = {0: "non functional", 1: "functional needs repair", 2: "functional"}

sample_sub = pd.read_csv(PATH_DATA + 'SubmissionFormat.csv')
test_set = pd.read_csv(PATH_DATA + 'test_values.csv')
test_set = preprocessing_data(test_set, test=True)
test_tensor = data_utils.TensorDataset(tensor(test_set.values))
test_loader = data_utils.DataLoader(dataset=test_tensor, batch_size=1)


model = load('../data/model/model.pth').to(DEVICE)
state_dict = load('../data/model/state.pth')
model.load_state_dict(state_dict)

model.eval()
preds = []

for i, values in enumerate(test_loader):
    pred = model(stack(values).float())
    print(pred)
    _, predicted = max(pred.data, 1)
    print(predicted)
    preds.append(dic[predicted])


print(preds)
