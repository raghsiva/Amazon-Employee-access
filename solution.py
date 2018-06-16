# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 21:19:46 2018

@author: Samsung
"""

import numpy as np
import pandas as pd
from sklearn import tree

train = pd.read_csv(r"G:\Python\Google-data\train.csv")
test = pd.read_csv(r"G:\Python\Google-data\test.csv")

target = train["ACTION"].values
train_features = train[["MGR_ID", "RESOURCE", "ROLE_ROLLUP_1", "ROLE_FAMILY"]].values
sol = tree.DecisionTreeClassifier()
sol = sol.fit(train_features, target)


test_features = test[["MGR_ID", "RESOURCE", "ROLE_ROLLUP_1", "ROLE_FAMILY"]].values
predict = sol.predict(test_features)
Id = np.array(test["id"]).astype(int)
my_solution = pd.DataFrame(predict, Id, columns = ["Action"])
my_solution.to_csv("G:\Python\Google-data\my_solution.csv", index_label=["Id"])