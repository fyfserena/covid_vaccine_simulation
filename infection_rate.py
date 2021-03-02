# -*- coding: utf-8 -*-
"""Infection rate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RHYfsoDHDBJXvv6Kq2xxredfESZ64TDs
"""

# from google.colab import files
# uploaded = files.upload()
# import io
# data1 = io.BytesIO(uploaded['SimualtionRo.csv'])

import numpy as np
import pandas as pd

data1 = 'SimualtionRo.csv'
df = pd.read_csv(data1)

df = df[:-1]

df1 = df.groupby(by=["Rate"]).sum()

df1[["Percentage","Po*Io",]]

occupation_by_risk = [list(df[df["Rate"]==i]["JobSector"]) for i in range(5)]
