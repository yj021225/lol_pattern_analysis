import pandas as pd
import numpy as np
import json

df = pd.read_json("./final_target_data.json")

combat_at14 = ['at14killsRatio', 'at14deathsRatio', 'at14assistsRatio',
'at14solokillsRatio', 'at14solodeathsRatio',
'at14dpm', 'at14dtpm']

combat_af14 = ['af14killsRatio', 'af14deathsRatio', 'af14assistsRatio',
'af14solokillsRatio', 'af14solodeathsRatio',
'af14dpm', 'af14dtpm']

x = df[combat_at14]
y = df['targetWin'].astype(int)

# print(x)
# print(y)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score

model = LinearRegression()
model.fit(x, y)

print("모델 계수 (theta) : ", model.coef_)
print("모델 절편 (b) : ", model.intercept_)

y_pred = model.predict(x)

threshold = 0.5
y_pred_class = (y_pred >= threshold).astype(int)

accuracy = accuracy_score(y, y_pred_class)
auroc = roc_auc_score(y, y_pred)

print(f"정확도 (Accuracy, threshold: {threshold}): {accuracy}")
print(f"AUROC: {auroc}")