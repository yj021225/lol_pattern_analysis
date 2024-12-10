import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

df = pd.read_json("./final_target_data.json")

# 데이터 로드
combat_at14 = ['at14killsRatio', 'at14deathsRatio', 'at14assistsRatio',
               'at14solokillsRatio', 'at14solodeathsRatio', 'at14dpm', 'at14dtpm']

combat_af14 = ['af14killsRatio', 'af14deathsRatio', 'af14assistsRatio',
               'af14solokillsRatio', 'af14solodeathsRatio', 'af14dpm', 'af14dtpm']

x = df[combat_at14 + combat_af14]
y = df['targetWin'].astype(int)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Random Forest로 Feature 중요도 계산
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)
feature_importances = rf_model.feature_importances_

# 중요도가 높은 Feature 선택
important_features = x.columns[feature_importances.argsort()[-5:]]  # 상위 5개
print("중요 Feature:", important_features)

# 주요 Feature로 다시 학습
x_train_selected = x_train[important_features]
x_test_selected = x_test[important_features]

rf_model.fit(x_train_selected, y_train)
y_pred = rf_model.predict(x_test_selected)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auroc = roc_auc_score(y_test, rf_model.predict_proba(x_test_selected)[:, 1])

print(f"정확도: {accuracy:.3f}, F1-Score: {f1:.3f}, AUROC: {auroc:.3f}")
