import pandas as pd
import numpy as np

data_path = './final_target_data.json'
df = pd.read_json(data_path)

combat_af14 = [
    'af14killsRatio', 'af14deathsRatio', 'af14assistsRatio',
    'af14solokillsRatio', 'af14solodeathsRatio', 'af14dpm', 'af14dtpm'
]

def calculate_combat(row):
    weights = {
        'af14killsRatio': 1.5,
        'af14deathsRatio': -1.0,
        'af14assistsRatio': 1.0,
        'af14solokillsRatio': 1.2,
        'af14solodeathsRatio': -0.8,
        'af14dpm': 0.01,
        'af14dtpm': -0.005
    }
    return sum(row[feature] * weight for feature, weight in weights.items())

df['combat'] = df.apply(calculate_combat, axis=1)

output_path = './final_target_data_combat.json'
df.to_json(output_path, orient='records', force_ascii=False, indent=4)

print(f"Updated data with 'combat' metric saved to {output_path}")
