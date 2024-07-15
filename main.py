import numpy as np
import tensorflow as tf
import pandas as pd


data = pd.DataFrame({
    'AlliesCount': [16],
    'EnemyCount': [15],
    'AlliesCoords': ['[(1,1), (2,3), (3,5), (4,7), (5,9), (6,11), (7,13), (8,15), (9,17), (10,19), (11,21), (12,23), (13,25), (14,27), (15,29), (16,31)]'],
    'EnemyCoords': ['[(5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12), (13,13), (14,14), (15,15), (16,16), (17,17), (18,18), (19,19)]'],
    'AlliesHealth': ['[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]'],
    'EnemyHealth': ['[100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]'],
    'Resources': ['[50, 5]'],
    'EnvironmentType': ['Plain'],
    'Weather': ['Rain'],
    'Time': [90]
})

def convert_list_column(column):
    return column.apply(lambda x: np.mean(eval(x)) if isinstance(x, str) else x)

data['AlliesCoords'] = convert_list_column(data['AlliesCoords'])
data['EnemyCoords'] = convert_list_column(data['EnemyCoords'])
data['AlliesHealth'] = convert_list_column(data['AlliesHealth'])
data['EnemyHealth'] = convert_list_column(data['EnemyHealth'])
data['Resources'] = convert_list_column(data['Resources'])

data['EnvironmentType'] = data['EnvironmentType'].astype('category').cat.codes
data['Weather'] = data['Weather'].astype('category').cat.codes

data = data.astype(float)
model = tf.keras.models.load_model('battlefield_analyzeRU1.keras')
prediction = model.predict(data)

decision = 'Attack' if prediction >= 0.5 else 'Retreat'
print("Решение:", decision)