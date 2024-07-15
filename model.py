import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import os

data = pd.read_csv('battlefield_data.csv', delimiter=';')
print(data.columns)


def convert_list_column(column):
    return column.apply(
        lambda x:  # def anon(x):
            np.mean(eval(x)) if isinstance(x, str) else x
    )


data['AlliesCoords'] = convert_list_column(data['AlliesCoords'])
data['EnemyCoords'] = convert_list_column(data['EnemyCoords'])
data['AlliesHealth'] = convert_list_column(data['AlliesHealth'])
data['EnemyHealth'] = convert_list_column(data['EnemyHealth'])
data['Resources'] = convert_list_column(data['Resources'])

# Преобразование категориальных данных в числовые
data['EnvironmentType'] = data['EnvironmentType'].astype('category').cat.codes
data['Weather'] = data['Weather'].astype('category').cat.codes

X = data.drop('Decision', axis=1).astype(float)
y = data['Decision'].map({'Attack': 1, 'Retreat': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1000, batch_size=1, validation_split=0.1)

model.evaluate(X_test, y_test)

model.save('battlefield_analyzeRU1.keras')
