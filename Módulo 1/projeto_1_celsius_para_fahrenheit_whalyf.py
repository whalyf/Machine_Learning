# Etapa 1: Importação das bibliotecas
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
tf.__version__

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


"""# Etapa 2: Importação da base de dados"""
temperature_df = pd.read_csv('content\Celsius-to-Fahrenheit.csv')

temperature_df.reset_index(drop = True, inplace = True)

temperature_df

temperature_df.head()

temperature_df.tail(10)

temperature_df.info()

temperature_df.describe()

"""# Etapa 3: Visualização da base de dados"""

sns.scatterplot(temperature_df['Celsius'], temperature_df['Fahrenheit']);

"""# Etapa 4: Configuração da base de dados de treinamento"""

X_train = temperature_df['Celsius']
y_train = temperature_df['Fahrenheit']

X_train.shape

y_train.shape

"""# Etapa 5: Construção e treinamento do modelo"""

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 1, input_shape = [1]))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss = 'mean_squared_error')

epochs_hist = model.fit(X_train, y_train, epochs = 500)

"""# Etapa 5: Avaliação do modelo"""

epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.legend(['Training loss']);

model.get_weights()

temp_c = 10
temp_f = model.predict([temp_c])
temp_f

temp_f1 = 9/5 * temp_c + 32
temp_f1
