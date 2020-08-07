# Etapa 1: Importação das bibliotecas
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
tf.__version__

"""# Etapa 2: Importação da base de dados"""

# Leitura do arquivo csv

vendas_df = pd.read_csv('/content/original.csv')

# Visualização de todos os registros
vendas_df

# Visualização dos 5 primeiros registros
vendas_df.head(5)

# Visualização dos 10 últimos registros
vendas_df.tail(10)

# Visualização de informações da base de dados
vendas_df.info()

# Descrição da base de dados
vendas_df.describe()

"""# Etapa 3: Visualização da base de dados"""

# Scatter plot do Seborn
sns.scatterplot(vendas_df['Temperature'], vendas_df['Revenue']);

"""# Etapa 4: Criação das variáveis da base de dados"""

# Criação das variáveis X_train e y_train
X_train = vendas_df['Temperature']
y_train = vendas_df['Revenue']

# Formato da variável X_train
X_train.shape

# Formato da variável y_train
y_train.shape

"""# Etapa 5: Criação e construção do modelo"""

# Construção do modelo sequencial
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 10, input_shape = [1]))
model.add(tf.keras.layers.Dense(units = 1))

# Sumário do modelo
model.summary()

# Compilar o modelo
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss = 'mean_squared_error')

# Treinamento
epochs_hist = model.fit(X_train, y_train, epochs = 1000)

"""# Etapa 6: Avaliação do modelo"""

# Visualização do dicionário com os resultados
epochs_hist.history.keys()

# Gráfico com os resultados da loss function
plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss']);

# Visualização dos pesos
model.get_weights()

# Previsões com o modelo treinado, com a temperatura de 5 graus
temp = 5
revenue = model.predict([temp])
print('Revenue Predictions Using Trained ANN =', revenue)

# Gráfico com a reta da regressão linear
plt.scatter(X_train, y_train, color = 'gray')
plt.plot(X_train, model.predict(X_train), color = 'red')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Generated vs. Temperature @Ice Cream WV');

"""# Etapa 7: Confirmar os resultados usando sklearn"""

X_train.shape

X_train = X_train.values.reshape(-1,1)

X_train.shape

y_train = y_train.values.reshape(-1,1)

y_train.shape

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

regressor.coef_

regressor.intercept_

plt.scatter(X_train, y_train, color = 'gray')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Generated vs. Temperature @Ice Cream WV');

temp = 5
revenue = regressor.predict([[temp]])
print('Revenue Predictions Using Trained ANN =', revenue)