import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Carregar dataset
dados = pd.read_csv("dataset.csv")

# Entradas
X = dados[["temperatura", "umidade"]]

# Saída
y = dados["classe"]

# Converter texto para números
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# Criar modelo
modelo = DecisionTreeClassifier(random_state=42)

# Treinar modelo
modelo.fit(X, y_enc)

def prever_conforto(temperatura, umidade):
    resultado = modelo.predict([[temperatura, umidade]])
    return encoder.inverse_transform(resultado)[0]
