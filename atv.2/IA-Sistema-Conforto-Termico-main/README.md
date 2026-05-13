# IA Sistema de Conforto Térmico

Este projeto simples compara duas abordagens para classificar o conforto térmico em função de temperatura (°C) e umidade (%):

1. **Sistema simbólico baseado em regras** (`regras.py`)
2. **Modelo de Machine Learning** utilizando `DecisionTreeClassifier` (`modelo_ml.py`)

## Arquivos

- `dataset.csv` – conjunto de dados usado para treinar o modelo de ML.
- `regras.py` – implementa a lógica `if/else` para classificação.
- `modelo_ml.py` – carrega o dataset, treina e faz previsões com a árvore de decisão.
- `main.py` – interface de terminal para o usuário inserir valores e comparar resultados.
- `requirements.txt` – dependências Python.

## Uso

1. Crie e ative um ambiente virtual (opcional, mas recomendado):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. Instale as dependências:
   ```powershell
   pip install -r requirements.txt
   ```
3. Execute o programa:
   ```powershell
   python main.py
   ```
4. Digite temperatura e umidade quando solicitado.

Os resultados das duas abordagens serão exibidos na tela.

## Observações

O dataset é artificial e pequeno; o foco é demonstrar a comparação entre regras e ML.
