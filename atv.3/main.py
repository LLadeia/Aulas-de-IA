from regras import classificar_regras
from modelo_ml import prever_conforto

print("=== Sistema de Conforto Térmico ===")

temperatura = float(input("Digite a temperatura: "))
umidade = float(input("Digite a umidade: "))

resultado_regras = classificar_regras(temperatura, umidade)
resultado_ml = prever_conforto(temperatura, umidade)

print("\nResultado por Regras:", resultado_regras)
print("Resultado por Decision Tree:", resultado_ml)
