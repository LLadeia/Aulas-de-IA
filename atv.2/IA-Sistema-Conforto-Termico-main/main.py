"""Main script to interact with the user and compare both approaches."""

from regras import classificar as regras_classificar
import modelo_ml


def main():
    print("Classificador de conforto térmico")
    try:
        temp = float(input("Digite a temperatura (°C): "))
        hum = float(input("Digite a umidade (%): "))
    except ValueError:
        print("Valores inválidos. Use números.")
        return

    resultado_regras = regras_classificar(temp, hum)
    resultado_ml = modelo_ml.predict(temp, hum)

    print("\nResultado do sistema de regras:", resultado_regras)
    print("Resultado do modelo de ML:", resultado_ml)


if __name__ == "__main__":
    main()
