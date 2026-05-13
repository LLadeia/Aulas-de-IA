# regras.py - com umidade
def classificar_regras(temperatura, umidade):
    # Frio: temperaturas baixas
    if temperatura <= 20:
        return "Frio"
    # Confortavel: faixa ideal
    elif 21 <= temperatura <= 26:
        # Mesmo com umidade mais alta, mantém confortável até 60%
        if umidade <= 60:
            return "Confortavel"
        else:
            # Umidade muito alta pode ser desconfortável
            return "Quente"
    # Quente: temperaturas altas ou combinação médio-alta temperatura + alta umidade
    elif temperatura >= 30:
        return "Quente"
    elif temperatura > 26:  # 27, 28, 29
        if umidade >= 65:
            return "Quente"
        else:
            return "Confortavel"
    else:
        return "Confortavel"