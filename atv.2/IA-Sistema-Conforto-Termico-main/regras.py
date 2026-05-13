"""Simple rule-based classifier for thermal comfort."""

def classificar(temperatura: float, umidade: float) -> str:
    """Return comfort class based on temperature and humidity rules.

    - temperatura < 20 -> Frio
    - 20 <= temperatura <= 26 e 30 <= umidade <= 60 -> Confortável
    - temperatura > 26 e umidade <= 60 -> Quente
    - temperatura > 26 e umidade > 60 -> Abafado
    """
    if temperatura < 20:
        return "Frio"
    if 20 <= temperatura <= 26 and 30 <= umidade <= 60:
        return "Confortável"
    if temperatura > 26 and umidade <= 60:
        return "Quente"
    if temperatura > 26 and umidade > 60:
        return "Abafado"
    # fallback (shouldn't happen with current rules)
    return "Desconhecido"
