# ============================================================================
# NOTEBOOK DIDÁTICO: CORRELAÇÃO E FEATURE IMPORTANCE
# ============================================================================
# Objetivo: Demonstrar como a correlação entre variáveis distorce a 
#           interpretação da feature importance em modelos de árvore
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configurações
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

print("="*70)
print("📊 NOTEBOOK DIDÁTICO: CORRELAÇÃO E FEATURE IMPORTANCE")
print("="*70)

# ============================================================================
# 1. CRIAÇÃO DO DATASET CONTROLADO
# ============================================================================

print("\n" + "="*70)
print("1. CRIAÇÃO DO DATASET CONTROLADO")
print("="*70)

np.random.seed(42)
n = 1000

# x1 e x2 são ALTAMENTE CORRELACIONADAS
x1 = np.random.normal(size=n)
x2 = x1 + np.random.normal(scale=0.1, size=n)  # correlação ~0.99
x3 = np.random.normal(size=n)  # independente

# Target: depende APENAS de x1 e x3
y = (x1 + x3 > 0).astype(int)

df = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'x3': x3,
    'target': y
})

print(f"\n📊 Shape do dataset: {df.shape}")
print(f"\n📋 Primeiras 5 linhas:")
print(df.head())

print(f"\n📋 Distribuição do target:")
print(df['target'].value_counts())

# ============================================================================
# 2. MATRIZ DE CORRELAÇÃO - O PROBLEMA JÁ APARECE AQUI
# ============================================================================

print("\n" + "="*70)
print("2. ANÁLISE DE CORRELAÇÃO")
print("="*70)

# Matriz de correlação
plt.figure(figsize=(8, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
            square=True, cbar_kws={'shrink': 0.8})
plt.title('Matriz de Correlação\nx1 e x2 são ALTAMENTE correlacionados (0.99)', 
          fontsize=12)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Correlação entre variáveis:")
print(f"  - Correlação x1 vs x2: {corr_matrix.loc['x1', 'x2']:.3f}")
print(f"  - Correlação x1 vs x3: {corr_matrix.loc['x1', 'x3']:.3f}")

print("\n⚠️ PROBLEMA IDENTIFICADO:")
print("  - x1 e x2 são praticamente a MESMA variável (correlação > 0.99)")
print("  - Isso causa redundância de informação")

# ============================================================================
# 3. CORRELAÇÃO COM O TARGET
# ============================================================================

print("\n" + "="*70)
print("3. CORRELAÇÃO DAS FEATURES COM O TARGET")
print("="*70)

corr_with_target = df.corr()['target'].sort_values(ascending=False)
print("\n📊 Correlação de cada feature com o target:")
for feat, corr in corr_with_target.items():
    if feat != 'target':
        print(f"  - {feat}: {corr:.3f}")

# Plot
plt.figure(figsize=(8, 4))
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in corr_with_target.drop('target').values]
bars = plt.bar(corr_with_target.drop('target').index, 
               corr_with_target.drop('target').values, 
               color=colors, edgecolor='black')
plt.ylabel('Correlação com o Target', fontsize=12)
plt.title('Correlação das Features com a Variável Alvo', fontsize=14)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.ylim(-0.1, 1.0)

# Adicionar valores
for bar, val in zip(bars, corr_with_target.drop('target').values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('correlation_with_target.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n💡 VERDADE SUBJACENTE:")
print("  - O target depende APENAS de x1 e x3")
print("  - x2 é uma cópia quase idêntica de x1 (redundante)")

# ============================================================================
# 4. TREINAMENTO DO MODELO
# ============================================================================

print("\n" + "="*70)
print("4. TREINAMENTO DO MODELO (DECISION TREE)")
print("="*70)

X = df[['x1', 'x2', 'x3']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n📊 Divisão dos dados:")
print(f"  - Treino: {X_train.shape[0]} amostras")
print(f"  - Teste:  {X_test.shape[0]} amostras")

# Decision Tree com profundidade limitada para evitar overfitting
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Acurácia do modelo: {accuracy:.4f}")
print("  (O modelo consegue prever bem, mesmo com features redundantes)")

# ============================================================================
# 5. FEATURE IMPORTANCE - O PROBLEMA APARECE AQUI
# ============================================================================

print("\n" + "="*70)
print("5. FEATURE IMPORTANCE - COMO O MODELO INTERPRETA AS FEATURES")
print("="*70)

importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n📊 Feature Importance (Decision Tree):")
print(importance_df.to_string(index=False))

# Plot
plt.figure(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
bars = plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
plt.xlabel('Importância', fontsize=12)
plt.title('Feature Importance - Decision Tree', fontsize=14)
plt.gca().invert_yaxis()

# Adicionar valores
for bar, val in zip(bars, importance_df['importance']):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. ANÁLISE DO PROBLEMA - POR QUE ISSO ACONTECE?
# ============================================================================

print("\n" + "="*70)
print("6. ANÁLISE DO PROBLEMA")
print("="*70)

print("\n🔍 O QUE ACONTECEU?")
print("-" * 50)

print("""
1. **VERDADE**: O target depende APENAS de x1 e x3. x2 é REDUNDANTE.
   
2. **MODELO**: A Decision Tree escolheu dividir a importância entre x1 e x2.
   
3. **RESULTADO**: x2 aparece como 'importante', mas na verdade é apenas 
   um "eco" de x1.
   
4. **CONCLUSÃO**: Se você olhar apenas a feature importance, pode concluir 
   ERRONEAMENTE que x2 é relevante para o problema.
""")

print("\n📊 COMPARAÇÃO: VERDADE VS MODELO")
print("-" * 50)

truth = pd.DataFrame({
    'feature': ['x1', 'x2', 'x3'],
    'verdadeira_relevancia': ['Sim (original)', 'Não (redundante)', 'Sim (original)'],
    'importancia_modelo': [importance_df[importance_df['feature']=='x1']['importance'].values[0],
                           importance_df[importance_df['feature']=='x2']['importance'].values[0],
                           importance_df[importance_df['feature']=='x3']['importance'].values[0]]
})

print(truth.to_string(index=False))

# ============================================================================
# 7. SIMULAÇÃO: O EFEITO DA CORRELAÇÃO EM DIFERENTES MODELOS
# ============================================================================

print("\n" + "="*70)
print("7. COMPARAÇÃO ENTRE DIFERENTES MODELOS")
print("="*70)

# Testar diferentes modelos
models_to_test = {
    'Decision Tree (max_depth=4)': DecisionTreeClassifier(max_depth=4, random_state=42),
    'Decision Tree (max_depth=10)': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest (n=100)': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

results_importance = {}

for name, model_instance in models_to_test.items():
    model_instance.fit(X_train, y_train)
    results_importance[name] = model_instance.feature_importances_
    
    # Acurácia
    y_pred = model_instance.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 {name}: Acurácia = {acc:.4f}")

# Plot comparativo das importâncias
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, importances) in zip(axes, results_importance.items()):
    bars = ax.bar(feature_names, importances, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_title(name, fontsize=12)
    ax.set_ylabel('Importância')
    ax.set_ylim(0, 0.8)
    
    # Adicionar valores
    for bar, val in zip(bars, importances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. COMO CONTORNAR O PROBLEMA?
# ============================================================================

print("\n" + "="*70)
print("8. COMO CONTORNAR O PROBLEMA?")
print("="*70)

print("\n📋 ESTRATÉGIAS PARA LIDAR COM FEATURES CORRELACIONADAS:")
print("-" * 50)

print("""
1. **ANÁLISE PRÉVIA** ✅ (MAIS IMPORTANTE!)
   - Calcular matriz de correlação ANTES de treinar
   - Identificar pares com correlação > 0.9
   - Decidir qual feature manter (baseado em conhecimento de domínio)

2. **REMOÇÃO MANUAL**
   - Remover x2 (a redundante) e manter x1
   - Exemplo: df.drop('x2', axis=1)

3. **ANÁLISE DE VIF (Fator de Inflação da Variância)**
   - Mede quantitativamente a multicolinearidade
   - VIF > 10 indica problema grave

4. **FEATURE AGGREGATION**
   - Combinar features correlacionadas em uma só (média, PCA)

5. **REGULARIZAÇÃO**
   - Modelos com regularização (Lasso, Ridge) são menos sensíveis
""")

# ============================================================================
# 9. DEMONSTRAÇÃO DA SOLUÇÃO
# ============================================================================

print("\n" + "="*70)
print("9. DEMONSTRAÇÃO DA SOLUÇÃO: REMOVENDO A FEATURE REDUNDANTE")
print("="*70)

# Remover x2 (a feature redundante)
X_clean = X.drop('x2', axis=1)

X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_clean, y, test_size=0.3, random_state=42, stratify=y
)

# Treinar modelo sem a feature redundante
model_clean = DecisionTreeClassifier(max_depth=4, random_state=42)
model_clean.fit(X_train_clean, y_train_clean)

# Importâncias após remoção
importances_clean = model_clean.feature_importances_

print("\n📊 Feature Importance SEM a feature redundante (x2):")
for feat, imp in zip(X_clean.columns, importances_clean):
    print(f"  - {feat}: {imp:.3f}")

# Plot comparativo
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Antes (com x2)
axes[0].bar(feature_names, importances, color=['#3498db', '#e74c3c', '#2ecc71'])
axes[0].set_title('Com x2 (redundante)', fontsize=12)
axes[0].set_ylim(0, 0.8)

# Depois (sem x2)
axes[1].bar(X_clean.columns, importances_clean, color=['#3498db', '#2ecc71'])
axes[1].set_title('Sem x2 (após remoção)', fontsize=12)
axes[1].set_ylim(0, 0.8)

for ax in axes:
    ax.set_ylabel('Importância')

plt.tight_layout()
plt.savefig('solution_demonstration.png', dpi=150, bbox_inches='tight')
plt.show()

# Acurácia antes e depois
acc_before = accuracy_score(y_test, model.predict(X_test))
acc_after = accuracy_score(y_test_clean, model_clean.predict(X_test_clean))

print(f"\n📊 Acurácia antes da remoção: {acc_before:.4f}")
print(f"📊 Acurácia depois da remoção: {acc_after:.4f}")
print(f"📊 Diferença: {acc_after - acc_before:.4f} (praticamente igual!)")

# ============================================================================
# 10. CONCLUSÃO DO NOTEBOOK DIDÁTICO
# ============================================================================

print("\n" + "="*70)
print("10. CONCLUSÃO")
print("="*70)

print("""
🎯 **LIÇÕES APRENDIDAS:**

1. ✅ **Correlação distorce feature importance**
   - Features redundantes dividem importância entre si
   - Você pode achar que x2 é importante, mas é só um 'eco' de x1

2. ✅ **Acurácia não é afetada**
   - O modelo continua prevendo bem mesmo com redundância
   - O problema está na **interpretação**, não na performance

3. ✅ **Solução: análise prévia**
   - SEMPRE calcule a matriz de correlação antes de modelar
   - Remova features redundantes ANTES de interpretar importância

4. ✅ **Mantenha apenas a informação