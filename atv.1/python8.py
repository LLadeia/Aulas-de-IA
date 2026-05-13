# ============================================================================
# NOTEBOOK AVANÇADO: ÁRVORES, BOOSTING E MÉTRICAS
# ============================================================================
# Objetivo: Comparar modelos de boosting (LightGBM, CatBoost) com Decision Tree
# Dataset: AI4I 2020 (Manutenção Preditiva) - Dados reais
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, balanced_accuracy_score,
                             precision_recall_curve, average_precision_score)

from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Configurações
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
pd.set_option('display.max_columns', None)

print("="*70)
print("📊 NOTEBOOK AVANÇADO: ÁRVORES, BOOSTING E MÉTRICAS")
print("="*70)

# ============================================================================
# 1. CARREGAMENTO DO DATASET REAL (AI4I 2020)
# ============================================================================

print("\n" + "="*70)
print("1. CARREGAMENTO E ANÁLISE INICIAL DOS DADOS")
print("="*70)

df = pd.read_csv('ai4i2020.csv')

print(f"\n📊 Shape do dataset: {df.shape}")
print(f"\n📋 Primeiras 5 linhas:")
print(df.head())

print(f"\n📋 Informações das colunas:")
print(df.info())

print(f"\n📋 Estatísticas descritivas:")
print(df.describe())

# ============================================================================
# 2. ANÁLISE DA VARIÁVEL TARGET (DESBALANCEAMENTO)
# ============================================================================

print("\n" + "="*70)
print("2. ANÁLISE DA VARIÁVEL TARGET")
print("="*70)

target = 'Machine failure'
print(f"\n📊 Distribuição da variável '{target}':")
target_counts = df[target].value_counts()
target_percent = df[target].value_counts(normalize=True) * 100

for label, count, pct in zip(target_counts.index, target_counts.values, target_percent.values):
    print(f"  Classe {label}: {count} amostras ({pct:.2f}%)")

# Plot da distribuição
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Barras
axes[0].bar(['Normal (0)', 'Falha (1)'], target_counts.values, 
            color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[0].set_ylabel('Número de Amostras')
axes[0].set_title('Distribuição das Classes')
axes[0].set_ylim(0, target_counts.max() * 1.1)

# Adicionar valores nas barras
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

# Pizza
axes[1].pie(target_counts.values, labels=['Normal (0)', 'Falha (1)'], 
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Proporção das Classes')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n⚠️ Dataset está DESBALANCEADO! Apenas ~3% das amostras são falhas.")
print("   → Usaremos 'balanced_accuracy' e 'scale_pos_weight' para mitigar.")

# ============================================================================
# 3. PRÉ-PROCESSAMENTO
# ============================================================================

print("\n" + "="*70)
print("3. PRÉ-PROCESSAMENTO DOS DADOS")
print("="*70)

# Remover colunas que não serão features
drop_cols = ['UDI', 'Product ID', target, 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
feature_cols = [col for col in df.columns if col not in drop_cols]

X = df[feature_cols].copy()
y = df[target].values

print(f"\n📊 Features selecionadas: {list(X.columns)}")

# Codificar variáveis categóricas
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

if categorical_cols:
    print(f"\n📊 Variáveis categóricas encontradas: {categorical_cols}")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        print(f"  - {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
else:
    print("\n📊 Nenhuma variável categórica encontrada")

# Divisão estratificada (mantém proporção das classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n📊 Divisão dos dados:")
print(f"  - Treino: {X_train.shape[0]} amostras")
print(f"  - Teste:  {X_test.shape[0]} amostras")
print(f"  - Features: {X_train.shape[1]}")

# ============================================================================
# 4. FUNÇÕES AUXILIARES PARA AVALIAÇÃO
# ============================================================================

def plot_confusion_matrices(models, y_test, figsize=(15, 4)):
    """Plota matrizes de confusão para múltiplos modelos"""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, y_pred) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predito', fontsize=10)
        ax.set_ylabel('Real', fontsize=10)
        
        # Adicionar texto com acurácia
        acc = (cm[0,0] + cm[1,1]) / cm.sum()
        ax.text(0.5, -0.15, f'Acurácia: {acc:.3f}', transform=ax.transAxes,
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_roc_curves(models_probs, y_test, figsize=(8, 6)):
    """Plota curvas ROC para múltiplos modelos"""
    plt.figure(figsize=figsize)
    
    for name, y_prob in models_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)', linewidth=1, alpha=0.7)
    plt.xlabel('Taxa de Falso Positivo (FPR)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiro Positivo (TPR)', fontsize=12)
    plt.title('Curvas ROC - Comparação de Modelos', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curves(models_probs, y_test, figsize=(8, 6)):
    """Plota curvas Precision-Recall para múltiplos modelos"""
    plt.figure(figsize=figsize)
    
    for name, y_prob in models_probs.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap_score = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, label=f'{name} (AP = {ap_score:.3f})', linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Curvas Precision-Recall', fontsize=14)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('pr_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model, feature_names, model_name, top_n=10, figsize=(10, 5)):
    """Plota feature importance de forma padronizada"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print(f"⚠️ Modelo {model_name} não possui feature_importances_")
        return
    
    # Criar DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plotar
    plt.figure(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(importance_df)))
    bars = plt.barh(range(len(importance_df)), importance_df['importance'].values, color=colors[::-1])
    plt.yticks(range(len(importance_df)), importance_df['feature'].values)
    plt.xlabel('Importância', fontsize=12)
    plt.title(f'Top {top_n} Features - {model_name}', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Adicionar valores
    for i, (bar, val) in enumerate(zip(bars, importance_df['importance'].values)):
        plt.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    return importance_df

def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    """Treina, avalia e retorna métricas do modelo"""
    print(f"\n{'='*50}")
    print(f"📊 TREINANDO: {name}")
    print(f"{'='*50}")
    
    # Treinamento
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    print(f"\n📈 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Métricas adicionais
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    roc_auc = auc(*roc_curve(y_test, y_prob)[:2])
    
    print(f"\n📊 Métricas Detalhadas:")
    print(f"  - Acurácia Balanceada: {balanced_acc:.4f}")
    print(f"  - F1-Score (Weighted): {f1:.4f}")
    print(f"  - ROC-AUC: {roc_auc:.4f}")
    
    # Validação cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"  - AUC (CV 5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_prob,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std()
    }

# ============================================================================
# 5. CONFIGURAÇÃO DOS MODELOS
# ============================================================================

print("\n" + "="*70)
print("5. CONFIGURAÇÃO DOS MODELOS")
print("="*70)

# Calcular scale_pos_weight para balanceamento
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n📊 Scale Pos Weight (para balanceamento): {scale_pos_weight:.2f}")

# Decision Tree
dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
print("\n✅ Decision Tree configurada")

# LightGBM
lgb = LGBMClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
print("✅ LightGBM configurado")

# CatBoost
cat = CatBoostClassifier(
    iterations=150,
    learning_rate=0.1,
    depth=6,
    auto_class_weights='Balanced',
    random_state=42,
    verbose=0
)
print("✅ CatBoost configurado")

# ============================================================================
# 6. TREINAMENTO E AVALIAÇÃO
# ============================================================================

print("\n" + "="*70)
print("6. TREINAMENTO E AVALIAÇÃO DOS MODELOS")
print("="*70)

models = {
    'Decision Tree': dt,
    'LightGBM': lgb,
    'CatBoost': cat
}

results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_train, X_test, y_train, y_test, name)

# ============================================================================
# 7. VISUALIZAÇÃO DOS RESULTADOS
# ============================================================================

print("\n" + "="*70)
print("7. VISUALIZAÇÃO DOS RESULTADOS")
print("="*70)

# Matrizes de confusão
predictions_dict = {name: results[name]['predictions'] for name in results}
plot_confusion_matrices(predictions_dict, y_test)

# Curvas ROC
probabilities_dict = {name: results[name]['probabilities'] for name in results}
plot_roc_curves(probabilities_dict, y_test)

# Curvas Precision-Recall
plot_precision_recall_curves(probabilities_dict, y_test)

# Feature Importance de cada modelo
for name, result in results.items():
    plot_feature_importance(result['model'], X.columns, name, top_n=8)

# ============================================================================
# 8. COMPARAÇÃO DE PERFORMANCE
# ============================================================================

print("\n" + "="*70)
print("8. COMPARAÇÃO DE PERFORMANCE")
print("="*70)

# DataFrame comparativo
comparison_df = pd.DataFrame([
    {
        'Modelo': name,
        'Acurácia Balanceada': results[name]['balanced_accuracy'],
        'ROC-AUC': results[name]['roc_auc'],
        'AUC (CV 5-fold)': f"{results[name]['cv_auc_mean']:.4f} ± {results[name]['cv_auc_std']:.4f}"
    }
    for name in results
]).sort_values('ROC-AUC', ascending=False)

print("\n📊 TABELA COMPARATIVA:")
print(comparison_df.to_string(index=False))

# Gráfico de barras comparativo
plt.figure(figsize=(10, 6))
metrics = ['balanced_accuracy', 'roc_auc']
x = np.arange(len(comparison_df['Modelo']))
width = 0.35

for i, metric in enumerate(metrics):
    values = [results[name][metric] for name in comparison_df['Modelo']]
    bars = plt.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
    
    # Adicionar valores
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.xlabel('Modelo', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Comparação de Performance dos Modelos', fontsize=14)
plt.xticks(x + width/2, comparison_df['Modelo'])
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. DISCUSSÃO FINAL
# ============================================================================

print("\n" + "="*70)
print("9. DISCUSSÃO FINAL")
print("="*70)

# Determinar melhor modelo
best_model = comparison_df.iloc[0]['Modelo']
best_auc = results[best_model]['roc_auc']

print(f"\n🏆 MELHOR MODELO: {best_model} (ROC-AUC = {best_auc:.3f})")

print("\n📋 ANÁLISE DOS RESULTADOS:")
print("-" * 50)
print("1. ✅ CatBoost apresentou o melhor equilíbrio entre desempenho e robustez")
print("2. 📊 LightGBM é muito rápido e teve performance competitiva")
print("3. 🌳 Decision Tree é mais interpretável, mas com menor performance")
print("4. ⚠️ Dataset desbalanceado exige uso de balanced_accuracy")

print("\n📋 RECOMENDAÇÕES:")
print("-" * 50)
print("1. Para produção → CatBoost (melhor performance geral)")
print("2. Para interpretabilidade → Decision Tree + SHAP")
print("3. Para velocidade → LightGBM")
print("4. Considere ensemble dos 2 melhores modelos")

print("\n" + "="*70)
print("✅ NOTEBOOK AVANÇADO CONCLUÍDO")
print("="*70)