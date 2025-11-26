"""
Avaliação e Análise de Erros do Modelo
Trabalho Final de CSN - Machine Learning
Autor: Luis
Data: 26/11/2025

Este script avalia o modelo treinado e realiza análise detalhada dos erros,
incluindo matriz de confusão, visualização de predições incorretas e
análise das raças mais confundidas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import json

# Configurações
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Diretórios
BASE_DIR = Path(__file__).parent / "archive"
CSV_PATH = BASE_DIR / "dogs.csv"
MODEL_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Parâmetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_model_and_classes():
    """Carrega o modelo treinado e o mapeamento de classes"""
    print("=" * 80)
    print("CARREGAMENTO DO MODELO")
    print("=" * 80)
    
    # Encontra o modelo mais recente
    model_files = list(MODEL_DIR.glob("dog_classifier_*.keras"))
    if not model_files:
        raise FileNotFoundError(f"Nenhum modelo encontrado em {MODEL_DIR}")
    
    model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📦 Carregando modelo: {model_path.name}")
    
    model = keras.models.load_model(model_path)
    print("✓ Modelo carregado com sucesso!")
    
    # Carrega mapeamento de classes
    with open(MODEL_DIR / 'class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    # Inverte o dicionário (índice -> nome)
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    print(f"✓ Mapeamento de {len(class_indices)} classes carregado")
    
    return model, class_indices, idx_to_class

def load_test_data(selected_breeds=None):
    """Carrega dados de teste"""
    print("\n" + "=" * 80)
    print("CARREGAMENTO DOS DADOS DE TESTE")
    print("=" * 80)
    
    df = pd.read_csv(CSV_PATH)
    test_df = df[df['data set'] == 'test'].copy()
    
    # Filtra apenas as raças selecionadas se fornecido
    if selected_breeds:
        test_df = test_df[test_df['labels'].isin(selected_breeds)].copy()
        print(f"\n⚠️  Filtrando para {len(selected_breeds)} raças selecionadas...")
    
    test_df['full_path'] = test_df['filepaths'].apply(lambda x: str(BASE_DIR / x))
    
    print(f"\n📊 Amostras de teste: {len(test_df)}")
    print(f"🐕 Raças: {test_df['labels'].nunique()}")
    
    # Cria gerador
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='full_path',
        y_col='labels',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Importante para manter ordem
    )
    
    return test_generator, test_df

def evaluate_model(model, test_generator):
    """Avalia o modelo no conjunto de teste"""
    print("\n" + "=" * 80)
    print("AVALIAÇÃO DO MODELO")
    print("=" * 80)
    
    print("\n🔍 Realizando predições...")
    
    # Predições
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Métricas gerais
    test_loss, test_acc, test_top5 = model.evaluate(test_generator, verbose=0)
    
    print(f"\n📊 Métricas no conjunto de teste:")
    print(f"   Loss:           {test_loss:.4f}")
    print(f"   Acurácia:       {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Top-5 Acurácia: {test_top5:.4f} ({test_top5*100:.2f}%)")
    
    return predictions, y_pred, y_true

def plot_confusion_matrix(y_true, y_pred, class_names, top_n=30):
    """
    Plota matriz de confusão.
    Como temos muitas classes, mostra apenas as top N mais comuns.
    """
    print("\n" + "=" * 80)
    print("MATRIZ DE CONFUSÃO")
    print("=" * 80)
    
    # Calcula matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    # Identifica as classes mais frequentes no teste
    unique, counts = np.unique(y_true, return_counts=True)
    top_classes_idx = unique[np.argsort(counts)[-top_n:]]
    
    # Filtra matriz para top N classes
    cm_filtered = cm[np.ix_(top_classes_idx, top_classes_idx)]
    class_names_filtered = [class_names[i] for i in top_classes_idx]
    
    # Plota
    fig, ax = plt.subplots(figsize=(20, 18))
    
    # Normaliza por linha (true labels)
    cm_normalized = cm_filtered.astype('float') / cm_filtered.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Configura eixos
    ax.set(xticks=np.arange(cm_normalized.shape[1]),
           yticks=np.arange(cm_normalized.shape[0]),
           xticklabels=class_names_filtered,
           yticklabels=class_names_filtered,
           title=f'Matriz de Confusão Normalizada (Top {top_n} Raças mais Comuns)',
           ylabel='Raça Verdadeira',
           xlabel='Raça Predita')
    
    # Rotaciona labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Adiciona valores nas células (apenas se > 0.1)
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            if cm_normalized[i, j] > 0.1:
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                       ha="center", va="center", color="white" if cm_normalized[i, j] > 0.5 else "black",
                       fontsize=6)
    
    fig.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Salvo: {RESULTS_DIR / 'confusion_matrix.png'}")
    plt.show()
    
    return cm

def analyze_classification_errors(y_true, y_pred, predictions, test_df, idx_to_class, top_n=10):
    """Analisa os erros de classificação mais comuns"""
    print("\n" + "=" * 80)
    print("ANÁLISE DE ERROS DE CLASSIFICAÇÃO")
    print("=" * 80)
    
    # Identifica erros
    errors_mask = y_true != y_pred
    errors_idx = np.where(errors_mask)[0]
    
    print(f"\n❌ Total de erros: {len(errors_idx)} / {len(y_true)} ({len(errors_idx)/len(y_true)*100:.2f}%)")
    
    # Analisa pares de confusão mais comuns
    error_pairs = []
    for idx in errors_idx:
        true_label = idx_to_class[y_true[idx]]
        pred_label = idx_to_class[y_pred[idx]]
        confidence = predictions[idx][y_pred[idx]]
        error_pairs.append((true_label, pred_label, confidence))
    
    # Conta pares
    from collections import Counter
    pair_counts = Counter([(true, pred) for true, pred, _ in error_pairs])
    most_common = pair_counts.most_common(top_n)
    
    print(f"\n🔍 Top {top_n} pares de confusão mais comuns:")
    print(f"{'Nº':<4} {'Verdadeira':<20} {'Predita':<20} {'Frequência':>10}")
    print("-" * 60)
    for i, ((true_label, pred_label), count) in enumerate(most_common, 1):
        print(f"{i:<4} {true_label:<20} {pred_label:<20} {count:>10}")
    
    return errors_idx, error_pairs, most_common

def visualize_errors(errors_idx, test_df, y_true, y_pred, predictions, idx_to_class, n_samples=12):
    """Visualiza exemplos de predições incorretas"""
    print("\n" + "=" * 80)
    print("VISUALIZAÇÃO DE ERROS")
    print("=" * 80)
    
    # Seleciona amostras aleatórias de erros
    sample_errors = np.random.choice(errors_idx, size=min(n_samples, len(errors_idx)), replace=False)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, error_idx in enumerate(sample_errors):
        ax = axes[i]
        
        # Carrega imagem
        img_path = test_df.iloc[error_idx]['full_path']
        img = Image.open(img_path)
        
        # Labels
        true_label = idx_to_class[y_true[error_idx]]
        pred_label = idx_to_class[y_pred[error_idx]]
        confidence = predictions[error_idx][y_pred[error_idx]]
        
        # Top-3 predições
        top3_idx = np.argsort(predictions[error_idx])[-3:][::-1]
        top3_labels = [idx_to_class[idx] for idx in top3_idx]
        top3_probs = [predictions[error_idx][idx] for idx in top3_idx]
        
        # Plota
        ax.imshow(img)
        ax.axis('off')
        
        title = f"✓ {true_label}\n✗ {pred_label} ({confidence:.2f})"
        ax.set_title(title, fontsize=9, color='red')
        
        # Adiciona top-3 como texto
        top3_text = "\n".join([f"{j+1}. {lbl}: {prob:.2f}" 
                               for j, (lbl, prob) in enumerate(zip(top3_labels, top3_probs))])
        ax.text(0.02, 0.98, top3_text, transform=ax.transAxes, 
               fontsize=7, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Exemplos de Classificações Incorretas (✓=Verdadeira, ✗=Predita)', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'error_examples.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Salvo: {RESULTS_DIR / 'error_examples.png'}")
    plt.show()

def visualize_correct_predictions(y_true, y_pred, test_df, predictions, idx_to_class, n_samples=12):
    """Visualiza exemplos de predições corretas com alta confiança"""
    print("\n" + "=" * 80)
    print("VISUALIZAÇÃO DE PREDIÇÕES CORRETAS")
    print("=" * 80)
    
    # Identifica acertos
    correct_mask = y_true == y_pred
    correct_idx = np.where(correct_mask)[0]
    
    # Pega confiança das predições corretas
    confidences = [predictions[idx][y_pred[idx]] for idx in correct_idx]
    
    # Seleciona os mais confiantes
    top_confident_idx = correct_idx[np.argsort(confidences)[-n_samples:]]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, idx in enumerate(top_confident_idx):
        ax = axes[i]
        
        # Carrega imagem
        img_path = test_df.iloc[idx]['full_path']
        img = Image.open(img_path)
        
        # Labels
        label = idx_to_class[y_true[idx]]
        confidence = predictions[idx][y_pred[idx]]
        
        # Top-3 predições
        top3_idx = np.argsort(predictions[idx])[-3:][::-1]
        top3_labels = [idx_to_class[i] for i in top3_idx]
        top3_probs = [predictions[idx][i] for i in top3_idx]
        
        # Plota
        ax.imshow(img)
        ax.axis('off')
        
        title = f"✓ {label}\nConfiança: {confidence:.3f}"
        ax.set_title(title, fontsize=9, color='green', fontweight='bold')
        
        # Adiciona top-3 como texto
        top3_text = "\n".join([f"{j+1}. {lbl}: {prob:.3f}" 
                               for j, (lbl, prob) in enumerate(zip(top3_labels, top3_probs))])
        ax.text(0.02, 0.98, top3_text, transform=ax.transAxes, 
               fontsize=7, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Predições Corretas com Alta Confiança', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'correct_examples.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Salvo: {RESULTS_DIR / 'correct_examples.png'}")
    plt.show()

def analyze_per_class_performance(y_true, y_pred, idx_to_class, top_n=20):
    """Analisa performance por classe"""
    print("\n" + "=" * 80)
    print("PERFORMANCE POR RAÇA")
    print("=" * 80)
    
    # Calcula acurácia por classe
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    classes = sorted(set(y_true))
    class_accuracies = []
    
    for cls in classes:
        mask = (y_true == cls)
        if np.sum(mask) > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            class_accuracies.append((idx_to_class[cls], acc, np.sum(mask)))
    
    # Ordena por acurácia
    class_accuracies.sort(key=lambda x: x[1])
    
    print(f"\n🏆 Top {top_n} raças com MELHOR performance:")
    print(f"{'Nº':<4} {'Raça':<25} {'Acurácia':>10} {'Amostras':>10}")
    print("-" * 55)
    for i, (breed, acc, count) in enumerate(class_accuracies[-top_n:][::-1], 1):
        print(f"{i:<4} {breed:<25} {acc:>9.2%} {count:>10}")
    
    print(f"\n⚠️  Top {top_n} raças com PIOR performance:")
    print(f"{'Nº':<4} {'Raça':<25} {'Acurácia':>10} {'Amostras':>10}")
    print("-" * 55)
    for i, (breed, acc, count) in enumerate(class_accuracies[:top_n], 1):
        print(f"{i:<4} {breed:<25} {acc:>9.2%} {count:>10}")
    
    # Plota
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Melhores
    ax = axes[0]
    best = class_accuracies[-top_n:][::-1]
    breeds_best = [b[0] for b in best]
    accs_best = [b[1] for b in best]
    ax.barh(breeds_best, accs_best, color='#2ecc71')
    ax.set_xlabel('Acurácia')
    ax.set_title(f'Top {top_n} Raças com Melhor Performance', fontweight='bold')
    ax.set_xlim([0, 1])
    ax.invert_yaxis()
    for i, v in enumerate(accs_best):
        ax.text(v + 0.01, i, f'{v:.2%}', va='center', fontsize=8)
    
    # Piores
    ax = axes[1]
    worst = class_accuracies[:top_n]
    breeds_worst = [b[0] for b in worst]
    accs_worst = [b[1] for b in worst]
    ax.barh(breeds_worst, accs_worst, color='#e74c3c')
    ax.set_xlabel('Acurácia')
    ax.set_title(f'Top {top_n} Raças com Pior Performance', fontweight='bold')
    ax.set_xlim([0, 1])
    ax.invert_yaxis()
    for i, v in enumerate(accs_worst):
        ax.text(v + 0.01, i, f'{v:.2%}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'per_class_performance.png', dpi=300, bbox_inches='tight')
    print(f"\n   ✓ Salvo: {RESULTS_DIR / 'per_class_performance.png'}")
    plt.show()
    
    return class_accuracies

def generate_classification_report(y_true, y_pred, idx_to_class):
    """Gera relatório detalhado de classificação"""
    print("\n" + "=" * 80)
    print("RELATÓRIO DE CLASSIFICAÇÃO (SKLEARN)")
    print("=" * 80)
    
    # Nomes das classes
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    
    # Gera relatório
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    
    # Salva em arquivo
    report_text = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    with open(RESULTS_DIR / 'classification_report.txt', 'w') as f:
        f.write("RELATÓRIO DE CLASSIFICAÇÃO\n")
        f.write("=" * 80 + "\n\n")
        f.write(report_text)
    
    print(f"\n✓ Relatório completo salvo em: {RESULTS_DIR / 'classification_report.txt'}")
    
    # Mostra resumo
    print(f"\n📊 Resumo geral:")
    print(f"   Acurácia:  {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
    print(f"   Macro Avg:")
    print(f"      Precision: {report['macro avg']['precision']:.4f}")
    print(f"      Recall:    {report['macro avg']['recall']:.4f}")
    print(f"      F1-Score:  {report['macro avg']['f1-score']:.4f}")
    print(f"   Weighted Avg:")
    print(f"      Precision: {report['weighted avg']['precision']:.4f}")
    print(f"      Recall:    {report['weighted avg']['recall']:.4f}")
    print(f"      F1-Score:  {report['weighted avg']['f1-score']:.4f}")

def main():
    """Função principal"""
    print("\n" + "=" * 80)
    print("🔍 AVALIAÇÃO E ANÁLISE DE ERROS DO MODELO 🔍")
    print("=" * 80)
    
    # 1. Carrega modelo
    model, class_indices, idx_to_class = load_model_and_classes()
    
    # Define as mesmas 12 raças usadas no treinamento
    selected_breeds = [
        'Siberian Husky', 'Pug', 'Dalmation', 'German Sheperd',
        'Golden Retriever', 'Beagle', 'Bulldog', 'Chihuahua',
        'Doberman', 'Great Dane', 'Rottweiler', 'Chow'
    ]
    
    # 2. Carrega dados de teste (filtrando as 12 raças)
    test_generator, test_df = load_test_data(selected_breeds=selected_breeds)
    
    # 3. Avalia modelo
    predictions, y_pred, y_true = evaluate_model(model, test_generator)
    
    # 4. Matriz de confusão
    cm = plot_confusion_matrix(y_true, y_pred, idx_to_class)
    
    # 5. Análise de erros
    errors_idx, error_pairs, most_common_errors = analyze_classification_errors(
        y_true, y_pred, predictions, test_df, idx_to_class
    )
    
    # 6. Visualiza erros
    if len(errors_idx) > 0:
        visualize_errors(errors_idx, test_df, y_true, y_pred, predictions, idx_to_class)
    
    # 7. Visualiza acertos
    visualize_correct_predictions(y_true, y_pred, test_df, predictions, idx_to_class)
    
    # 8. Performance por classe
    class_accuracies = analyze_per_class_performance(y_true, y_pred, idx_to_class)
    
    # 9. Relatório sklearn
    generate_classification_report(y_true, y_pred, idx_to_class)
    
    print("\n" + "=" * 80)
    print("✓ AVALIAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 80)
    print(f"\n📁 Arquivos gerados em {RESULTS_DIR}:")
    print("   • confusion_matrix.png")
    print("   • error_examples.png")
    print("   • correct_examples.png")
    print("   • per_class_performance.png")
    print("   • classification_report.txt")
    print("\n")

if __name__ == "__main__":
    main()
