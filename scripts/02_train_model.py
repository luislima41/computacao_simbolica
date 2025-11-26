"""
Treinamento de Modelo CNN para Classificação de Raças de Cachorros
Trabalho Final de CSN - Machine Learning
Autor: Luis
Data: 26/11/2025

Este script implementa transfer learning usando EfficientNetB0 pré-treinado no ImageNet.
Inclui data augmentation, callbacks para early stopping e salvamento do melhor modelo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime

# Configurações de GPU (se disponível)
print("=" * 80)
print("CONFIGURAÇÃO DO AMBIENTE")
print("=" * 80)
print(f"\n🔧 TensorFlow versão: {tf.__version__}")
print(f"🎮 GPUs disponíveis: {len(tf.config.list_physical_devices('GPU'))}")

if tf.config.list_physical_devices('GPU'):
    print("✓ GPU detectada! Treinamento será acelerado.")
    # Configurar para crescimento de memória dinâmico
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠ Nenhuma GPU detectada. Treinamento será em CPU.")

# Seed para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

# Diretórios
BASE_DIR = Path(__file__).parent / "archive"
CSV_PATH = BASE_DIR / "dogs.csv"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Hiperparâmetros
IMG_SIZE = (224, 224)  # Tamanho de entrada do EfficientNetB0
BATCH_SIZE = 32
EPOCHS = 30  # Suficiente para 12 classes
LEARNING_RATE = 0.00005  # Bem menor para treinar todas as camadas

def load_data():
    """Carrega o dataset e prepara os caminhos"""
    print("\n" + "=" * 80)
    print("CARREGAMENTO DOS DADOS")
    print("=" * 80)
    
    df = pd.read_csv(CSV_PATH)
    
    # Adiciona o caminho completo
    df['full_path'] = df['filepaths'].apply(lambda x: str(BASE_DIR / x))
    
    print(f"\n📊 Total de imagens: {len(df)}")
    print(f"🐕 Total de raças: {df['labels'].nunique()}")
    
    # Seleciona 12 raças visualmente MUITO distintas para melhor acurácia
    selected_breeds = [
        'Siberian Husky',    # Olhos azuis, pelagem cinza/branca
        'Pug',               # Focinho achatado, pequeno
        'Dalmation',         # Manchas pretas únicas
        'German Sheperd',    # Pastor alemão clássico
        'Golden Retriever',  # Dourado, pelo longo
        'Beagle',            # Tricolor, orelhas caídas
        'Bulldog',           # Corpo atarracado, focinho achatado
        'Chihuahua',         # Muito pequeno
        'Doberman',          # Preto/marrom, orelhas pontiagudas
        'Great Dane',        # Gigante
        'Rottweiler',        # Preto com marcas marrom
        'Chow'               # Língua azul, pelagem densa
    ]
    
    # Separar por conjunto
    train_df = df[df['data set'] == 'train'].copy()
    valid_df = df[df['data set'] == 'valid'].copy()
    test_df = df[df['data set'] == 'test'].copy()
    
    # Filtra apenas as raças selecionadas
    print(f"\n⚠️  Filtrando para {len(selected_breeds)} raças visualmente distintas...")
    print(f"   Raças selecionadas: {', '.join(selected_breeds[:4])}...")
    
    train_df = train_df[train_df['labels'].isin(selected_breeds)].copy()
    valid_df = valid_df[valid_df['labels'].isin(selected_breeds)].copy()
    test_df = test_df[test_df['labels'].isin(selected_breeds)].copy()
    
    common_breeds = set(train_df['labels'].unique())
    
    print(f"\n📦 Divisão do dataset:")
    print(f"   Treino:     {len(train_df):5d} imagens ({len(train_df)/len(df)*100:5.1f}%)")
    print(f"   Validação:  {len(valid_df):5d} imagens ({len(valid_df)/len(df)*100:5.1f}%)")
    print(f"   Teste:      {len(test_df):5d} imagens ({len(test_df)/len(df)*100:5.1f}%)")
    
    return train_df, valid_df, test_df, len(common_breeds)

def create_data_generators(train_df, valid_df, test_df):
    """
    Cria geradores de dados com data augmentation.
    
    Data Augmentation é crucial para:
    - Reduzir overfitting
    - Aumentar a robustez do modelo a variações
    - Simular diferentes condições de captura
    """
    print("\n" + "=" * 80)
    print("CRIAÇÃO DOS GERADORES DE DADOS")
    print("=" * 80)
    
    # Generator para treino COM data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # Normalização [0,1]
        rotation_range=20,           # Rotação aleatória ±20°
        width_shift_range=0.2,       # Deslocamento horizontal ±20%
        height_shift_range=0.2,      # Deslocamento vertical ±20%
        shear_range=0.2,             # Cisalhamento
        zoom_range=0.2,              # Zoom aleatório ±20%
        horizontal_flip=True,        # Espelhamento horizontal
        fill_mode='nearest'          # Preenchimento de pixels vazios
    )
    
    # Generator para validação e teste SEM augmentation (apenas normalização)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("\n🔄 Data Augmentation configurado:")
    print("   • Rotação: ±20°")
    print("   • Deslocamento: ±20%")
    print("   • Zoom: ±20%")
    print("   • Espelhamento horizontal")
    print("   • Cisalhamento: 0.2")
    
    # Cria os geradores
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='full_path',
        y_col='labels',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    valid_generator = val_test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='full_path',
        y_col='labels',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='full_path',
        y_col='labels',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n✓ Geradores criados:")
    print(f"   Treino:    {train_generator.samples} amostras, {len(train_generator)} batches")
    print(f"   Validação: {valid_generator.samples} amostras, {len(valid_generator)} batches")
    print(f"   Teste:     {test_generator.samples} amostras, {len(test_generator)} batches")
    
    # Salva o mapeamento de classes
    class_indices = train_generator.class_indices
    with open(MODEL_DIR / 'class_indices.json', 'w') as f:
        json.dump(class_indices, f, indent=2)
    print(f"\n💾 Mapeamento de classes salvo em: {MODEL_DIR / 'class_indices.json'}")
    
    return train_generator, valid_generator, test_generator, class_indices

def build_model(num_classes):
    """
    Constrói modelo usando Transfer Learning com EfficientNetB0.
    
    Transfer Learning:
    - Usa conhecimento aprendido no ImageNet (1.4M imagens, 1000 classes)
    - Congela as camadas convolucionais (extração de features)
    - Treina apenas as camadas finais de classificação
    - Muito mais eficiente que treinar do zero
    
    EfficientNetB0:
    - Arquitetura moderna, balanceando profundidade, largura e resolução
    - Compound scaling method
    - Excelente trade-off entre acurácia e eficiência
    """
    print("\n" + "=" * 80)
    print("CONSTRUÇÃO DO MODELO")
    print("=" * 80)
    
    # Carrega modelo base pré-treinado
    base_model = EfficientNetB0(
        include_top=False,           # Remove camada de classificação original
        weights='imagenet',          # Usa pesos pré-treinados
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Descongela TODAS as camadas para treinar o modelo completo
    base_model.trainable = True
    
    print(f"\n🏗️  Arquitetura base: EfficientNetB0")
    print(f"   Parâmetros totais: {base_model.count_params():,}")
    print(f"   Camadas congeladas: {len(base_model.layers)}")
    
    # Constrói o modelo completo
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    
    # Modelo base
    x = base_model(inputs, training=False)
    
    # Global Average Pooling (reduz espacialidade)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Camadas densas de classificação
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Regularização
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Camada de saída
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compila o modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )
    
    print(f"\n📊 Modelo final:")
    print(f"   Parâmetros treináveis: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")
    print(f"   Parâmetros não-treináveis: {sum([np.prod(v.shape) for v in model.non_trainable_weights]):,}")
    print(f"   Classes de saída: {num_classes}")
    
    return model

def create_callbacks(model_name='dog_classifier'):
    """
    Cria callbacks para treinamento.
    
    Callbacks:
    - EarlyStopping: Para quando não há melhoria (evita overfitting)
    - ModelCheckpoint: Salva o melhor modelo
    - ReduceLROnPlateau: Reduz learning rate quando estagna
    """
    print("\n" + "=" * 80)
    print("CONFIGURAÇÃO DE CALLBACKS")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"{model_name}_{timestamp}.keras"
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,  # Ajustado para deadline
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,  # Ajustado
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\n✓ Callbacks configurados:")
    print("   • Early Stopping (patience=20)")
    print("   • Model Checkpoint")
    print("   • Reduce LR on Plateau (factor=0.5, patience=8)")
    print(f"\n💾 Modelo será salvo em: {model_path}")
    
    return callbacks, model_path

def plot_training_history(history, save_path='training_history.png'):
    """Plota curvas de treinamento"""
    print("\n📈 Gerando gráficos de treinamento...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Acurácia
    ax = axes[0, 0]
    ax.plot(history.history['accuracy'], label='Treino', linewidth=2)
    ax.plot(history.history['val_accuracy'], label='Validação', linewidth=2)
    ax.set_title('Acurácia do Modelo', fontsize=14, fontweight='bold')
    ax.set_xlabel('Época')
    ax.set_ylabel('Acurácia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss
    ax = axes[0, 1]
    ax.plot(history.history['loss'], label='Treino', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validação', linewidth=2)
    ax.set_title('Perda (Loss) do Modelo', fontsize=14, fontweight='bold')
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-5 Acurácia
    ax = axes[1, 0]
    ax.plot(history.history['top5_accuracy'], label='Treino', linewidth=2)
    ax.plot(history.history['val_top5_accuracy'], label='Validação', linewidth=2)
    ax.set_title('Top-5 Acurácia', fontsize=14, fontweight='bold')
    ax.set_xlabel('Época')
    ax.set_ylabel('Top-5 Acurácia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Comparação final
    ax = axes[1, 1]
    final_metrics = {
        'Treino': [history.history['accuracy'][-1], history.history['top5_accuracy'][-1]],
        'Validação': [history.history['val_accuracy'][-1], history.history['val_top5_accuracy'][-1]]
    }
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, final_metrics['Treino'], width, label='Treino', color='#3498db')
    ax.bar(x + width/2, final_metrics['Validação'], width, label='Validação', color='#2ecc71')
    ax.set_title('Métricas Finais', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(['Acurácia', 'Top-5 Acurácia'])
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adiciona valores nas barras
    for i, (k, v) in enumerate(final_metrics.items()):
        for j, val in enumerate(v):
            x_pos = j + (i - 0.5) * width
            ax.text(x_pos, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Salvo: {save_path}")
    plt.show()

def main():
    """Função principal de treinamento"""
    print("\n" + "=" * 80)
    print("🐕 TREINAMENTO DO CLASSIFICADOR DE RAÇAS DE CACHORROS 🐕")
    print("=" * 80)
    
    # 1. Carrega dados
    train_df, valid_df, test_df, num_classes = load_data()
    
    # 2. Cria geradores
    train_gen, valid_gen, test_gen, class_indices = create_data_generators(train_df, valid_df, test_df)
    
    # 3. Constrói modelo
    model = build_model(num_classes)
    
    # 4. Configura callbacks
    callbacks, model_path = create_callbacks()
    
    # 5. Treina modelo
    print("\n" + "=" * 80)
    print("🚀 INICIANDO TREINAMENTO")
    print("=" * 80)
    print(f"\n⚙️  Configuração:")
    print(f"   Épocas máximas: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Imagem size: {IMG_SIZE}")
    print("\n")
    
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Avaliação final no conjunto de teste
    print("\n" + "=" * 80)
    print("📊 AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("=" * 80)
    
    test_loss, test_acc, test_top5 = model.evaluate(test_gen, verbose=1)
    
    print(f"\n✓ Resultados finais:")
    print(f"   Loss:           {test_loss:.4f}")
    print(f"   Acurácia:       {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Top-5 Acurácia: {test_top5:.4f} ({test_top5*100:.2f}%)")
    
    # 7. Salva métricas
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_top5_accuracy': float(test_top5),
        'num_classes': num_classes,
        'training_samples': len(train_df),
        'validation_samples': len(valid_df),
        'test_samples': len(test_df),
        'epochs_trained': len(history.history['loss']),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(MODEL_DIR / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Resultados salvos em: {MODEL_DIR / 'training_results.json'}")
    
    # 8. Plota histórico
    plot_training_history(history)
    
    print("\n" + "=" * 80)
    print("✓ TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    print(f"\n📁 Arquivos gerados:")
    print(f"   • {model_path}")
    print(f"   • {MODEL_DIR / 'class_indices.json'}")
    print(f"   • {MODEL_DIR / 'training_results.json'}")
    print(f"   • training_history.png")
    print("\n")

if __name__ == "__main__":
    main()
