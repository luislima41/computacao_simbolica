"""
Análise Exploratória do Dataset de Cachorros
Trabalho Final de CSN - Machine Learning
Autor: Luis
Data: 26/11/2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from PIL import Image
import os

# Configurações
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Diretório base
BASE_DIR = Path(__file__).parent.parent / "data" / "dataset"
CSV_PATH = Path(__file__).parent.parent / "data" / "dataset" / "dogs.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_dataset_info():
    """Carrega e analisa o CSV com informações do dataset"""
    print("=" * 80)
    print("ANÁLISE EXPLORATÓRIA DO DATASET DE CACHORROS")
    print("=" * 80)
    
    # Define as 12 raças selecionadas para o projeto
    selected_breeds = [
        'Siberian Husky', 'Pug', 'Dalmation', 'German Sheperd',
        'Golden Retriever', 'Beagle', 'Bulldog', 'Chihuahua',
        'Doberman', 'Great Dane', 'Rottweiler', 'Chow'
    ]
    
    df = pd.read_csv(CSV_PATH)
    print(f"\n📊 Total de imagens original: {len(df)}")
    print(f"🐕 Raças originais: {df['labels'].nunique()}")
    
    # Filtra apenas as 12 raças selecionadas
    df = df[df['labels'].isin(selected_breeds)].copy()
    print(f"\n⚠️  Filtrando para 12 raças selecionadas...")
    print(f"📊 Total de imagens filtradas: {len(df)}")
    print(f"🐕 Raças selecionadas: {df['labels'].nunique()}")
    print(f"📁 Colunas: {list(df.columns)}")
    
    return df

def analyze_class_distribution(df):
    """Analisa a distribuição de classes (raças)"""
    print("\n" + "=" * 80)
    print("DISTRIBUIÇÃO DE CLASSES POR CONJUNTO")
    print("=" * 80)
    
    # Por conjunto (train/valid/test)
    print("\n📦 Imagens por conjunto:")
    dataset_counts = df['data set'].value_counts()
    for dataset, count in dataset_counts.items():
        print(f"   {dataset:8s}: {count:5d} imagens ({count/len(df)*100:.1f}%)")
    
    # Por raça
    print(f"\n🐕 Total de raças diferentes: {df['labels'].nunique()}")
    
    # Distribuição por raça em cada conjunto
    for dataset in ['train', 'valid', 'test']:
        subset = df[df['data set'] == dataset]
        print(f"\n{dataset.upper()}:")
        print(f"   Raças: {subset['labels'].nunique()}")
        print(f"   Imagens: {len(subset)}")
        print(f"   Média por raça: {len(subset)/subset['labels'].nunique():.1f}")
    
    return df

def plot_class_distribution(df):
    """Cria gráficos de distribuição de classes"""
    print("\n📈 Gerando gráficos de distribuição...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribuição geral por conjunto
    ax = axes[0, 0]
    dataset_counts = df['data set'].value_counts()
    dataset_counts.plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_title('Distribuição de Imagens por Conjunto', fontsize=14, fontweight='bold')
    ax.set_xlabel('Conjunto')
    ax.set_ylabel('Número de Imagens')
    ax.tick_params(axis='x', rotation=0)
    for i, v in enumerate(dataset_counts.values):
        ax.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2. Top 15 raças no conjunto de treino
    ax = axes[0, 1]
    train_df = df[df['data set'] == 'train']
    top_breeds = train_df['labels'].value_counts().head(15)
    top_breeds.plot(kind='barh', ax=ax, color='#9b59b6')
    ax.set_title('Top 15 Raças (Conjunto de Treino)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Número de Imagens')
    ax.set_ylabel('Raça')
    ax.invert_yaxis()
    
    # 3. Distribuição de imagens por raça (histograma)
    ax = axes[1, 0]
    breed_counts = train_df['labels'].value_counts()
    ax.hist(breed_counts.values, bins=30, color='#e67e22', edgecolor='black', alpha=0.7)
    ax.set_title('Distribuição da Quantidade de Imagens por Raça', fontsize=14, fontweight='bold')
    ax.set_xlabel('Número de Imagens por Raça')
    ax.set_ylabel('Frequência')
    ax.axvline(breed_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {breed_counts.mean():.1f}')
    ax.legend()
    
    # 4. Comparação train/valid/test por raça (box plot)
    ax = axes[1, 1]
    data_for_box = []
    labels_for_box = []
    for dataset in ['train', 'valid', 'test']:
        subset = df[df['data set'] == dataset]
        counts = subset['labels'].value_counts().values
        data_for_box.append(counts)
        labels_for_box.append(dataset)
    
    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Distribuição de Imagens por Raça em Cada Conjunto', fontsize=14, fontweight='bold')
    ax.set_ylabel('Número de Imagens')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Salvo: {OUTPUT_DIR / 'class_distribution.png'}")
    plt.show()

def analyze_image_properties(df):
    """Analisa propriedades das imagens (tamanho, dimensões)"""
    print("\n" + "=" * 80)
    print("ANÁLISE DE PROPRIEDADES DAS IMAGENS")
    print("=" * 80)
    
    # Amostra de 100 imagens para análise
    sample_size = min(100, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    widths = []
    heights = []
    aspects = []
    file_sizes = []
    
    print(f"\n🔍 Analisando amostra de {sample_size} imagens...")
    
    for idx, row in sample_df.iterrows():
        img_path = BASE_DIR / row['filepaths']
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspects.append(w/h)
            
            file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
        except Exception as e:
            continue
    
    print(f"\n📏 Dimensões das imagens:")
    print(f"   Largura  - min: {min(widths):4d}px, max: {max(widths):4d}px, média: {np.mean(widths):6.1f}px")
    print(f"   Altura   - min: {min(heights):4d}px, max: {max(heights):4d}px, média: {np.mean(heights):6.1f}px")
    print(f"   Aspecto  - min: {min(aspects):4.2f}, max: {max(aspects):4.2f}, média: {np.mean(aspects):4.2f}")
    print(f"   Tamanho  - min: {min(file_sizes):6.1f}KB, max: {max(file_sizes):6.1f}KB, média: {np.mean(file_sizes):6.1f}KB")
    
    return widths, heights, aspects

def visualize_sample_images(df, n_samples=12):
    """Visualiza amostras aleatórias de imagens"""
    print("\n" + "=" * 80)
    print("VISUALIZAÇÃO DE IMAGENS DE EXEMPLO")
    print("=" * 80)
    
    # Seleciona raças aleatórias
    breeds = df['labels'].unique()
    selected_breeds = np.random.choice(breeds, size=min(n_samples, len(breeds)), replace=False)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, breed in enumerate(selected_breeds):
        breed_df = df[df['labels'] == breed]
        sample_row = breed_df.sample(n=1).iloc[0]
        img_path = BASE_DIR / sample_row['filepaths']
        
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"{breed}\n({sample_row['data set']})", fontsize=10, fontweight='bold')
            axes[idx].axis('off')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Erro ao carregar\n{breed}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sample_images.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Salvo: {OUTPUT_DIR / 'sample_images.png'}")
    plt.show()

def check_data_integrity(df):
    """Verifica integridade dos dados"""
    print("\n" + "=" * 80)
    print("VERIFICAÇÃO DE INTEGRIDADE DOS DADOS")
    print("=" * 80)
    
    missing_files = 0
    total_files = 0
    
    print("\n🔍 Verificando existência de arquivos...")
    
    for idx, row in df.iterrows():
        total_files += 1
        img_path = BASE_DIR / row['filepaths']
        if not img_path.exists():
            missing_files += 1
            if missing_files <= 5:  # Mostra apenas os 5 primeiros
                print(f"   ⚠ Arquivo não encontrado: {row['filepaths']}")
    
    if missing_files > 0:
        print(f"\n   ⚠ Total de arquivos faltando: {missing_files}/{total_files}")
    else:
        print(f"   ✓ Todos os {total_files} arquivos existem!")
    
    # Verifica valores nulos
    print(f"\n📋 Valores nulos no CSV:")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("   ✓ Nenhum valor nulo encontrado!")
    else:
        print(null_counts)

def main():
    """Função principal"""
    # Carrega dados
    df = load_dataset_info()
    
    # Análises
    analyze_class_distribution(df)
    plot_class_distribution(df)
    analyze_image_properties(df)
    visualize_sample_images(df)
    check_data_integrity(df)
    
    print("\n" + "=" * 80)
    print("✓ ANÁLISE EXPLORATÓRIA CONCLUÍDA")
    print("=" * 80)
    print("\nArquivos gerados:")
    print("   • class_distribution.png")
    print("   • sample_images.png")
    print("\n")

if __name__ == "__main__":
    main()
