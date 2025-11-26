# 🐕 Classificação de Raças de Cachorros com Deep Learning

**Trabalho Final de CSN - Machine Learning**  
**Autor:** Luis  
**Data:** 26/11/2025

---

## 📋 Descrição do Projeto

Este projeto implementa um sistema de classificação de raças de cachorros usando **Transfer Learning** com **Convolutional Neural Networks (CNN)**. O modelo é capaz de identificar mais de 70 raças diferentes de cachorros a partir de imagens.

### 🎯 Objetivos

- Implementar e treinar um modelo de classificação de imagens usando CNNs
- Aplicar técnicas de Transfer Learning com arquitetura moderna (EfficientNetB0)
- Utilizar Data Augmentation para melhorar a generalização
- Realizar análise exploratória detalhada dos dados
- Analisar erros de classificação e interpretar resultados

---

## 📊 Dataset

O dataset utilizado contém imagens de **70+ raças de cachorros**, organizado em:
- **Train**: Conjunto de treinamento (~7000+ imagens)
- **Validation**: Conjunto de validação (~1000+ imagens)  
- **Test**: Conjunto de teste (~1000+ imagens)

Estrutura do dataset:
```
archive/
├── dogs.csv                    # Metadados das imagens
├── train/                      # Imagens de treinamento
│   ├── Afghan/
│   ├── Labrador/
│   └── ...
├── valid/                      # Imagens de validação
└── test/                       # Imagens de teste
```

---

## 🏗️ Arquitetura do Modelo

### Transfer Learning com EfficientNetB0

O modelo utiliza **EfficientNetB0** pré-treinado no ImageNet como base:

```
Input (224x224x3)
    ↓
EfficientNetB0 (congelado)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, ReLU) + Dropout(0.5)
    ↓
Dense(256, ReLU) + Dropout(0.3)
    ↓
Dense(num_classes, Softmax)
```

### Por que Transfer Learning?

1. **Eficiência**: Aproveita features já aprendidas no ImageNet
2. **Menos dados**: Não precisa de milhões de imagens para treinar
3. **Melhor performance**: Geralmente supera modelos treinados do zero
4. **Tempo**: Treino muito mais rápido (minutos vs. dias)

### EfficientNetB0

- Arquitetura moderna baseada em **compound scaling**
- Balanceia profundidade, largura e resolução
- Excelente trade-off entre acurácia e eficiência
- ~5.3M parâmetros

---

## 🔧 Técnicas Utilizadas

### 1. Data Augmentation

Para aumentar a diversidade dos dados de treino:
- Rotação aleatória (±20°)
- Deslocamento horizontal/vertical (±20%)
- Zoom aleatório (±20%)
- Espelhamento horizontal
- Cisalhamento (shear)

### 2. Callbacks

- **EarlyStopping**: Para quando não há melhoria (patience=10)
- **ModelCheckpoint**: Salva o melhor modelo
- **ReduceLROnPlateau**: Reduz learning rate quando estagna

### 3. Regularização

- **Dropout** (0.3 e 0.5) para prevenir overfitting
- **Data Augmentation** como regularização implícita

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.12 (⚠️ **não use Python 3.13!**)
- GPU NVIDIA (opcional, mas recomendado)
  - Com CUDA: treino ~100x mais rápido
  - Sem GPU: treino em CPU (mais lento, mas funcional)

### 1. Instalação das Dependências

```powershell
# Criar ambiente virtual (recomendado)
python -m venv venv

# Ativar ambiente
.\venv\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.txt
```

### 2. Análise Exploratória dos Dados

```powershell
python 01_exploratory_analysis.py
```

**Saídas:**
- `class_distribution.png`: Gráficos de distribuição de classes
- `sample_images.png`: Amostras de imagens do dataset

**Análises realizadas:**
- Distribuição de imagens por conjunto (train/valid/test)
- Distribuição de imagens por raça
- Análise de propriedades das imagens (dimensões, tamanho)
- Verificação de integridade dos dados

### 3. Treinamento do Modelo

```powershell
python 02_train_model.py
```

**Saídas:**
- `models/dog_classifier_YYYYMMDD_HHMMSS.keras`: Modelo treinado
- `models/class_indices.json`: Mapeamento de classes
- `models/training_results.json`: Métricas de treinamento
- `training_history.png`: Curvas de treinamento

**Tempo estimado:**
- Com GPU: 10-30 minutos
- Sem GPU: 2-4 horas

**Hiperparâmetros:**
- Batch size: 32
- Learning rate: 0.001
- Épocas máximas: 50 (com early stopping)
- Image size: 224x224

### 4. Avaliação e Análise de Erros

```powershell
python 03_evaluate_model.py
```

**Saídas (em `results/`):**
- `confusion_matrix.png`: Matriz de confusão normalizada
- `error_examples.png`: Exemplos de classificações incorretas
- `correct_examples.png`: Exemplos de predições corretas
- `per_class_performance.png`: Performance por raça
- `classification_report.txt`: Relatório detalhado

**Análises realizadas:**
- Matriz de confusão das principais raças
- Pares de confusão mais comuns
- Performance individual por raça
- Visualização de erros e acertos
- Métricas: Accuracy, Precision, Recall, F1-Score

---

## 📈 Resultados Esperados

### Métricas Típicas

Com o dataset fornecido e a arquitetura implementada, espera-se:

- **Acurácia (Top-1)**: 70-85%
- **Acurácia (Top-5)**: 90-95%
- **Training time**: 10-30 min (GPU) / 2-4h (CPU)

### Interpretação dos Resultados

#### 🎯 Bons Resultados
- Raças muito distintas (ex: Chihuahua vs. Great Dane)
- Raças com características únicas (ex: Dalmatian - manchas)

#### ⚠️ Desafios Comuns
- Raças similares (ex: Golden Retriever vs. Labrador)
- Imagens com fundo complexo
- Diferentes ângulos/poses
- Variação intra-raça (cor, tamanho)

---

## 🔍 Análise de Erros

### O que observar:

1. **Raças confundidas**: Quais pares são mais confundidos?
   - Ex: Se confunde Husky com Malamute → raças realmente similares
   
2. **Performance por raça**: Algumas raças são mais fáceis?
   - Raças únicas (Dálmata) tendem a ter melhor performance
   - Raças similares (Spaniels) tendem a ter mais erros

3. **Confiança das predições**: 
   - Alta confiança em erros → modelo está "convicto" mas errado
   - Baixa confiança → modelo está "em dúvida"

4. **Top-5 Accuracy**: 
   - Se Top-5 >> Top-1 → modelo considera múltiplas raças plausíveis
   - Útil para aplicações com "sugestões"

### Possíveis Observações

Se você observar que:
- **Raças grandes são confundidas entre si**: Modelo pode estar identificando tamanho em vez de características faciais
- **Cores similares causam erros**: Textura/cor pode dominar sobre forma
- **Puppies vs Adults**: Idade pode confundir o modelo

Essas observações **não são problemas**, mas **insights valiosos** sobre o que o modelo aprendeu!

---

## 🧠 Conceitos de Computação Numérica Aplicados

### 1. Convolução (Convolutional Layers)

A operação de convolução é uma **operação matemática** entre uma imagem e um filtro (kernel):

$$
(f * g)(x, y) = \sum_{i=-k}^{k} \sum_{j=-k}^{k} f(i, j) \cdot g(x-i, y-j)
$$

- **Filtros**: Detectam features (bordas, texturas, formas)
- **Compartilhamento de pesos**: Mesmos filtros em toda imagem
- **Invariância a translação**: Detecta features independente da posição

### 2. Backpropagation e Gradiente Descendente

O treinamento usa **gradiente descendente** para minimizar a função de perda:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

Onde:
- $\theta$: Parâmetros (pesos) da rede
- $\eta$: Learning rate
- $\nabla_\theta L$: Gradiente da função de perda
- $L$: Loss function (categorical cross-entropy)

**Backpropagation** usa a **regra da cadeia** para calcular gradientes eficientemente através das camadas.

### 3. Softmax e Cross-Entropy

A camada final usa **softmax** para converter logits em probabilidades:

$$
P(y=k|x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

A **categorical cross-entropy loss** mede a diferença entre predição e verdade:

$$
L = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)
$$

### 4. Normalização Batch

EfficientNet usa **Batch Normalization** para estabilizar treinamento:

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

Benefícios:
- Acelera convergência
- Permite learning rates maiores
- Regularização implícita

### 5. Pooling

**Global Average Pooling** reduz dimensões espaciais:

$$
GAP(X) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} X_{i,j}
$$

Vantagens:
- Reduz parâmetros
- Invariância a translação
- Reduz overfitting

---

## 📚 Estrutura do Código

```
.
├── 01_exploratory_analysis.py    # Análise exploratória dos dados
├── 02_train_model.py             # Treinamento do modelo
├── 03_evaluate_model.py          # Avaliação e análise de erros
├── requirements.txt              # Dependências do projeto
├── README.md                     # Este arquivo
├── archive/                      # Dataset (não versionado)
│   ├── dogs.csv
│   ├── train/
│   ├── valid/
│   └── test/
├── models/                       # Modelos treinados (gerado)
│   ├── dog_classifier_*.keras
│   ├── class_indices.json
│   └── training_results.json
└── results/                      # Resultados da avaliação (gerado)
    ├── confusion_matrix.png
    ├── error_examples.png
    ├── correct_examples.png
    ├── per_class_performance.png
    └── classification_report.txt
```

---

## 💡 Dicas e Troubleshooting

### GPU não detectada

Se você tem GPU NVIDIA mas TensorFlow não detecta:

```powershell
# Verifica CUDA
nvidia-smi

# Instala CUDA Toolkit e cuDNN (se necessário)
# Baixar de: https://developer.nvidia.com/cuda-downloads

# Verifica TensorFlow com GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Memória insuficiente (GPU)

Se ocorrer `Out of Memory`:

```python
# Em 02_train_model.py, reduza o BATCH_SIZE
BATCH_SIZE = 16  # ou 8
```

### Treino muito lento (CPU)

Se não tiver GPU e quiser resultados mais rápidos:

```python
# Reduza EPOCHS e use menos dados
EPOCHS = 10

# Ou use um subset do dataset
train_df = train_df.sample(frac=0.3)  # 30% dos dados
```

### Erros de importação

```powershell
# Reinstala TensorFlow
pip uninstall tensorflow keras
pip install tensorflow==2.15.0
```

---

## 🎓 Relatório da Experimentação

### Perguntas para Responder na Apresentação

1. **Qual foi a acurácia final no conjunto de teste?**
   - Responda com números exatos
   - Compare Top-1 vs Top-5

2. **Quais raças o modelo confunde mais?**
   - Mostre os pares de confusão mais comuns
   - Explique por que (características similares?)

3. **O modelo está aprendendo características ou decorando?**
   - Compare acurácia treino vs validação
   - Se treino >> validação → overfitting
   - Se ambos altos → generalização boa

4. **Quais raças têm melhor/pior performance?**
   - Liste top 5 melhores e piores
   - Explique possíveis motivos

5. **O que o modelo aprendeu?**
   - Está identificando raças ou outros padrões?
   - Ex: cor, tamanho, background?

6. **Data Augmentation ajudou?**
   - Compare com/sem (se tiver tempo)
   - Observe curvas de treino/validação

### Estrutura Sugerida da Apresentação (10-15 min)

1. **Introdução** (2 min)
   - Problema: Classificação de raças
   - Dataset: 70+ raças, ~9000 imagens

2. **Metodologia** (4 min)
   - Transfer Learning com EfficientNetB0
   - Data Augmentation
   - Arquitetura do modelo

3. **Resultados** (5 min)
   - Métricas gerais
   - Matriz de confusão
   - Exemplos de erros e acertos
   - Performance por raça

4. **Análise e Discussão** (3 min)
   - O que funcionou bem
   - Desafios encontrados
   - Possíveis melhorias

5. **Conclusão** (1 min)
   - Resumo dos resultados
   - Aprendizados

---

## 🚀 Possíveis Extensões

Se quiser ir além:

1. **Fine-tuning**: Descongelar últimas camadas do EfficientNet
2. **Ensemble**: Treinar múltiplos modelos e combinar
3. **Outras arquiteturas**: ResNet50, VGG16, EfficientNetB3
4. **Grad-CAM**: Visualizar o que o modelo "olha"
5. **Test-Time Augmentation**: Múltiplas versões da imagem no teste
6. **Class Balancing**: Lidar com classes desbalanceadas

---

## 📖 Referências

- **EfficientNet**: Tan & Le (2019) - "EfficientNet: Rethinking Model Scaling for CNNs"
- **Transfer Learning**: Yosinski et al. (2014) - "How transferable are features in deep neural networks?"
- **Data Augmentation**: Shorten & Khoshgoftaar (2019) - "A survey on Image Data Augmentation"
- **TensorFlow Documentation**: https://www.tensorflow.org/
- **Keras Applications**: https://keras.io/api/applications/

---

## ✅ Checklist para a Apresentação

- [ ] Executei análise exploratória
- [ ] Treinei o modelo até convergência
- [ ] Avaliei no conjunto de teste
- [ ] Gerei todas as visualizações
- [ ] Analisei os erros mais comuns
- [ ] Entendi o que o modelo aprendeu
- [ ] Preparei slides com resultados
- [ ] Testei código antes da apresentação

---

## 📝 Notas Finais

Este projeto demonstra:
- ✅ Aplicação prática de CNNs
- ✅ Transfer Learning eficiente
- ✅ Boas práticas de Machine Learning
- ✅ Análise crítica de resultados
- ✅ Documentação completa

**Boa sorte na apresentação! 🎉**

---

## 📧 Contato

**Autor:** Luis  
**Disciplina:** Computação Simbólica e Numérica (CSN)  
**Data:** 26/11/2025

