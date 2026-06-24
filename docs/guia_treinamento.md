# Guia de Treinamento

## Preparação do Dataset

### Formato das Anotações

O YOLO utiliza arquivos `.txt` no formato:

```
<class_id> <x_center> <y_center> <width> <height>
```

Onde:
- `class_id`: índice da classe (0 a 4)
- `x_center`, `y_center`: coordenadas normalizadas (0 a 1) do centro do bounding box
- `width`, `height`: largura e altura normalizadas do bounding box

Exemplo:
```
0 0.5 0.5 0.8 0.1
2 0.3 0.6 0.2 0.15
```

### Ferramentas de Anotação Recomendadas

- **LabelImg** (gratuito): https://github.com/heartexlabs/labelImg
- **CVAT** (web, gratuito): https://www.cvat.ai
- **roboflow** (web, freemium): https://roboflow.com

Exporte sempre no formato **YOLO**.

### Organização das Imagens

```
dataset/
├── images/
│   ├── train/       # Imagens de treino
│   └── val/         # Imagens de validação
├── labels/
│   ├── train/       # Labels de treino (mesmo nome .txt)
│   └── val/         # Labels de validação
└── data.yaml        # Configuração do dataset
```

Cada imagem deve ter seu arquivo de label correspondente:
- `img001.jpg` → `img001.txt`
- `img002.png` → `img002.txt`

## Divisão Treino/Validação

```bash
# Mover arquivos da pasta train/ para train/ e val/ conforme split configurado
python src/prepare_dataset.py --auto-split
```

O split padrão é 85% treino / 15% validação (configurável em `config/settings.yaml`).

## Aumentação de Dados

Para melhorar a generalização do modelo com datasets pequenos:

```bash
python src/prepare_dataset.py --augment
```

Transformações aplicadas:
- Flip horizontal (50%) e vertical (10%)
- Rotação aleatória (±30°)
- Variação de brilho e contraste
- Escalonamento
- Blur Gaussiano

## Estatísticas do Dataset

```bash
python src/prepare_dataset.py
```

Exibe:
- Quantidade de imagens por split
- Distribuição de classes (instâncias por tipo de tubo)

## Treinamento

### Comando Básico

```bash
python src/train.py
```

Usa as configurações de `config/settings.yaml` (padrão: 100 épocas, batch 16, yolo26n).

### Parâmetros Customizados

```bash
# Modelo médio (mais preciso, mais lento)
python src/train.py --model yolo26m

# Mais épocas com batch menor
python src/train.py --epochs 200 --batch 8

# Treinar na GPU (índice 0)
python src/train.py --device 0

# Usar CPU
python src/train.py --device cpu

# Continuar treinamento de onde parou
python src/train.py --resume
```

### Arquiteturas Disponíveis

| Modelo    | Tamanho | mAP@COCO | Velocidade (T4 TensorRT) | Recomendação               |
|-----------|---------|----------|--------------------------|----------------------------|
| yolo26n   | Nano    | 40.1%    | 1.7ms                    | CPU / edge / dispositivos leves |
| yolo26s   | Small   | 47.8%    | 2.8ms                    | Equilíbrio geral           |
| yolo26m   | Medium  | 52.5%    | 4.6ms                    | GPU, melhor precisão       |
| yolo26l   | Large   | 54.3%    | 6.8ms                    | GPU potente                |
| yolo26x   | XLarge  | 56.8%    | 11.8ms                   | GPU de alta performance    |

### Monitoramento

Durante o treinamento, os logs são salvos em:
```
models/pipe_detector/
├── weights/
│   ├── best.pt       # Melhor modelo (menor loss)
│   └── last.pt       # Último checkpoint
├── results.csv       # Métricas por época
└── ...
```

### Visualizar Métricas

```bash
python src/visualize.py --metrics models/pipe_detector/
```

Gera gráficos de loss, precisão, recall e mAP.

## Novidades do YOLO26

O YOLO26 traz melhorias significativas em relação a versões anteriores:

- **Inferência NMS-free**: Pipeline end-to-end sem Non-Maximum Suppression, reduzindo latência
- **Sem DFL**: Detection head mais leve sem Distribution Focal Loss
- **Otimizador MuSGD**: Híbrido Muon+SGD para treinamento mais estável
- **STAL**: Label assignment que prioriza objetos pequenos
- **Progressive Loss**: Transição gradual de supervisão para a cabeça de inferência

## Dicas para Melhores Resultados

1. **Qualidade do dataset**: Use pelo menos 50-100 imagens por classe
2. **Variedade**: Inclua diferentes ângulos, iluminações e fundos
3. **Anotações precisas**: Bounding boxes devem cobrir exatamente o objeto
4. **Aumentação**: Use `--augment` se tiver menos de 200 imagens
5. **Pré-treinamento**: O modelo carrega pesos COCO por padrão (transfer learning)
6. **YOLO26 é NMS-free**: Diferente de versões anteriores, o YOLO26 faz inferência end-to-end sem NMS, simplificando o deployment
7. **Hiperparâmetros**: Ajuste `lr0`, `batch` e `epochs` em `config/settings.yaml`
