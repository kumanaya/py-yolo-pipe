# YOLO Pipe Counter

Sistema de visão computacional para detecção, classificação e contagem de tubos industriais usando YOLO26 (Ultralytics).

## Recursos

- Detecta e classifica variações de tubos (reto, curvado, conector, válvula, flange)
- Contagem automática por classe e total
- Suporte a imagens, vídeos e webcam
- Aumentação de dataset integrada
- Exportação de resultados para CSV
- Visualização com bounding boxes e painel de contagem
- Interface via linha de comando

## Estrutura do Projeto

```
py-yolo-pipe/
├── input/              # Imagens/vídeos para inferência
├── output/             # Resultados processados
├── dataset/            # Dataset de treino (imagens + labels YOLO)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml       # Config do dataset para YOLO
├── models/             # Pesos treinados
│   └── best.pt
├── config/
│   └── settings.yaml   # Configurações do projeto
├── src/
│   ├── train.py        # Treinamento do modelo
│   ├── inference.py    # Inferência e contagem
│   ├── prepare_dataset.py  # Preparação e aumento de dados
│   ├── visualize.py    # Ferramentas de visualização
│   └── utils.py        # Utilitários
├── docs/
│   ├── guia_instalacao.md
│   ├── guia_treinamento.md
│   └── guia_inferencia.md
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar modelo
python src/train.py

# Inferência em uma imagem
python src/inference.py --source input/imagem.jpg --show

# Inferência em vídeo
python src/inference.py --source input/video.mp4 --save
```

## Classes Detectáveis

| ID | Classe          | Descrição                          |
|----|-----------------|------------------------------------|
| 0  | pipe_straight   | Tubo reto                          |
| 1  | pipe_bent       | Tubo curvado / joelho              |
| 2  | pipe_connector  | Conecção / T-junção                |
| 3  | pipe_valve      | Válvula / registro                 |
| 4  | pipe_flange     | Flange / placa de conexão          |

O modelo aprende automaticamente as variações visuais de cada classe através do treinamento com dataset anotado.

## Dataset

Coloque suas imagens anotadas (formato YOLO) em `dataset/images/train` e `dataset/labels/train`.

```bash
# Estatísticas do dataset
python src/prepare_dataset.py

# Divisão treino/validação
python src/prepare_dataset.py --auto-split

# Aumentação de dados
python src/prepare_dataset.py --augment
```

## Licença

MIT
