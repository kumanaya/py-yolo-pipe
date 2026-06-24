# Guia de Inferência

## Comandos Básicos

### Imagem Única

```bash
python src/inference.py --source input/imagem.jpg
```

Apenas exibe contagem no terminal. Para visualizar:

```bash
python src/inference.py --source input/imagem.jpg --show
```

Para salvar imagem anotada:

```bash
python src/inference.py --source input/imagem.jpg --save
```

Resultado salvo em `output/imagem.jpg`.

### Lote de Imagens

```bash
python src/inference.py --source input/
```

Processa todas as imagens em `input/` e exibe sumário.

### Vídeo

```bash
python src/inference.py --source input/video.mp4 --save
```

```bash
python src/inference.py --source input/video.mp4 --show
```

### Webcam (Tempo Real)

```bash
python src/inference.py --source 0 --show
```

Pressione `q` para sair da janela de visualização.

## Opções

| Argumento      | Descrição                                      | Padrão           |
|----------------|------------------------------------------------|------------------|
| `--source`     | Caminho da imagem, vídeo, pasta ou ID câmera   | `input`          |
| `--model`      | Caminho do modelo treinado                     | `models/best.pt` |
| `--conf`       | Threshold de confiança (0-1)                   | `0.5`            |
| `--iou`        | Threshold IoU para NMS (0-1)                   | `0.45`           |
| `--save`       | Salvar resultados em output/                   | `False`          |
| `--show`       | Exibir janela com resultados                   | `False`          |
| `--csv`        | Exportar contagens para arquivo CSV            | -                |
| `--stats-only` | Apenas estatísticas no terminal                | `False`          |

## Exportação CSV

```bash
python src/inference.py --source input/ --csv resultados.csv
```

Gera CSV com colunas: `source`, `pipe_straight`, `pipe_bent`, `pipe_connector`, `pipe_valve`, `pipe_flange`.

## Thresholds

Ajuste `--conf` e `--iou` para equilibrar precisão e recall:

```bash
# Alta precisão (menos falsos positivos)
python src/inference.py --source input/ --conf 0.7

# Alto recall (detecta mais, pode ter falsos positivos)
python src/inference.py --source input/ --conf 0.3

# Menos sobreposição de boxes
python src/inference.py --source input/ --iou 0.3
```

## Modelos Customizados

```bash
# Usar modelo específico
python src/inference.py --source input/ --model models/meu_modelo.pt
```

## Interpretação dos Resultados

O sistema exibe:

1. **Bounding boxes** coloridos ao redor de cada tubo detectado
2. **Rótulo** com o nome da classe e confiança
3. **Painel de contagem** no canto superior esquerdo com:
   - Quantidade por classe (cada cor representa um tipo)
   - Total de tubos na cena

### Exemplo de Saída no Terminal

```
==================================================
Pipe Detection & Counting Results
==================================================
File: tubulacao_industrial.jpg
  pipe_straight: 12
  pipe_bent: 4
  pipe_connector: 3
  pipe_valve: 2
  pipe_flange: 5
  TOTAL: 26
```

## Troubleshooting

### "Model not found"
Treine um modelo primeiro: `python src/train.py`

### Baixa detecção
- Reduza `--conf` (ex: `--conf 0.25`)
- Treine com mais épocas ou mais dados
- Verifique se as classes em `dataset/data.yaml` correspondem ao treinamento

### Câmera lenta
- Use modelo nano (`train.py --model yolo26n`)
- Reduza a resolução da câmera
