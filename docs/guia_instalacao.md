# Guia de Instalação

## Pré-requisitos

- **Python** 3.9 ou superior
- **pip** (gerenciador de pacotes Python)
- Recomendado: **GPU NVIDIA** com CUDA para treinamento mais rápido

## Passo a Passo

### 1. Clonar ou copiar o projeto

```bash
cd py-yolo-pipe
```

### 2. Criar ambiente virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Verificar instalação

```bash
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

### 5. (Opcional) Instalar CUDA para GPU

Se tiver GPU NVIDIA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verifique se o PyTorch detecta a GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Estrutura de Diretórios

O projeto já vem com a estrutura de pastas criada:

```
py-yolo-pipe/
├── input/       # Coloque suas imagens/vídeos aqui
├── output/      # Resultados aparecerão aqui
├── dataset/     # Dataset de treinamento
├── models/      # Modelos treinados
├── config/      # Configurações
├── src/         # Código fonte
└── docs/        # Documentação
```

## Troubleshooting

### "ultralytics" não encontrado
Certifique-se de ativar o ambiente virtual e reinstalar: `pip install -r requirements.txt`

### Erro de memória no treinamento
Reduza `batch` em `config/settings.yaml` ou use `--batch 4`

### Câmera não abre
Verifique se o índice está correto: `0` para câmera padrão, `1` para secundária
