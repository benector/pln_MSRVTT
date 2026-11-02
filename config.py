"""Parâmetros centrais: altere aqui para reproduzir outros cenários."""
SEED = 42
DATASET_REPO = "VLM2Vec/MSR-VTT"
# O repositório disponibiliza arquivos JSON com anotações textuais.
TRAIN_URL = "https://huggingface.co/datasets/VLM2Vec/MSR-VTT/resolve/main/msrvtt_train_9k.json"
TEST_URL = "https://huggingface.co/datasets/VLM2Vec/MSR-VTT/resolve/main/msrvtt_test_1k.json"
VALIDATION_FRACTION = 0.15
MIN_FREQ = 2
MAX_TRAIN_CAPTIONS = None  # None: usar todos os exemplos disponíveis
OUTPUT = "resultados"
