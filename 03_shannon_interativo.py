"""Experimento INDEPENDENTE de geração de Shannon.

Nao importa nem executa 02_experimentos.py. Carrega somente dados/treino.json,
ajusta UMA VEZ um LM BackoffAbsoluto trigrama e oferece um menu interativo.
Todos os registros ficam em resultados_shannon/, sem tocar resultados/.

Execute na pasta LM_MSRVTT: python 04_shannon_interativo.py
"""
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

from config import MIN_FREQ
from lm_utils import ModeloNgramas, construir_vocab, mapear, tokenizar


PASTA = Path(__file__).resolve().parent
SAIDA = PASTA / "resultados_shannon"
ARQUIVO = SAIDA / "geracoes.csv"
METADADOS = SAIDA / "configuracao.json"
MODELO = "BackoffAbsoluto"
ORDEM = 3
AMOSTRAS_POR_PREFIXO = 5
MAX_NOVOS_TOKENS = 20

PREFIXOS = [
    ("Geração livre", ""),
    ("Pessoa iniciando ação", "a man is"),
    ("Mulher iniciando ação", "a woman is"),
    ("Atividade culinária", "a woman is cooking"),
    ("Atividade esportiva", "people are playing"),
    ("Animal", "a dog is"),
    ("Música", "a man is singing"),
    ("Cena cotidiana", "a child is sitting"),
    ("Contexto genérico", "in the room"),
    ("Contexto incomum", "a dinosaur is"),
]


def carregar_modelo():
    origem = PASTA / "dados" / "treino.json"
    if not origem.exists():
        raise FileNotFoundError(f"Dados não encontrados: {origem}")
    inicio = time.perf_counter()
    registros = json.loads(origem.read_text(encoding="utf-8"))
    treino = [tokenizar(r["caption"], "lower")
              for r in registros if r["caption"].strip()]
    vocab = construir_vocab(treino, MIN_FREQ)
    modelo = ModeloNgramas(MODELO, ORDEM, mapear(treino, vocab), vocab)
    print(f"\nModelo: {MODELO}, trigrama; {len(treino)} legendas; "
          f"vocabulário={len(vocab)}; preparo={time.perf_counter()-inicio:.1f}s")
    return modelo, vocab, len(treino)


def adicionar_csv(linha):
    SAIDA.mkdir(exist_ok=True)
    novo = not ARQUIVO.exists() or ARQUIVO.stat().st_size == 0
    with ARQUIVO.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "sessao", "prefixo", "amostra", "texto", "seed",
            "modelo", "ordem", "novos_tokens_max"])
        if novo:
            writer.writeheader()
        writer.writerow(linha)


def gerar(modelo, prefixos, sessao, seed_inicial, contador):
    for prefixo in prefixos:
        tokens = tokenizar(prefixo, "lower") if prefixo else []
        print(f"\nPREFIXO: {prefixo or '(livre)'}")
        for i in range(1, AMOSTRAS_POR_PREFIXO + 1):
            seed = seed_inicial + contador
            contador += 1
            texto = modelo.gerar(tokens, max_tokens=MAX_NOVOS_TOKENS, seed=seed)
            print(f"  {i}. {texto}", flush=True)
            adicionar_csv({
                "sessao": sessao,
                "prefixo": prefixo or "(livre)",
                "amostra": i, "texto": texto, "seed": seed,
                "modelo": MODELO, "ordem": ORDEM,
                "novos_tokens_max": MAX_NOVOS_TOKENS,
            })
    print(f"\nSalvo em: {ARQUIVO}")
    return contador


def escolher():
    print("\n" + "=" * 58)
    print("SHANNON - escolha um ou vários prefixos")
    print("=" * 58)
    for i, (categoria, prefixo) in enumerate(PREFIXOS, 1):
        print(f"{i:2}. {categoria}: {prefixo or '(livre)'}")
    print(" P. Digitar um prefixo próprio")
    print(" T. Gerar com todos os prefixos sugeridos")
    print(" Q. Encerrar")
    escolha = input("\nEscolha (ex.: 1,3,5): ").strip().lower()
    if escolha in ("q", "sair"):
        return None
    if escolha == "t":
        return [p for _, p in PREFIXOS]
    if escolha == "p":
        prefixo = input("Digite as primeiras palavras (Enter = livre): ").strip()
        return [prefixo]
    try:
        numeros = [int(p.strip()) for p in escolha.split(",")]
        if not numeros or any(i < 1 or i > len(PREFIXOS) for i in numeros):
            raise ValueError
    except ValueError:
        print("Opção inválida. Tente novamente.")
        return []
    prefixos = list(dict.fromkeys(PREFIXOS[i - 1][1] for i in numeros))
    if input("Adicionar também um prefixo próprio? [s/N]: ").strip().lower() == "s":
        prefixos.append(input("Seu prefixo: ").strip())
    return list(dict.fromkeys(prefixos))


def main():
    print("Experimento de geração de Shannon (independente do 02_experimentos.py)")
    print("O modelo é ajustado somente uma vez nesta sessão.")
    modelo, vocab, n_frases = carregar_modelo()
    SAIDA.mkdir(exist_ok=True)
    sessao = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_inicial = random.SystemRandom().randint(1, 2_000_000_000)
    METADADOS.write_text(json.dumps({
        "modelo": MODELO, "ordem": ORDEM, "min_freq": MIN_FREQ,
        "n_legendas_treino": n_frases, "tamanho_vocabulario": len(vocab),
        "normalizacao": "lower", "amostras_por_prefixo": AMOSTRAS_POR_PREFIXO,
        "max_novos_tokens": MAX_NOVOS_TOKENS,
        "observacao": "Treino apenas uma vez na abertura; resultados append no CSV",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    contador = 0
    while True:
        prefixos = escolher()
        if prefixos is None:
            print(f"\nAté mais! Todas as gerações estão em {ARQUIVO}")
            return
        if not prefixos:
            continue
        contador = gerar(modelo, prefixos, sessao, seed_inicial, contador)


if __name__ == "__main__":
    main()
