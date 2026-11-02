"""Pré-processamento, métricas e integração com a NLTK.

Laplace, Witten-Bell, Kneser-Ney e interpolação com desconto absoluto usam
classes oficiais da NLTK. A ÚNICA estratégia implementada neste arquivo é
o backoff absoluto normalizado, indisponível como classe pronta em nltk.lm.
StupidBackoff também usa NLTK, mas não produz probabilidades normalizadas.
"""
import math
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache

from nltk.lm import (
    Laplace,
    WittenBellInterpolated,
    KneserNeyInterpolated,
    AbsoluteDiscountingInterpolated,
    StupidBackoff,
)
from nltk.lm.preprocessing import padded_everygram_pipeline


def tokenizar(texto, normalizacao="lower"):
    """Separa palavras, números e, opcionalmente, sinais de pontuação.

    O texto é colocado em minúsculas nos dois cenários experimentais.
    Não exige download de tokenizadores ou modelos externos.
    """
    if normalizacao == "lower":
        return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", texto.lower())
    if normalizacao == "sem_pontuacao":
        return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", texto.lower())
    if normalizacao == "sem_normalizacao":
        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", texto)
    raise ValueError(f"Normalização inválida: {normalizacao}")


def construir_vocab(treino, min_freq=2):
    """Constrói o vocabulário SOMENTE a partir das legendas de treinamento.

    Palavras com uma única ocorrência são removidas do vocabulário e
    posteriormente substituídas por <UNK> em todas as partições.
    """
    frequencias = Counter(token for frase in treino for token in frase)
    vocab = {token for token, freq in frequencias.items() if freq >= min_freq}
    vocab.update({"<UNK>", "<s>", "</s>"})
    return vocab


def mapear(frases, vocab):
    """Converte tokens desconhecidos em <UNK>, preservando a divisão por frase."""
    return [[token if token in vocab else "<UNK>" for token in frase]
            for frase in frases]


def eventos(frases, n):
    """Gera eventos (histórico, próximo token) para avaliação.

    A NLTK usa seu próprio padded_everygram_pipeline para TREINAMENTO.
    Nesta avaliação, cada frase tem n-1 inícios e um marcador de término,
    que é contabilizado no denominador da perplexidade.
    """
    for frase in frases:
        sequencia = ["<s>"] * (n - 1) + list(frase) + ["</s>"]
        for pos in range(n - 1, len(sequencia)):
            yield tuple(sequencia[pos - n + 1:pos]), sequencia[pos]


# Nomes explícitos, para que o método que está sendo avaliado seja
# identificável diretamente na documentação da biblioteca NLTK.
CLASSES_NLTK = {
    "Laplace": Laplace,
    "WittenBell": WittenBellInterpolated,
    "KneserNey": KneserNeyInterpolated,
    "InterpolacaoDescontoAbsoluto": AbsoluteDiscountingInterpolated,
    "StupidBackoff": StupidBackoff,
}


class ModeloNgramas:
    """Padroniza a interface de treino e avaliação.

    Quatro métodos usam a API oficial da NLTK. O backoff absoluto, única
    exceção, usa contagens de n-gramas e desconto fixo normalizado.
    """

    def __init__(self, metodo, n, treino, vocab):
        if metodo not in CLASSES_NLTK and metodo != "BackoffAbsoluto":
            raise ValueError(f"Método desconhecido: {metodo}")
        self.metodo = metodo
        self.n = n
        self.vocab = set(vocab)
        self.eh_probabilistico = metodo != "StupidBackoff"

        if metodo == "BackoffAbsoluto":
            # Exceção explicitada no relatório: backoff probabilístico manual.
            # Conta sequências de ordem 1..n no treino uma única vez.
            self.contagens = [None] + [defaultdict(Counter) for _ in range(n)]
            for frase in treino:
                for ordem in range(1, n + 1):
                    for contexto, palavra in eventos([frase], ordem):
                        self.contagens[ordem][contexto][palavra] += 1
            self.totais = [None] + [
                {contexto: sum(cont.values())
                 for contexto, cont in self.contagens[ordem].items()}
                for ordem in range(1, n + 1)
            ]
            self.unigramas = self.contagens[1][()]
            self.total_unigramas = sum(self.unigramas.values())
            self.tamanho_vocab = len(self.vocab - {"<s>"})
            self.cache_massa = {}
            self.modelo = None
            return

        # Configurações explícitas da NLTK. Não há fórmulas próprias.
        if metodo == "KneserNey":
            self.modelo = KneserNeyInterpolated(n, discount=0.1)
        elif metodo == "InterpolacaoDescontoAbsoluto":
            self.modelo = AbsoluteDiscountingInterpolated(n, discount=0.75)
        elif metodo == "StupidBackoff":
            self.modelo = StupidBackoff(alpha=0.4, order=n)
        else:
            self.modelo = CLASSES_NLTK[metodo](n)

        # NLTK cria everygrams de todas as ordens necessárias (1..n),
        # adiciona os marcadores de fronteira e ajusta as contagens.
        # Fazemos apenas a passagem das sequências previamente mapeadas.
        data, palavras_vocab = padded_everygram_pipeline(n, treino)
        self.modelo.fit(data, palavras_vocab)
        self._otimizar_consultas_nltk()

    def _otimizar_consultas_nltk(self):
        """Caches exatos, sem substituir fórmulas das classes da NLTK.

        KneserNey._continuation_counts faz uma varredura em todos os
        contextos de ordem maior a cada consulta. A tabela abaixo armazena
        previamente as MESMAS contagens que o método original retornaria.
        As fórmulas de unigram_score e alpha_gamma continuam as da NLTK.
        """
        estimador = getattr(self.modelo, "estimator", None)
        if estimador is None:
            return

        if self.metodo == "KneserNey":
            continua = defaultdict(int)
            totais = defaultdict(int)
            # A NLTK consulta counts[len(context)+2] e compara o sufixo
            # do prefixo com o contexto informado.
            for ordem in range(2, self.n + 1):
                for prefixo, contagens in self.modelo.counts[ordem].items():
                    contexto = tuple(prefixo[1:])
                    for palavra, frequencia in contagens.items():
                        if frequencia > 0:
                            continua[(palavra, contexto)] += 1
                            totais[contexto] += 1

            def contagens_continuacao(palavra, contexto=tuple()):
                contexto = tuple(contexto)
                return continua.get((palavra, contexto), 0), totais.get(contexto, 0)

            estimador._continuation_counts = contagens_continuacao

        # São caches das funções da própria NLTK, não implementações
        # alternativas de suavização/interpolação.
        if hasattr(estimador, "_gamma"):
            estimador._gamma = lru_cache(maxsize=100_000)(estimador._gamma)
        estimador.alpha_gamma = lru_cache(maxsize=300_000)(estimador.alpha_gamma)
        estimador.unigram_score = lru_cache(maxsize=50_000)(estimador.unigram_score)

    @lru_cache(maxsize=300_000)
    def prob(self, palavra, contexto):
        """Probabilidade NLTK ou backoff absoluto normalizado."""
        palavra = palavra if palavra in self.vocab else "<UNK>"
        contexto = tuple(token if token in self.vocab else "<UNK>"
                         for token in contexto)
        if self.metodo == "BackoffAbsoluto":
            return self._prob_backoff(palavra, contexto, self.n)
        return float(self.modelo.score(palavra, contexto[-(self.n - 1):]))

    @lru_cache(maxsize=300_000)
    def _prob_backoff(self, palavra, contexto, ordem):
        """Backoff absoluto com distribuição de ordem inferior renormalizada.

        Para eventos observados: (contagem - D) / total.
        A massa descontada é reservada aos eventos não observados e
        redistribuída segundo a distribuição de ordem inferior.
        Contextos nunca vistos recuam integralmente para a ordem inferior.
        """
        if ordem == 1:
            # Laplace na base da recursão evita probabilidades nulas.
            return ((self.unigramas.get(palavra, 0) + 1) /
                    (self.total_unigramas + self.tamanho_vocab))
        contexto_atual = contexto[-(ordem - 1):]
        cont = self.contagens[ordem].get(contexto_atual, Counter())
        total = self.totais[ordem].get(contexto_atual, 0)
        if not total:
            return self._prob_backoff(palavra, contexto, ordem - 1)
        desconto = 0.5
        if cont[palavra] > 0:
            return (cont[palavra] - desconto) / total
        massa = desconto * len(cont) / total
        chave = (ordem, contexto_atual)
        if chave not in self.cache_massa:
            soma_observados = sum(
                self._prob_backoff(termo, contexto, ordem - 1)
                for termo in cont
            )
            self.cache_massa[chave] = max(1e-12, 1 - soma_observados)
        return (massa * self._prob_backoff(palavra, contexto, ordem - 1)
                / self.cache_massa[chave])

    def score_frase(self, frase):
        """Média das log-probabilidades/log-escores (maior é melhor).

        O piso numérico evita log(0); as saídas brutas ainda são da NLTK.
        """
        log_scores = [math.log(max(self.prob(palavra, hist), 1e-300))
                      for hist, palavra in eventos([frase], self.n)]
        return sum(log_scores) / len(log_scores)

    def perplexidade(self, frases):
        """Exponencial da perda logarítmica média sobre palavras + EOS.

        StupidBackoff é a exceção: a saída deve ser identificada como
        pseudo-perplexidade e não comparada numericamente à PPL normalizada.
        """
        soma_log = 0.0
        quantidade = 0
        for frase in frases:
            if not frase:
                continue
            n_eventos = len(frase) + 1
            soma_log += self.score_frase(frase) * n_eventos
            quantidade += n_eventos
        return math.exp(-soma_log / quantidade) if quantidade else float("nan")

    def gerar(self, prefixo=None, max_tokens=20, seed=42):
        """Amostragem sequencial de Shannon.

        Nos métodos da NLTK, chama LanguageModel.generate diretamente.
        No backoff manual, amostra a distribuição normalizada por contexto.
        """
        rng = random.Random(seed)
        texto = list(prefixo or [])
        cache_distribuicao = {}
        for _ in range(max_tokens):
            if self.metodo == "BackoffAbsoluto":
                contexto = tuple((["<s>"] * (self.n - 1) + texto)[-(self.n - 1):])
                if contexto not in cache_distribuicao:
                    palavras = sorted(self.vocab - {"<s>"})
                    pesos = [self.prob(w, contexto) for w in palavras]
                    cache_distribuicao[contexto] = (palavras, pesos)
                palavras, pesos = cache_distribuicao[contexto]
                palavra = rng.choices(palavras, weights=pesos, k=1)[0]
            else:
                palavra = self.modelo.generate(
                    num_words=1, text_seed=texto, random_seed=rng
                )
            if palavra == "</s>":
                break
            if palavra in ("<s>", "<UNK>"):
                continue
            texto.append(palavra)
        return " ".join(texto)


def estatisticas(treino, teste, vocab, n):
    """Conta tipos, singletons e OOV de forma independente do classificador.

    Os OOV são contados ANTES de o teste ser convertido para <UNK>.
    Contagens de n-gramas são obtidas somente do treinamento.
    """
    contagens = Counter((contexto, palavra)
                        for frase in treino for contexto, palavra in eventos([frase], n))
    total_teste = sum(len(frase) for frase in teste)
    oov = sum(token not in vocab for frase in teste for token in frase)
    return {
        "singletons": sum(freq == 1 for freq in contagens.values()),
        "ngrams_distintos": len(contagens),
        "oov": oov,
        "tokens_teste": total_teste,
        "taxa_oov": oov / total_teste if total_teste else 0,
    }
