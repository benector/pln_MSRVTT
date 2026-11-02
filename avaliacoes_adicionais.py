"""

Frases absurdas sao criadas de um conjunto de moldes explicitos, sem consultar
os textos do teste ao treinar o modelo ou escolher limiares. Os resultados
ficam separados da avaliacao por permutacao, para nao confundir tarefas.
"""
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from lm_utils import tokenizar, mapear, construir_vocab, ModeloNgramas, estatisticas
from config import MIN_FREQ, SEED


SUJEITOS = [
    "the refrigerator", "the bicycle", "the pillow", "the cloud",
    "the toothbrush", "the mountain", "the sandwich", "the clock",
    "the pencil", "the carpet", "the keyboard", "the watermelon",
    "the umbrella", "the suitcase", "the window", "the teapot",
]
PREDICADOS = [
    "is teaching algebra to the ocean",
    "is negotiating a salary with the moon",
    "is eating a dictionary for breakfast",
    "is interviewing a rainbow for a job",
    "is writing a love letter to gravity",
    "is arguing about politics with a shadow",
    "is driving a thought across the sky",
    "is buying a ticket for yesterday",
    "is reading the temperature of a dream",
    "is painting the sound of silence with soup",
]


def avaliar_semantica(modelo, vocab, bruto_validacao, bruto_teste, limiar, out):
    """Subteste independente: 160 legendas reais e 160 frases absurdas.

    A primeira classe vem de legendas de teste nunca usadas para ajustar o
    limiar. A segunda e composta por combinacoes absurdas escritas previamente.
    Nao e uma metrica universal de semantica: avalia somente esses exemplos.
    """
    negativas = [f"{s} {p}" for s in SUJEITOS for p in PREDICADOS]
    assert len(negativas) == len(set(negativas)) == 160
    positivas = [r["caption"] for r in bruto_teste
                 if len(tokenizar(r["caption"])) >= 5][:len(negativas)]
    if len(positivas) < len(negativas):
        raise ValueError("Teste insuficiente para comparacao semantica equilibrada")
    linhas=[]
    for rotulo, frases, origem in ((1, positivas, "legenda_original"),
                                   (0, negativas, "frase_absurda")):
        tokens_brutos=[tokenizar(x) for x in frases]
        tokens=mapear(tokens_brutos,vocab)
        for frase, seq, brutos in zip(frases,tokens,tokens_brutos):
            escore=modelo.score_frase(seq)
            linhas.append({"origem":origem,"rotulo":rotulo,
                           "frase":frase,"score":escore,
                           "predito":int(escore>=limiar),
                           "oov_tokens":sum(t not in vocab for t in brutos),
                           "tokens":len(brutos)})
    df=pd.DataFrame(linhas)
    df.to_csv(out/"classificacao_semantica.csv",index=False)
    tn=int(((df.rotulo==0)&(df.predito==0)).sum())
    fp=int(((df.rotulo==0)&(df.predito==1)).sum())
    fn=int(((df.rotulo==1)&(df.predito==0)).sum())
    tp=int(((df.rotulo==1)&(df.predito==1)).sum())
    p=tp/(tp+fp) if tp+fp else 0
    r=tp/(tp+fn) if tp+fn else 0
    return {"descricao":"Subteste adicional de frases absurdas previamente definidas",
            "limiar":"Mesmo limiar escolhido na validacao para permutacoes",
            "n_positivas":len(positivas),"n_negativas":len(negativas),
            "acuracia":float((df.rotulo==df.predito).mean()),
            "precisao":p,"revocacao":r,"f1":2*p*r/(p+r) if p+r else 0,
            "matriz_confusao":{"tn":tn,"fp":fp,"fn":fn,"tp":tp}}


def comparar_tamanhos(bruto_treino, bruto_validacao, bruto_teste,
                      normalizacao, metodo, n, out, logger):
    """Subamostras de video_ids; validacao e teste constantes.

    Evita escolher fracoes de legendas aleatorias do mesmo video. O vocabulario
    e refeito a partir de cada fracao, com os mesmos criterios.
    """
    ids=sorted({r["video_id"] for r in bruto_treino})
    random.Random(SEED).shuffle(ids)
    resultados=[]
    for fracao in (0.25, 0.50, 0.75, 1.0):
        ids_escolhidos=set(ids[:max(1,round(len(ids)*fracao))])
        linhas=[r for r in bruto_treino if r["video_id"] in ids_escolhidos]
        treino=[tokenizar(r["caption"],normalizacao) for r in linhas]
        val=[tokenizar(r["caption"],normalizacao) for r in bruto_validacao]
        teste=[tokenizar(r["caption"],normalizacao) for r in bruto_teste]
        vocab=construir_vocab(treino,MIN_FREQ)
        modelo=ModeloNgramas(metodo,n,mapear(treino,vocab),vocab)
        ppl_val=modelo.perplexidade(mapear(val,vocab))
        ppl_teste=modelo.perplexidade(mapear(teste,vocab))
        stats=estatisticas(treino,teste,vocab,n)
        linha={"fracao_treino":fracao,"videos_treino":len(ids_escolhidos),
               "legendas_treino":len(linhas),"vocabulario":len(vocab),
               "metodo":metodo,"n":n,"normalizacao":normalizacao,
               "perplexidade_val":ppl_val,"perplexidade_teste":ppl_teste,**stats}
        resultados.append(linha)
        pd.DataFrame(resultados).to_csv(out/"efeito_tamanho_treino.csv",index=False)
        logger.info("Tamanho treino %.0f%%: %d videos; PPL teste %.3f",
                    fracao*100,len(ids_escolhidos),ppl_teste)
    return resultados
