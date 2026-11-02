"""Experimentos de modelagem de linguagem n-grama.

Fluxo:
1. Carrega partições fixas, checa a ausência de vídeos repetidos.
2. Para cada método/ordem: tokeniza, cria vocabulário no TREINO, ajusta LM
   e mede perplexidade na validação e no teste.
3. StupidBackoff é mantido à parte por usar escores não normalizados.
4. Seleciona por PPL de validação o melhor trigrama probabilístico.
5. Avalia frases originais versus embaralhadas e frases absurdas.
6. Gera frases de Shannon e testa 25/50/75/100% do volume de treino.
7. Salva cada configuração logo após a conclusão (permite retomada).

Implementações probabilísticas: Laplace/WB/KN/Interpolação via NLTK;
backoff absoluto normalizado é a única implementação própria.
"""
import json
import math
import random
import time
import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import SEED, MIN_FREQ, OUTPUT
from lm_utils import tokenizar, construir_vocab, mapear, ModeloNgramas, estatisticas
from avaliacoes_adicionais import avaliar_semantica, comparar_tamanhos

OUT=Path(OUTPUT)
OUT.mkdir(exist_ok=True)
# 13 configurações: 4 métodos NLTK probabilísticos + 1 backoff normalizado
# manual + StupidBackoff NLTK (somente score, não PPL), em ordens 2 e 3,
# além de uma comparação de normalização com Kneser--Ney.
CONFIGURACOES = [
    ("lower", 2, "Laplace"),
    ("lower", 3, "Laplace"),
    ("lower", 2, "WittenBell"),
    ("lower", 3, "WittenBell"),
    ("lower", 2, "KneserNey"),
    ("lower", 3, "KneserNey"),
    ("lower", 2, "InterpolacaoDescontoAbsoluto"),
    ("lower", 3, "InterpolacaoDescontoAbsoluto"),
    ("lower", 2, "BackoffAbsoluto"),
    ("lower", 3, "BackoffAbsoluto"),
    ("lower", 2, "StupidBackoff"),
    ("lower", 3, "StupidBackoff"),
    ("sem_pontuacao", 3, "KneserNey"),
]

LOG = logging.getLogger("msrvtt")

def configurar_logs():
    """Duplica o registro de progresso no terminal e em arquivo."""
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT/"execucao.log", encoding="utf-8", mode="a")):
        handler.setFormatter(fmt)
        LOG.addHandler(handler)

def salvar_csv_atomico(df, destino):
    """Substitui o CSV somente depois de escrever uma cópia temporária."""
    tmp=destino.with_suffix(destino.suffix+".tmp")
    df.to_csv(tmp,index=False)
    os.replace(tmp,destino)

def salvar_json_atomico(obj, destino):
    tmp=destino.with_suffix(destino.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
    os.replace(tmp,destino)

def carregar(nome):
    """Lê uma partição da pasta dados sem baixar arquivos."""
    return json.loads((Path("dados")/f"{nome}.json").read_text(encoding="utf-8"))

def preparar(registros,modo):
    return [tokenizar(r["caption"],modo) for r in registros if r["caption"].strip()]

def grafico_barras(df,coluna,arquivo,titulo):
    """Gera gráfico sem valores inventados: lê as métricas calculadas."""
    fig,ax=plt.subplots(figsize=(10,5))
    if df.empty:
        plt.close(fig)
        return
    df.plot.bar(x="config",y=coluna,ax=ax,legend=False)
    ax.set_title(titulo)
    ax.set_ylabel(coluna.replace("_"," "))
    ax.set_xlabel("")
    ax.tick_params(axis="x",rotation=45)
    fig.tight_layout()
    fig.savefig(OUT/arquivo,dpi=170)
    plt.close(fig)

def gerar_somente_com_prefixos(bruto):
    """Reaproveita a selecao ja registrada, sem recalcular os experimentos."""
    caminho=OUT/"metricas.csv"
    if not caminho.exists():
        raise RuntimeError("Execute primeiro: python 02_experimentos.py --reiniciar")
    metricas=pd.read_csv(caminho)
    candidatos=metricas[(metricas.normalizacao=="lower")&
                        (metricas.n==3)&(metricas.status=="ok")].dropna(
                            subset=["perplexidade_val"])
    if candidatos.empty:
        raise RuntimeError("Nenhum trigrama concluido no CSV; execute os experimentos primeiro.")
    melhor=candidatos.sort_values("perplexidade_val").iloc[0]
    nome=str(melhor["metodo"])
    LOG.info("Shannon: reconstruindo %s, trigrama, selecionado na validacao",nome)
    treino=[tokenizar(r["caption"],"lower") for r in bruto["treino"] if r["caption"].strip()]
    vocab=construir_vocab(treino,MIN_FREQ)
    modelo=ModeloNgramas(nome,3,mapear(treino,vocab),vocab)
    posicoes=[i for i,x in enumerate(sys.argv) if x=="--prefixo"]
    prefixos=[]
    for pos in posicoes:
        if pos+1>=len(sys.argv) or sys.argv[pos+1].startswith("--"):
            raise ValueError('Use --prefixo "a man" (pode repetir).')
        prefixos.append(sys.argv[pos+1].strip())
    if not prefixos:
        prefixos=[""]  # geracao sem prefixo
    saidas=[]
    for j,p in enumerate(prefixos):
        tokens=tokenizar(p,"lower")
        print(f"\nPREFIXO: {p or '(livre)'}",flush=True)
        for k in range(5):
            frase=modelo.gerar(tokens,max_tokens=20,seed=SEED+1000+j*5+k)
            print(f"  {k+1}. {frase}",flush=True)
            saidas.append({"prefixo":p or "(livre)","amostra":k+1,
                           "texto":frase,"modelo":nome})
    destino=OUT/"geracoes_interativas.csv"
    salvar_csv_atomico(pd.DataFrame(saidas),destino)
    LOG.info("Shannon: %d frases geradas, salvas em %s",len(saidas),destino)

def main():
    configurar_logs()
    inicio_total=time.perf_counter()
    configuracoes = [c for c in CONFIGURACOES
                    if c[2] in {"InterpolacaoDescontoAbsoluto", "BackoffAbsoluto", "StupidBackoff"}] \
        if "--somente-generalizacao" in sys.argv else CONFIGURACOES
    inicio=time.perf_counter()
    bruto={k:carregar(k) for k in ("treino","validacao","teste")}
    LOG.info("Carregamento dos dados: %.2fs",time.perf_counter()-inicio)
    LOG.info("Particoes carregadas: treino=%d, validacao=%d, teste=%d legendas",
             len(bruto["treino"]),len(bruto["validacao"]),len(bruto["teste"]))
    ids={k:{r["video_id"] for r in bruto[k]} for k in bruto}
    if ids["treino"]&ids["validacao"] or ids["treino"]&ids["teste"] or ids["validacao"]&ids["teste"]:
        raise RuntimeError("Vazamento: mesmo video_id presente em mais de uma particao")
    LOG.info("Videos sem sobreposicao: treino=%d, validacao=%d, teste=%d",
             *(len(ids[k]) for k in ("treino","validacao","teste")))
    if "--somente-geracao" in sys.argv:
        gerar_somente_com_prefixos(bruto)
        return
    LOG.info("Iniciando %d configuracoes; cada resultado sera salvo ao terminar.",len(configuracoes))
    preparacoes={}
    caminho_metricas=OUT/"metricas.csv"
    # Retomada automatica: NAO APAGA resultados concluidos anteriormente.
    # Para refazer tudo explicitamente: python 02_experimentos.py --reiniciar
    if "--reiniciar" in sys.argv:
        resultados=[]
        LOG.warning("--reiniciar: iniciando CSV novo (resultados antigos serao substituidos).")
    elif caminho_metricas.exists():
        resultados=pd.read_csv(caminho_metricas).to_dict("records")
        LOG.info("Retomada: %d linhas encontradas em %s",len(resultados),caminho_metricas)
    else:
        resultados=[]
    # Não misturar números da implementação antiga com esta revisão.
    # Com --reiniciar todos os modelos são calculados do zero; na retomada,
    # somente linhas da mesma versão do algoritmo são reaproveitadas.
    versoes = {
        "Laplace": "nltk_v4", "WittenBell": "nltk_v4",
        "KneserNey": "nltk_kn_v4",
        "InterpolacaoDescontoAbsoluto": "nltk_absinterp_v4",
        "BackoffAbsoluto": "manual_absbackoff_v4",
        "StupidBackoff": "nltk_stupid_v4",
    }
    antigas = [r for r in resultados
               if r.get("versao_implementacao") != versoes.get(r.get("metodo"))]
    if antigas:
        LOG.warning("Ignorando %d resultado(s) de implementacoes antigas.",len(antigas))
        resultados = [r for r in resultados if r not in antigas]
        salvar_csv_atomico(pd.DataFrame(resultados), caminho_metricas)
    concluidos={
        (str(r["normalizacao"]),int(r["n"]),str(r["metodo"]))
        for r in resultados if r.get("status")=="ok"
        and pd.notna(r.get("n"))
    }
    if not caminho_metricas.exists() or "--reiniciar" in sys.argv:
        salvar_csv_atomico(pd.DataFrame(columns=["normalizacao","n","metodo","config","status","perplexidade_val","perplexidade_teste","tempo_preparacao_s","tempo_estatisticas_s","tempo_treino_s","tempo_validacao_s","tempo_teste_s","tempo_total_s"]),caminho_metricas)
    for indice,(normalizacao,n,metodo) in enumerate(configuracoes,1):
        if (normalizacao,n,metodo) in concluidos:
            LOG.info("[%d/%d] IGNORADO (ja concluido): %s n=%d %s",indice,len(configuracoes),normalizacao,n,metodo)
            continue
        inicio_cfg=time.perf_counter()
        tempo_prep=0.0
        # Prepara cada normalização uma única vez e reutiliza os mesmos
        # tokens/vocabulário para uma comparação justa entre métodos.
        if normalizacao not in preparacoes:
            inicio=time.perf_counter()
            original={k:preparar(v,normalizacao) for k,v in bruto.items()}
            vocab=construir_vocab(original["treino"],MIN_FREQ)
            dados={k:mapear(v,vocab) for k,v in original.items()}
            preparacoes[normalizacao]=(original,vocab,dados)
            tempo_prep=time.perf_counter()-inicio
            LOG.info("Preparo %s: %.2fs; vocabulario=%d",normalizacao,tempo_prep,len(vocab))
        original,vocab,dados=preparacoes[normalizacao]
        LOG.info("[%d/%d] Inicio %s n=%d %s",indice,len(configuracoes),normalizacao,n,metodo)
        tempos={"tempo_preparacao_s":round(tempo_prep,3),"tempo_estatisticas_s":None,"tempo_treino_s":None,"tempo_validacao_s":None,"tempo_teste_s":None}
        linha={"normalizacao":normalizacao,"n":n,"metodo":metodo,
               "versao_implementacao":versoes[metodo],
               "config":f"{metodo} n={n} {normalizacao}"}
        modelo=None
        try:
            inicio=time.perf_counter()
            # Singletons são calculados sobre o treino; OOV, sobre o teste
            # ANTES de mapear palavras desconhecidas para <UNK>.
            stats=estatisticas(original["treino"],original["teste"],vocab,n)
            tempos["tempo_estatisticas_s"]=round(time.perf_counter()-inicio,3)
            LOG.info("[%d/%d] Estatisticas: %.2fs",indice,len(configuracoes),tempos["tempo_estatisticas_s"])
            inicio=time.perf_counter()
            modelo=ModeloNgramas(metodo,n,dados["treino"],vocab)
            tempos["tempo_treino_s"]=round(time.perf_counter()-inicio,3)
            LOG.info("[%d/%d] Treinamento: %.2fs",indice,len(configuracoes),tempos["tempo_treino_s"])
            inicio=time.perf_counter()
            ppl_val=modelo.perplexidade(dados["validacao"])
            tempos["tempo_validacao_s"]=round(time.perf_counter()-inicio,3)
            LOG.info("[%d/%d] Validacao: %.2fs; PPL=%.4f",indice,len(configuracoes),tempos["tempo_validacao_s"],ppl_val)
            inicio=time.perf_counter()
            ppl_test=modelo.perplexidade(dados["teste"])
            tempos["tempo_teste_s"]=round(time.perf_counter()-inicio,3)
            if modelo.eh_probabilistico:
                linha.update({"status":"ok","perplexidade_val":ppl_val,
                              "perplexidade_teste":ppl_test,**stats})
                LOG.info("[%d/%d] Teste: %.2fs; PPL=%.4f",indice,len(configuracoes),tempos["tempo_teste_s"],ppl_test)
            else:
                # StupidBackoff não representa distribuição normalizada.
                # Guarda pseudo-perplexidade em colunas distintas, sem usá-la
                # em ranking por PPL ou na seleção para classificação.
                linha.update({"status":"ok","perplexidade_val":float("nan"),
                              "perplexidade_teste":float("nan"),
                              "pseudo_perplexidade_val":ppl_val,
                              "pseudo_perplexidade_teste":ppl_test,**stats})
                LOG.info("[%d/%d] Teste: %.2fs; pseudo-PPL=%.4f (nao comparavel)",indice,len(configuracoes),tempos["tempo_teste_s"],ppl_test)
        except Exception as exc:
            LOG.exception("[%d/%d] Falha: %s",indice,len(configuracoes),exc)
            linha.update({"status":"erro","erro":str(exc)})
        finally:
            linha.update(tempos)
            linha["tempo_total_s"]=round(time.perf_counter()-inicio_cfg,3)
            # Se esta configuracao ja tinha falhado, atualiza a linha em vez de duplicar.
            resultados=[r for r in resultados if not (
                str(r.get("normalizacao"))==normalizacao and
                str(r.get("metodo"))==metodo and
                pd.notna(r.get("n")) and int(r["n"])==n
            )]
            resultados.append(linha)
            salvar_csv_atomico(pd.DataFrame(resultados),caminho_metricas)
            LOG.info("[%d/%d] RESULTADO SALVO em %s | tempo total da configuracao: %.2fs",indice,len(configuracoes),caminho_metricas,linha["tempo_total_s"])
            if modelo is not None:
                del modelo
    # Modo somente metricas: evita classificar/gerar durante uma avaliacao parcial.
    # Nao altera o calculo de perplexidade e preserva os CSVs por configuracao.
    if "--apenas-metricas" in sys.argv:
        df=pd.DataFrame(resultados)
        ok=df[(df.get("status")=="ok") & (df.get("normalizacao")=="lower")]
        if not ok.empty:
            grafico_barras(ok,"perplexidade_teste","perplexidade_metodos.png",
                           "Perplexidade no teste interno por configuracao")
        LOG.info("TEMPO TOTAL DOS EXPERIMENTOS (somente metricas): %.2fs",time.perf_counter()-inicio_total)
        return
    df=pd.DataFrame(resultados)
    faltantes=[cfg for cfg in configuracoes if cfg not in {
        (str(r["normalizacao"]),int(r["n"]),str(r["metodo"]))
        for r in resultados if r.get("status")=="ok" and pd.notna(r.get("n"))
    }]
    if faltantes:
        LOG.warning("%d configuracoes ainda nao concluiram: %s",len(faltantes),faltantes)
    candidatos=df[(df.normalizacao=="lower")&(df.n==3)&(df.status=="ok")].dropna(subset=["perplexidade_val"])
    if candidatos.empty:
        LOG.error("Nenhum trigrama lower foi concluido; veja metricas.csv e execucao.log")
        return
    melhor=candidatos.sort_values("perplexidade_val").iloc[0]
    nome=melhor["metodo"]
    LOG.info("Selecionado pela validacao: %s. Recriando modelo para avaliacao final.",nome)
    if "lower" not in preparacoes:
        inicio=time.perf_counter()
        original={k:preparar(v,"lower") for k,v in bruto.items()}
        vocab=construir_vocab(original["treino"],MIN_FREQ)
        dados={k:mapear(v,vocab) for k,v in original.items()}
        preparacoes["lower"]=(original,vocab,dados)
        LOG.info("Preparo lower para classificacao/geracao: %.2fs",time.perf_counter()-inicio)
    original,vocab,dados=preparacoes["lower"]
    inicio=time.perf_counter()
    modelo=ModeloNgramas(nome,3,dados["treino"],vocab)
    LOG.info("Retreinamento do selecionado: %.2fs",time.perf_counter()-inicio)

    inicio_classificacao=time.perf_counter()
    # Avaliação empírica I: negativos por permutação de tokens das legendas.
    # Não é um detector geral de semântica: mede a distinção frente a corrupção sintática.
    rng=random.Random(SEED)
    positivos=[f for f in dados["teste"] if len(f)>=5][:400]
    pares=[]
    for i,frase in enumerate(positivos):
        bag=frase.copy()
        rng.shuffle(bag)
        if bag==frase: bag=bag[::-1]
        pares.append({"par_id":i,"rotulo":1,"frase":" ".join(frase),
                      "score":modelo.score_frase(frase)})
        pares.append({"par_id":i,"rotulo":0,"frase":" ".join(bag),
                      "score":modelo.score_frase(bag)})
    aval=pd.DataFrame(pares)
    if aval.empty: raise RuntimeError("Sem frases suficientes para avaliação empírica.")
    # Limiar determinado em VALIDAÇÃO, sem inspecionar scores do teste.
    val_frases=[f for f in dados["validacao"] if len(f)>=5][:400]
    scores_val=[]
    for f in val_frases:
        bag=f.copy();rng.shuffle(bag)
        scores_val += [(modelo.score_frase(f),1),(modelo.score_frase(bag),0)]
    if not scores_val: raise RuntimeError("Validação insuficiente para definir limiar.")
    limiares=np.quantile([s for s,_ in scores_val],np.linspace(0,1,101))
    limiar=max(limiares,key=lambda t:sum((s>=t)==bool(y) for s,y in scores_val))
    aval["predito"]=(aval.score>=limiar).astype(int)
    salvar_csv_atomico(aval,OUT/"classificacao_frases.csv")
    acuracia=float((aval.rotulo==aval.predito).mean())
    tp=int(((aval.rotulo==1)&(aval.predito==1)).sum())
    tn=int(((aval.rotulo==0)&(aval.predito==0)).sum())
    fp=int(((aval.rotulo==0)&(aval.predito==1)).sum())
    fn=int(((aval.rotulo==1)&(aval.predito==0)).sum())
    precisao=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precisao*recall/(precisao+recall) if precisao+recall else 0
    resumo={"modelo_selecionado":nome,"n":3,"normalizacao":"lower",
            "criterio_selecao":"menor perplexidade na validacao",
            "limiar_validacao":float(limiar),
            "acuracia_teste":acuracia,"precisao_teste":precisao,
            "recall_teste":recall,"f1_teste":f1,
            "matriz_confusao":{"tn":tn,"fp":fp,"fn":fn,"tp":tp},
            "n_pares":len(positivos)}
    salvar_json_atomico(resumo,OUT/"resumo.json")
    LOG.info("Classificacao: %.2fs; resumo e predicoes salvos",time.perf_counter()-inicio_classificacao)
    # A tarefa de frases embaralhadas e a de frases semanticamente absurdas
    # sao avaliadas separadamente para evitar inflar a acuracia principal.
    inicio_semantica=time.perf_counter()
    semantica=avaliar_semantica(modelo,vocab,bruto["validacao"],
                              bruto["teste"],limiar,OUT)
    salvar_json_atomico(semantica,OUT/"resumo_semantica.json")
    LOG.info("Subteste semantico: %.2fs; acuracia=%.4f",
             time.perf_counter()-inicio_semantica,semantica["acuracia"])
    inicio_geracao=time.perf_counter()

    # Avaliação empírica II: geração livre e condicionada por prefixos.
    prefixos=[[],["a","man"],["a","woman"],["people","are"]]
    # --prefixo "a dog" pode ser repetido; --interativo solicita um texto.
    if "--prefixo" in sys.argv:
        posicoes=[i for i,x in enumerate(sys.argv) if x=="--prefixo"]
        for pos in posicoes:
            if pos+1>=len(sys.argv) or sys.argv[pos+1].startswith("--"):
                raise ValueError('Informe o texto apos --prefixo, ex.: --prefixo "a dog"')
            prefixos.append(tokenizar(sys.argv[pos+1],"lower"))
    if "--interativo" in sys.argv:
        digitado=input("Digite um prefixo (Enter para geracao livre): ").strip()
        prefixos.append(tokenizar(digitado,"lower") if digitado else [])
    geracoes=[]
    for j,prefixo in enumerate(prefixos):
        for k in range(5):
            geracoes.append({"prefixo":" ".join(prefixo) or "(livre)",
                             "amostra":k+1,"texto":modelo.gerar(prefixo,max_tokens=20,
                                                                 seed=SEED+100*j+k)})
    salvar_csv_atomico(pd.DataFrame(geracoes),OUT/"geracoes.csv")
    LOG.info("Geracao: %.2fs; geracoes.csv salvo",time.perf_counter()-inicio_geracao)
    # Experimento de ablação do volume de treino, com as particoes externas
    # preservadas e subamostras por video_id.
    inicio_tamanho=time.perf_counter()
    comparar_tamanhos(bruto["treino"],bruto["validacao"],bruto["teste"],
                      "lower",nome,3,OUT,LOG)
    LOG.info("Efeito da quantidade de dados: %.2fs",time.perf_counter()-inicio_tamanho)
    inicio_graficos=time.perf_counter()

    # Gráficos dos resultados calculados (nunca valores fictícios).
    fatia=df[(df.normalizacao=="lower")&(df.n==3)].dropna(subset=["perplexidade_teste"])
    grafico_barras(fatia,"perplexidade_teste","perplexidade_metodos.png",
                   "Perplexidade no teste — trigramas")
    fatia=df[(df.normalizacao=="lower")].dropna(subset=["perplexidade_teste"])
    grafico_barras(fatia,"perplexidade_teste","ordem_ngramas.png",
                   "Efeito da ordem de n-gramas — cinco metodos")
    fatia=df[(df.n==3)&(df.metodo=="KneserNey")].dropna(subset=["perplexidade_teste"])
    grafico_barras(fatia,"perplexidade_teste","normalizacoes.png",
                   "Efeito da normalização — Kneser-Ney, trigramas")
    fig,ax=plt.subplots(figsize=(5,4))
    matriz=np.array([[tn,fp],[fn,tp]])
    img=ax.imshow(matriz,cmap="Blues")
    for (i,j),v in np.ndenumerate(matriz):
        ax.text(j,i,str(v),ha="center",va="center",color="black")
    ax.set_xticks([0,1],["Corrompida","Original"])
    ax.set_yticks([0,1],["Corrompida","Original"])
    ax.set_xlabel("Predição");ax.set_ylabel("Rótulo")
    ax.set_title("Matriz de confusão — frases")
    fig.tight_layout();fig.savefig(OUT/"matriz_confusao.png",dpi=170);plt.close(fig)
    LOG.info("Graficos: %.2fs",time.perf_counter()-inicio_graficos)
    LOG.info("TEMPO TOTAL DOS EXPERIMENTOS: %.2fs (%.2f min)",time.perf_counter()-inicio_total,(time.perf_counter()-inicio_total)/60)
    LOG.info("Resultados e graficos em: %s",OUT.resolve())

if __name__=="__main__":main()
