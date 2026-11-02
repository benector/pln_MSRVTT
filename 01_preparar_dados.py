"""Baixa apenas as ANOTAÇÕES textuais do MSR-VTT, sem vídeos."""
import json
import random
import re
import time
from pathlib import Path
import requests
from config import TRAIN_URL, TEST_URL, SEED, VALIDATION_FRACTION, MAX_TRAIN_CAPTIONS

DATA = Path("dados")
DATA.mkdir(exist_ok=True)

def baixar(url, destino):
    """Mantém cache local e informa falhas de conexão com clareza."""
    if destino.exists():
        return
    resposta = requests.get(url, timeout=120)
    resposta.raise_for_status()
    destino.write_bytes(resposta.content)

def extrair(obj):
    """Lê objetos {video_id, caption} ou estruturas com lista de captions."""
    if isinstance(obj, dict):
        if "data" in obj: return extrair(obj["data"])
        if "annotations" in obj: return extrair(obj["annotations"])
        if "sentences" in obj: return extrair(obj["sentences"])
        if "video_id" in obj and ("caption" in obj or "captions" in obj):
            vid = str(obj["video_id"])
            valor = obj.get("caption", obj.get("captions"))
            if isinstance(valor, list):
                return [{"video_id":vid,"caption":str(v)} for v in valor if isinstance(v,str)]
            if isinstance(valor,str):
                return [{"video_id":vid,"caption":valor}]
        # Suporta mapeamento video_id -> captions.
        linhas=[]
        for chave,valor in obj.items():
            if isinstance(valor,list) and all(isinstance(x,str) for x in valor):
                linhas += [{"video_id":str(chave),"caption":x} for x in valor]
        return linhas
    if isinstance(obj,list):
        linhas=[]
        for item in obj: linhas += extrair(item)
        return linhas
    return []

def limpar(registros):
    """Remove vazios e duplicatas exatas dentro do mesmo vídeo."""
    vistos=set()
    saida=[]
    for r in registros:
        cap=" ".join(str(r["caption"]).split())
        vid=str(r["video_id"])
        chave=(vid,cap.casefold())
        if len(cap.split())<3 or chave in vistos: continue
        vistos.add(chave)
        saida.append({"video_id":vid,"caption":cap})
    return saida

def gravar(nome, linhas):
    (DATA/nome).write_text(json.dumps(linhas,ensure_ascii=False,indent=2),encoding="utf-8")

def main():
    inicio_total=time.perf_counter()
    arquivos=[("train_raw.json",TRAIN_URL),("test_raw.json",TEST_URL)]
    for nome,url in arquivos:
        inicio=time.perf_counter()
        print("Verificando",nome,flush=True)
        baixar(url,DATA/nome)
        print(f"Tempo download/cache {nome}: {time.perf_counter()-inicio:.2f}s",flush=True)
    treino=limpar(extrair(json.loads((DATA/"train_raw.json").read_text(encoding="utf-8"))))
    teste=limpar(extrair(json.loads((DATA/"test_raw.json").read_text(encoding="utf-8"))))
    if not treino or not teste: raise RuntimeError("Formato do JSON não reconhecido ou captions ausentes.")
    ids_treino={r["video_id"] for r in treino}
    ids_teste={r["video_id"] for r in teste}
    if ids_treino & ids_teste:
        raise RuntimeError("Vazamento: IDs de vídeos presentes em treino e teste.")
    rng=random.Random(SEED)
    ids=list(ids_treino);rng.shuffle(ids)
    qtd_val=max(1,round(len(ids)*VALIDATION_FRACTION))
    ids_val=set(ids[:qtd_val])
    validacao=[r for r in treino if r["video_id"] in ids_val]
    treino=[r for r in treino if r["video_id"] not in ids_val]
    if MAX_TRAIN_CAPTIONS:
        treino=treino[:MAX_TRAIN_CAPTIONS]
    # Os arquivos de teste não são usados para ajuste de parâmetros.
    gravar("treino.json",treino)
    gravar("validacao.json",validacao)
    gravar("teste.json",teste)
    for nome,linhas in (("treino",treino),("validacao",validacao),("teste",teste)):
        print(f"{nome}: {len(linhas)} descrições, {len(set(r['video_id'] for r in linhas))} vídeos")
    print("Dados preparados. Não foram baixados vídeos.")
    duracao=time.perf_counter()-inicio_total
    (DATA/"tempo_preparacao.json").write_text(json.dumps({"tempo_preparacao_total_s":round(duracao,3)},indent=2),encoding="utf-8")
    print(f"TEMPO TOTAL PREPARACAO: {duracao:.2f}s",flush=True)

if __name__=="__main__": main()
