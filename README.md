# MSR-VTT — Modelos de linguagem

## Implementações nesta versão

- Laplace: NLTK Laplace.
- Witten--Bell: NLTK WittenBellInterpolated.
- Kneser--Ney: NLTK KneserNeyInterpolated, desconto 0,1.
- Interpolação: NLTK AbsoluteDiscountingInterpolated, desconto 0,75.
- Backoff absoluto: implementação manual normalizada, desconto 0,5.
- Stupid Backoff: NLTK StupidBackoff, alpha 0,4.

Há seis métodos com bigramas e trigramas, além de Kneser--Ney trigrama
sem pontuação: **13 configurações**.

A implementação manual de backoff é a única exceção ao uso das classes
prontas da NLTK. Sua fórmula e as fontes bibliográficas estão descritas
no arquivo REFERENCIAS_BACKOFF.md. Ela não é uma reprodução integral
do Katz backoff com Good--Turing.

O Stupid Backoff não retorna distribuição normalizada. Seus resultados
são salvos nas colunas pseudo_perplexidade_val e
pseudo_perplexidade_teste, e não são comparados com perplexidade nem
usados para selecionar o melhor modelo.

## Instalação

**Atualize os scripts dentro da sua pasta LM_MSRVTT existente,
preservando dados/ (o particionamento que você já usou).**
O ZIP desta revisão contém só código e documentação, não substitui
os JSONs do seu computador. Abrir o terminal na pasta LM_MSRVTT:

    conda env create -f environment.yml
    conda activate lm-msrvtt

Para executar os experimentos completos do zero:

    python 01_preparar_dados.py
    python 02_experimentos.py 

O comando --reiniciar substitui as métricas antigas. Nesta entrega,
os resultados antigos não são empacotados. Para retomar uma execução
interrompida, executar 02_experimentos.py sem --reiniciar.

Se precisar obter o protocolo 1K-A novamente, o script
01_preparar_dados.py baixa as anotações textuais e separa os 9.000
vídeos em treino/validação, reservando os 1.000 vídeos de teste.
Não execute a preparação novamente se os JSONs atuais já estiverem
corretos e você deseja preservar exatamente o mesmo particionamento.

## Saídas

- resultados/metricas.csv: PPL, pseudo-PPL, singletons, OOV e tempos.
- resultados/resumo.json: classificação de originais versus permutadas.
- resultados/resumo_semantica.json: frases absurdas (teste separado).
- resultados/efeito_tamanho_treino.csv: frações 25%, 50%, 75% e 100%.
- resultados/geracoes.csv: amostras de Shannon integradas.
- resultados/execucao.log: progresso e registro de tempo.



## Shannon independente

O script 03_shannon_interativo.py abre um menu com dez prefixos,
escolha T para todos e P para um prefixo próprio. Salva os resultados
em resultados_shannon/geracoes.csv, acrescentando novas rodadas.
