# Experimento independente de Shannon

Coloque `03_shannon_interativo.py` **na mesma pasta** que `lm_utils.py`,
`config.py` e a pasta `dados/` do projeto existente. Não precisa trocar ou
executar `02_experimentos.py`.

Com o ambiente Conda já ativo, abra o terminal em `LM_MSRVTT` e execute:

```bash
python 04_shannon_interativo.py
```

O script lê **somente** `dados/treino.json` e cria um trigrama
`BackoffAbsoluto`, escolhido por ter sido o melhor nos resultados anteriores.
Esse único treinamento leva alguns segundos e é feito uma vez por sessão.
Ele não executa as 11 configurações, a classificação, a avaliação do tamanho
do treino ou o script `02_experimentos.py`.

## Menu

- Digite `1` para gerar sem prefixo.
- Digite `2,3,5` para gerar com vários prefixos sugeridos.
- Digite `T` para testar todos os dez prefixos.
- Digite `P` para escrever um prefixo próprio.
- Digite `Q` para encerrar.

Cada prefixo produz cinco frases, com até 20 novos tokens. O programa fica
aberto para você fazer quantos testes quiser, sem reajustar o modelo entre
escolhas.

## Saídas independentes

Os arquivos são criados em:

```text
LM_MSRVTT/
  resultados_shannon/
    geracoes.csv
    configuracao.json
```

O CSV registra todas as gerações de várias rodadas, os prefixos e as sementes aleatórias utilizadas.


