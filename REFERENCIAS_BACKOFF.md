# Referências da implementação manual de backoff

## Fontes

1. Katz, S. M. (1987). Estimation of Probabilities from Sparse Data for
   the Language Model Component of a Speech Recognizer. IEEE Transactions
   on Acoustics, Speech, and Signal Processing, 35(3), 400-401.
   DOI: https://doi.org/10.1109/TASSP.1987.1165125
   PDF: https://www.cis.upenn.edu/~danroth/Teaching/CS598-05/Papers/Katz87.pdf

2. Chen, S. F.; Goodman, J. (1999). An Empirical Study of Smoothing
   Techniques for Language Modeling. Computer Speech & Language,
   13(4), 359-394. https://doi.org/10.1006/csla.1999.0128

3. NLTK Language Modeling API:
   https://www.nltk.org/api/nltk.lm.html

## Formulação usada no projeto

A implementação segue o princípio de backoff com desconto: reserva
probabilidade ao descontar contagens de continuações observadas
e redistribui essa massa somente aos eventos não observados, usando
uma distribuição de menor ordem devidamente renormalizada.

Para histórico h, contagem total C(h), número de tipos observados T(h),
desconto D=0,5 e modelo inferior P_low:

- Quando C(h,w)>0: P(w|h) = (C(h,w)-D)/C(h).
- Quando C(h,w)=0: P(w|h) = [D*T(h)/C(h)] *
  P_low(w) / [1 - soma_{u:C(h,u)>0} P_low(u)].
- Quando C(h)=0, usa-se diretamente P_low(w).
- A base de unigramas utiliza Laplace.

O termo no denominador do segundo caso restringe e renormaliza a
distribuição inferior ao conjunto de continuações não observadas.
Assim, as probabilidades sobre o vocabulário somam 1. A normalização
foi verificada em teste de contexto observado e não observado.

**Precisão bibliográfica:** isso é uma variante de backoff com desconto
absoluto fixo e renormalização. Não é reprodução integral de Katz (1987),
que usa desconto associado a Good--Turing. Os trabalhos citados
fundamentam o princípio de recuo e redistribuição de massa; o valor
D=0,5 é uma escolha experimental deste projeto.

O Stupid Backoff é implementado pela NLTK e avaliado em escore separado,
pois não forma distribuições de probabilidade normalizadas.
