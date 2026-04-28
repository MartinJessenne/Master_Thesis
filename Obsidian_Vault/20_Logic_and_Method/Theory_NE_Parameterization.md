# Description
Le but ici est d'étudier la dynamique de convergence d'OMWU selon l'équilibre de Nash vers lequel il est censé converger. 

# Résultats Attendus
Nous espérons pouvoir mettre en lumière : 
- une condition sur les NE (trop proche de la frontière du simplexe)
- un seuil (vitesse d'approche à ce bord du simplexe ~à préciser)
à partir duquel s'exhibe le comportement pathologique de OMWU. 

# Todo

- [[Profil des stratégies dans le plan 2D]]
- [[Comparaison Last iterate, Random Iterate et Best Iterate]]
- [[Comparaison baseline théorique]]
- [[Jeu à équilibre arbitraire du papier]]

# Méthode
## Développement du calcul
àÆ’‚¬°tant donné un jeu $$x^TAy$$ avec la matrice de jeu $2 \times 2$  $$A =  \begin{pmatrix}
 a&b  \\
 c&d  \\
\end{pmatrix}$$ les équilibres de Nash $x^*$ et $y^*$ vérifient les conditions d'indifférence suivantes : 
- indifférence de $y^*$ face à la stratégie de $x^*$ : $$(x^*)^T A \begin{pmatrix}
 1\\
0\end{pmatrix} = (x^*)^T A \begin{pmatrix}
 0\\
1\end{pmatrix}$$
- indifférence de $x^*$ à la stratégie de $y^*$: $$\begin{pmatrix} 1 & 0\end{pmatrix} A y^* =\begin{pmatrix} 0 & 1\end{pmatrix} A y^* $$
Ainsi en notant $x^*_0 = p$ et $y^*_0 = q$ on trouve les relations : 
$$\left\{\begin{matrix}
 ap + c(1-p) = bp + d(1-p)
 \\
 aq + b(1-q) = cq + d(1-q)
\end{matrix}\right.$$
soit 
$$\left\{\begin{matrix}
 p = \frac{d-c}{a-b-c+d}
 \\
 q = \frac{d-b}{a-b-c+d}
 \end{matrix}\right.$$ On a donc une expression explicite de l'équilibre de Nash en fonction des coefficients de la matrice de jeu. 
 L'objectif est maintenant de créer des jeux paramétrés dont l'équilibre de Nash se rapproche plus ou moins vite d'un point d'intérêt sur le simplexe. 

## Expression des jeux paramétrés
Soit donc $\lambda, \gamma \in [0,1]$ des paramètres d'intérêts. 
Le but est de proposer une série de jeux $A_{\lambda, \gamma}$ dont les équilibres de Nash valent 
$$\begin{split} 
x^* &= \begin{pmatrix} p(\lambda) & 1-p(\lambda)\end{pmatrix} \\
y^* &= \begin{pmatrix} q(\gamma) & 1-q(\gamma) \end{pmatrix}
\end{split}$$
pour $p, q$ des fonctions arbitraires. 

Pour ce faire, on pose 
$$\begin{cases}
S &= a - b - c + d \\
p(\lambda) &= \frac{d-c}{S} \\
q(\gamma) &= \frac{d-b}{S}
\end{cases}$$
Il nous reste un degré de liberté, pour s'en débarrasser, on pose arbitrairement $d = 1$ et on trouve alors les relations pour chaque coefficient : 
$$\begin{cases}
a &= 1 + S(1 - p(\lambda) - q(\gamma))\\
b &= 1- S \times q(\gamma) \\
c &= 1- S\times p(\lambda)\\
d &= 1 \\
\end{cases}$$ 

Détail d'implémentation : les algorithmes OMWU OFTRL requièrent une [[normalisation de la matrice de jeu]], (cf. [[ablation study normalisation]])

# Implémentation
[[Problèmes d'overflow]]


