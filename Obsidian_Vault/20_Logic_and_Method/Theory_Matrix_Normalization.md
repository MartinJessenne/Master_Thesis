### 3. Normalisation de l'Espace des Jeux

#### 3.1. Motivation Théorique

Les garanties théoriques de convergence et les bornes du Duality Gap pour l'algorithme Optimistic Multiplicative Weights Update (OMWU), telles que présentées dans la littérature, reposent sur l'hypothèse fondamentale que les vecteurs de perte (ou les matrices de gains) sont strictement bornés. En particulier, on suppose que pour toute matrice de jeu $A$, ses coefficients vérifient $A_{i,j} \in [0, 1]$.

La paramétrisation développée précédemment permet de contrôler avec précision la position de l'équilibre de Nash $(x^*, y^*)$. Cependant, pour un facteur d'échelle $S$ fixé arbitrairement, les coefficients générés $a, b, c$ et $d$ peuvent arbitrairement dépasser $1$ ou devenir négatifs (par exemple lorsque l'équilibre est poussé très près des frontières). Utiliser la matrice brute fausserait artificiellement l'amplitude des gradients perçus par l'algorithme et invaliderait la comparaison temporelle (le nombre d'itérations avant divergence/convergence) entre différentes topologies.

Il est donc impératif de contraindre la matrice dans l'intervalle $[0, 1]$ sans altérer la dynamique intrinsèque du jeu ni les coordonnées de son équilibre.

#### 3.2. Transformation Affine des Utilités

D'après le théorème d'utilité de von Neumann-Morgenstern, appliquer une transformation affine strictement positive aux gains d'un jeu à somme nulle modifie l'échelle des gains mais préserve intégralement l'ordre des préférences et, par conséquent, les équilibres de Nash.

Soit $A$ la matrice brute générée, avec $m = \min_{i,j}(A_{i,j})$ et $M = \max_{i,j}(A_{i,j})$.

Nous cherchons une transformation affine $f(x) = \alpha x + \beta$ (avec $\alpha > 0$) telle que la plus petite valeur de la matrice devienne $0$ et la plus grande devienne $1$. Cela revient à résoudre le système suivant :

  

$$\begin{cases} \alpha m + \beta = 0 \\ \alpha M + \beta = 1 \end{cases}$$

Résolution :

De la première équation, on isole $\beta$ :

  

$$\beta = -\alpha m$$

On substitue cette expression dans la seconde équation :

  

$$\alpha M - \alpha m = 1 \implies \alpha(M - m) = 1 \implies \alpha = \frac{1}{M - m}$$

(Note : On suppose $M > m$, ce qui est toujours le cas pour un jeu non trivial).

En remplaçant $\alpha$ dans l'expression de $\beta$, on obtient :

  

$$\beta = -\frac{m}{M - m}$$

Fonction finale :

La transformation à appliquer à chaque coefficient $x$ de la matrice s'écrit donc :

  

$$f(x) = \frac{x}{M - m} - \frac{m}{M - m} = \frac{x - m}{M - m}$$

La nouvelle matrice normalisée, notée $A_{norm}$, dont les composantes sont définies par $(A_{norm})_{i,j} = f(A_{i,j})$, possède des valeurs strictement comprises dans $[0, 1]$.

#### 3.3. Invariance de l'àÆ’‚¬°quilibre de Nash

L'équilibre de Nash est calculé via des conditions d'indifférence reposant sur l'espérance mathématique, qui est un opérateur linéaire.

Si l'on reprend l'équation d'indifférence pour le joueur ligne avec la matrice transformée :

  

$$f(a)q + f(b)(1-q) = f(c)q + f(d)(1-q)$$

En développant avec $f(x) = \alpha x + \beta$ :

  

$$(\alpha a + \beta)q + (\alpha b + \beta)(1-q) = (\alpha c + \beta)q + (\alpha d + \beta)(1-q)$$

En factorisant $\alpha$ et en isolant $\beta$ :

  

$$\alpha \big( aq + b(1-q) \big) + \beta(q + 1 - q) = \alpha \big( cq + d(1-q) \big) + \beta(q + 1 - q)$$

Puisque $q + (1-q) = 1$, les termes en $\beta$ s'annulent des deux côtés :

  

$$\alpha \big( aq + b(1-q) \big) = \alpha \big( cq + d(1-q) \big)$$

Puisque $\alpha > 0$, on peut diviser par $\alpha$ pour retrouver exactement l'équation d'indifférence de la matrice d'origine :

  

$$aq + b(1-q) = cq + d(1-q)$$

Conclusion : La matrice normalisée $A_{norm}$ satisfait les contraintes de l'algorithme OMWU tout en garantissant de cibler très exactement l'équilibre $(P(\gamma), Q(\lambda))$ paramétré lors de la génération.

