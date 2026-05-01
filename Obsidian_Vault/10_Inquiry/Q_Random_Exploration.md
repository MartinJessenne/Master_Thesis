# Description
Étant donné $A_δ$ pour $δ \in [0,1]$, le but est de générer une matrice de perturbation aléatoire $δ$ afin de prendre $$\text{Omwu}(A_δ + δ)$$
Ceci pour étudier le comportement de Omwu au voisinage de la matrice pathologique $A_δ$.

# Hypothèse 
Une telle approche paraît inutile
- Deux matrices proches en norme peuvent pourtant avoir des Équilibres de Nash totalement éloignés. En perturbant $A_δ$ on perd ainsi théoriquement toutes ses propriétés (structure de l'équilibre de Nash, conditionnement de ce NE...) 
- De plus, on ne peut pas efficacement et visuellement caractériser la topologie du voisinage de $A_δ$ ainsi, en effet, il y a 4 composantes aléatoires de perturbation. 

# Tentative
- Afin de pouvoir tout de même caractériser visuellement le voisinage de $A_δ$ j'ai perturbé seulement la première ligne ($[\frac{1}{2} + δ + \epsilon_0, \frac{1}{2} + \epsilon_1]$), ce qui déjà paraît relativement bizarre, avec $(\epsilon_i)_{i \in {1,2}}$ généré uniformément sur $[-0.1, 0.1]$ ou quelque chose comme ça
- On pouvait ainsi plotter l'écart entre l'historique des duality gap (convergence last_iterate donc) entre $A_0$ et $A_δ + δ$, que j'ai calculé en norme $L^2$, on observe une singularité autour de 0 en effet, mais cela n'est pas très informatif, notamment par le choix de la norme $L^2$. [[Autre métrique voisinage aléatoire]]
- 

# 2ème Set up
Étant donné $A_δ$ pour $δ \in [0,1]$ et $\epsilon \in [0,1]$, le but est de générer une matrice de perturbation aléatoire $U \in [-1, 1]^{2 \times 2}$ afin de prendre $$\text{Omwu}(A_δ + \epsilon U)$$Ceci pour étudier le comportement de Omwu au voisinage de la matrice pathologique $A_δ$.
On peut ainsi générer des perturbations aléatoire avec $A_δ + \epsilon U \in B_\infty(A_δ, \epsilon)$ 
La présentation des résultats est aussi simplifié, on ne présente plus que $\epsilon$ sur l'axe des abscisses et la métrique de duality gap en ordonnée. 

# Étapes
1. backend sur rust
2. frontend sur rust
3. une fois la matrice des vecteurs de chaque explo plotter le tout avec les box & whiskers
4. analyser

ensuite distance L^2 à l'équilibre de Nash
1. extraire les stratégies : ffi interface pour obtenir une list of result
2. créer une fonction python pour accéder aux champs des strats et juste plot la distance L^2

Lemme 5 : 
1. normalement petite modification mineure, juste copier-coller la fonction, mais en changeant le calcul de A_lambda,gamma en la matrice du lemme 5, faire un ou deux plot trnql et montrer que c'est à peu près pareil sur les courbes paramétrées ou en random
2. Exposer les problèmes de NaN pour cette méthode

Tout rédiger

# Résultat


