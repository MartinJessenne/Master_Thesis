---
type:
status: Open
related_pillar: "[[Thesis_Master_Plan]]"
tags:
  - thesis
created: 2026-05-18 11:14
---
> [!info] Quick Summary
> What is this note about?
> 
> Currently I've not thought that much into how to chose the parametric curves along which the $A_{\lambda, \gamma}$ matrix should approach the Nash Equilibrium. I've used arbitrary linear and quadratic approaches that aren't justified by any particular theory.
> 
> The aim of this note is to think of a better approach, justified from a theoretical point of view. 

# Ideas Dump
1. Leverage the previous experimental result of the 2D strategy visualization. 
Looking at the following 2D profile of the strategies. 
![[Parametric Curves Optimization.png]]
We notice that the strategies are orbiting around the final Nash Equilibrium, located at $x^*_1 = 1$ and $y^*_1 = 1/2$. We see that we are orbiting around this NE and that $y^*_1$ moves approximately between $[0.05, 0.85]$, and $x^*_1$ moves between $[0.45, 1.]$. Thus the coordinate that moves the farthest from the NE is the $y$ coordinate. 

# Solutions Candidates : 
Constat de base : lorsque le NE se situe à $[x^*_1, y^*_1] = [1, 1/2]$ 
on orbite assez loin. Le truc, ça va être de déplacer progressivement
ce NE pour finir par le faire atterrir à [1, 1/2] et remarquer ce qu'il se passe
On peut d'abord faire un mouvement linéaire, puis tenter d'autres trucs.
(Question : toujours à vitesse linéaire ou bien ça vaut le coup de changer les vitesses. 
Hypothèse : La vitesse de convergence ne change absolument rien imo, on raisonne point par point, l'algorithme cherche juste et rien qu'à converger vers $[x^*_1, y^*_1] = [1 -\lambda, 1/2 - \gamma]$ il se fout de savoir quelles étaient les valeurs à l'expérience d'avant!)

Le truc qu'il faut être capable de résoudre, c'est de capter quelle métrique on va utiliser, 
imo le profil 2D va être pas très beau, on va avoir une superposition de profil avec plein d'itération
en soit il faut le proposer, trouver une manière d'animer ça en python, où on voit l'évolution
du déplacement de l'équilibre de Nash et du profil qui cherche à l'atteindre. 

Une fois qu'on a codé ça, il faut ensuite une métrique pour pouvoir comparer ça et en tirer des conclusion
Idéalement la métrique devrait être capable de répondre à la question suivante : 
- étant donné une direction d'évolution du NE, voici la chaoticité de la convergence. 
2 choix : 
1. Rester dans l'espace des stratégies et utiliser l'autre fonction que l'on a développé : distance L2 au NE
Développer une métrique qui se base dessus pour réduire un set d'itération à un point et plotter le tout
2. Se placer dans l'espace du duality gap et utiliser les métriques que l'on a déjà déterminé 

# Familles de courbes paramétriques à tester : 
 Pour explorer le déplacement de l'équilibre de Nash dans le simplexe (avec régularisation type entropie/OMWU), voici des
  courbes pertinentes à tester :
   * Orbites Circulaires : c + R(cos θ, sin θ). Tester différents rayons R permet de voir comment la dynamique se comporte à
     mesure qu'on s'approche des frontières du simplexe.
   * Spirales Logarithmiques : r = ae^{bθ}. Ces courbes sont très naturelles pour les dynamiques de poids multiplicatifs
     (comme OMWU) car elles reflètent une croissance/décroissance exponentielle des probabilités.
   * Courbes de Lissajous : Pour une exploration plus complexe et périodique qui "balaie" le simplexe de manière non
     triviale.
   * Segments Radiaux : Des lignes droites partant du centre vers les sommets pour mesurer la stabilité de l'équilibre sous
     une dérive constante.
# Implementation Design
![[Parametric Curves Optimization 2026-05-18 13.32.09.excalidraw]]