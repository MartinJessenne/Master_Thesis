---
type: Logic
status: Open
related_pillar: "[[Thesis_Master_Plan]]"
tags:
  - thesis
created: 2026-05-15 11:04
---
# Description

> [!info] Quick Summary
> What is this note about?

# Architecture Design
## Qu'est-ce qu'on veut ? 
On veut pouvoir voir plus clairement le comportement de omwu sur A_delta en perturbant le tout. Le problème, c'est que c'est cool pour une valeur d'epsilon actuellement, on comprend qu'on fait des explorations aléatoire dans la boule et on ne se pose pas plus de questions. 
Mais tout s'écroule lorsqu'on cherche à comparer des explorations pour différentes valeurs d'epsilon étant donné qu'un jeu d'exploration va être inclus dans l'autre, ce qui pourrait "propager" un comportement chaotique à toutes les autres itérations pour des epsilon plus grands. 

Dès lors, il faut donc absolument séparer les itérations et adapter la méthodes pour compartimenter les expériences. 
Pour ce faire pour l'instant je distingue deux choix possibles : 
### 1. Forcer la génération dans des tranches concentriques 
C'est à dire : on itère sur le vecteur des valeurs de \epsilon : $(\epsilon_i = i/n)_{i \in [1,n]}$, pour l'itération 1 on tire les coefficient de U de manière uniforme dans : 
$[0, \epsilon_1]$, 
pour l'itération i, on tire dans : 
$[\epsilon_{i-1}, \epsilon_i]$
et on ajoute un signe aléatoire pour s'assurer qu'on est bien dans la boule unité et pas juste dans une sous partie positive. 

**Avantage** : 
On a bien l'idée initiale, explorer progressivement des portions concentriques de la boule unité, et ce avec le même nombre de tirage.  
**Inconvénient**: 
1. #Ask Il faut réussir à faire en sorte que chaque thread puisse générer ses nombres et son signe aléatoire. 
2. #Ask Il faut que chaque thread puisse accéder à la fois à $\epsilon_i$ (ce qui est déjà le cas), mais aussi à $\epsilon_{i-1}$ ce qui est plus compliqué : peut-être avec .window()

Input :
	num_slices: int // le nombre de sous intervalles explorés dans [0,1]
Output: 
	mat_results: $$ \begin{pmatrix}
\text{slice}_i \text{\_metric\_run}_1 & \text{slice}_i\text{run\_2}...  \\
\text{slice}_{i+1} \text{\_metric\_run}_1 & \text{run\_2}...  \\
\end{pmatrix}$$

le plot ressemblerait globalement à ce à quoi il ressemble maintenant dans l'idée, pas trop de modification à ce niveau là. 

--
### 2. Générer aléatoirement dans tout $B_\infty(0, 1)$ 
Dans ce cas, on change tout car les valeurs d'epsilon ne servent plus à rien. **en réalité dans les cas, on peut s'en passer, maintenant, à la manière d'un `linspace`, il suffit juste de préciser le découpage de l'intervalle $[0,1]$** 
Mais ce qu'on fait, c'est que pour 10_000 explorations aléatoires on calcule la norme de U (trois choix de normes implémentés).
ensuite on plot la métrique obtenue pour le duality gap en fonction de la valeur de la norme de la perturbation. 

Voir à quoi ça ressemble. 
Input : 
	num_exploration: int

Output : 
	Choix 1:
		vec_metrics: Array1
		vec_norms : Array1
	contrainte importantes: les indices doivent coïncider pour exploiter le résultat. 
	Choix 2 : 
		array_result: Array2
		$$ \begin{pmatrix}
\text{norm\_run}_1&\text{norm\_run}_2... \\
\text{metric\_run}_1&\text{metric\_run}_2...  \\
\end{pmatrix}$$
et à quoi ressemblerait le plot? 
le truc c'est que ça risque de ressembler à une série temporelle très aléatoires avec des pics très marqués et peut-être moins facilement exploitable qu'une analyse lissée sur de nombreux runs comme c'est le cas dans le choix 1. 

Maintenant, est-ce que les deux méthodes nécessitent des implémentations totalement différentes? Dès le début ou bien des étapes sont factorisables? 

### Pseudo Code
```rust
ALGORITHM: random_exploration
INPUT:  bounds, num_explo, num_iterations_per_explo, method_type (enum: concentric, norm(norm_name)),  
OUTPUT: sum

sum ← 0
FOR i ← 1 TO n
    sum ← sum + i
END FOR
RETURN sum
```
# Questions bloquantes
- À quel point puis-je permettre à chaque rayon worker d'avoir son propre générateur aléatoire et de modifier la range de la loi uniforme depuis laquelle on sample à chaque itération? Et comment m'y prendre


# Whiteboard Implementation 
![[Excalidraw/Concentric_Exploration_Design]]
