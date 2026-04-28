# Profil des Stratégies dans le Plan 2D

## Conceptual Logic
L'analyse de la convergence en observant uniquement le "Duality Gap" ne donne qu'une vision unidimensionnelle de l'optimisation et cache la dynamique comportementale de l'algorithme. Dans les jeux à somme nulle (par exemple, le jeu de Matching Pennies), les algorithmes d'apprentissage présentent souvent une dynamique rotationnelle cyclique. Plutôt que de pointer directement vers l'équilibre de Nash, les stratégies "tournent" autour de celui-ci (formant des cycles limites), spirallent vers l'extérieur (divergence) ou vers l'intérieur (convergence lente).

Pour comprendre *physiquement* ce que fait l'algorithme et valider la théorie, il est fondamental de tracer la trajectoire des probabilités d'action au cours de l'entraînement. Pour un jeu symétrique $2 \times 2$, le joueur 1 n'a qu'un seul degré de liberté pertinent (puisque sa seconde action a pour probabilité $1 - x_1$). On peut donc visualiser l'état complet du système à l'itération $t$ comme un simple point de coordonnées $(x_1^t, y_1^t)$ dans un plan carré unitaire $[0, 1] \times [0, 1]$.

Cet exercice vise à construire une visualisation cartésienne du profil des stratégies, mettant en évidence le comportement géométrique de l'algorithme (trajectoire), son point d'initialisation, sa position finale, et l'équilibre théorique du jeu.

## API Reference Table

| Librairie / Outil | Méthode / Classe | Description et Usage |
| :--- | :--- | :--- |
| **Matplotlib** | ax.plot(x, y, 'o-', alpha, markersize) | Utilisé pour tracer la ligne continue de la trajectoire reliant les itérations successives (x1, y1). |
| **Matplotlib** | ax.scatter(x, y, color, marker, zorder) | Parfait pour superposer des marqueurs distinctifs uniques (comme une étoile pour le départ, une croix pour l'équilibre) sans tracer de lignes. |
| **Matplotlib** | ax.set_xlim(0, 1) et ax.set_ylim(0, 1) | Indispensable pour contraindre le repère visuel au domaine de définition valide des probabilités (le simplexe). |
| **NumPy** | data[::step_size] | Technique de "slicing" pour sous-échantillonner un tableau. Afficher 400 000 points sur un plot 2D saturera la mémoire de Matplotlib. Il faut impérativement extraire une fraction des points. |
| **Matplotlib** | ax.set_aspect('equal') | (Optionnel) Force l'axe X et l'axe Y à avoir la même échelle visuelle pour éviter que les cycles limites ne paraissent ovales au lieu de circulaires. |

## Logical Checklist

- [ ] **Extraction des variables** : Extraire les composantes de la première action à partir des tableaux history_x et history_y récupérés depuis le fichier numpy (ils peuvent être de dimension Nx2, il faut isoler l'indice 0).
- [ ] **Sous-échantillonnage temporel** : Définir un paramètre (ex. subsample_rate) pour ne garder qu'une itération sur N (par exemple, viser l'affichage de 1000 à 2000 points maximum pour fluidifier le rendu).
- [ ] **Tracé de la trajectoire globale** : Utiliser la méthode plot pour relier temporellement les états sous-échantillonnés de x1 et y1. Utiliser une légère transparence pour ne pas masquer d'autres éléments s'il y a de nombreux chevauchements.
- [ ] **Balises de repérage spatio-temporel** :
    - Ajouter un scatter au point initial $(x_1^0, y_1^0)$ avec une couleur/forme distincte.
    - Ajouter un scatter au dernier point de la trajectoire pour voir où l'algorithme s'est arrêté.
    - Ajouter un scatter à la coordonnée de l'équilibre de Nash théorique (généralement $(0.5, 0.5)$ pour les jeux non perturbés).
- [ ] **Esthétique du plan** : Nommer les axes ("Stratégie Joueur 1 (Action 1)", etc.), fixer les limites strictes à [0, 1] et activer la grille de lecture.

## Structural Outline

```python
# Dans ton notebook marimo ou un fichier de plotting, implémente une nouvelle fonction :

def plot_2d_strategy_profile(x_history, y_history):
    # 1. Extraction de la première dimension (probabilité de l'action 1)
    # Assure-toi de la forme des vecteurs dans ton test.npz
    # x1 = x_history[:, 0]
    # y1 = y_history[:, 0]
    
    # 2. Sous-échantillonnage de la trajectoire
    # n_points_cibles = 1000
    # step = max(1, len(x1) // n_points_cibles)
    # x1_visu = x1[::step]
    # y1_visu = y1[::step]
    
    # 3. Création de la figure carrée
    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.set_aspect('equal') # Optionnel mais recommandé
    
    # 4. Tracé du chemin parcouru
    # ax.plot(x1_visu, y1_visu, '-', alpha=0.5, linewidth=1, color='blue', label='Trajectoire OMWU')
    
    # 5. Ajout des marqueurs clés d'analyse
    # Départ
    # ax.scatter(x1[0], y1[0], color='green', marker='o', s=100, label='Départ (T=0)', zorder=5)
    
    # Arrivée (en s'assurant de prendre la vraie dernière itération)
    # ax.scatter(x1[-1], y1[-1], color='red', marker='s', s=100, label='Arrivée (T=max)', zorder=5)
    
    # Équilibre de Nash théorique (à ajuster si delta est utilisé)
    # ax.scatter(0.5, 0.5, color='black', marker='X', s=150, label='Nash Eq.', zorder=5)
    
    # 6. Finitions et contraintes du repère
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_xlabel('Probabilité Action 1 (Joueur X)')
    # ax.set_ylabel('Probabilité Action 1 (Joueur Y)')
    # ax.legend()
    # ax.grid(True)
    
    # return fig
```

# Distance $L^2$ des stratégies à l'équilibre de Nash
#todo/expérimentation 
Dans le graphe des distances, on remarque que les stratégies orbitent autour de l'équilibre de Nash, l'idée est alors de quantifier cette orbite d'une manière adéquate à partir de la distance $L^2$ entre les itérations $(x_1, y_1)$ et le Nash $(x_1^*, y_1^*)$ calculé théoriquement. 

## Brouillon Implémentation
1. Faire un vecteur contenant les $\| (x_1, y_1) - (x_1^*, y_1^*)\|_2$ 
2. Plotter ce vecteur en fonction du numéro de l'itération
3. Essayer d'y appliquer les mêmes métriques de chaos (variation totale, max last 10% iterates, max var last 10% iterates)
4. 

