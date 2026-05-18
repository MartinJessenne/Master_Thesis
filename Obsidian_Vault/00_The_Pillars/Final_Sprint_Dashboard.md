---
type: Pillar
status: Active
related_pillar: "[[Thesis_Master_Plan]]"
tags: [thesis, dashboard, final_sprint]
---
# 🚀 Final Sprint Dashboard (May 11 - May 21)

Ce dashboard retrace la route pour le rendu du dernier jet du mémoire prévu le **Jeudi 21 mai 2026 à 14h30**.

## 📅 Timeline (Gantt)

```mermaid
gantt
    title Master Thesis Final Sprint (May 13 - May 21)
    dateFormat  YYYY-MM-DD
    axisFormat  %d/%m
    
    section Implémentation (Code)
    Échantillonnage par couronnes       :done, a1, 2026-05-13, 2026-05-18
    Contrôle de densité & optimisation  :done, a2, 2026-05-14, 2026-05-18
    Total Variation via cumsum          :active, a3, 2026-05-18, 1d
    
    section Graphes & Visuels
    Production rapide des graphes       :active, b1, 2026-05-18, 1d
    
    section Rédaction (Typst)
    Justification matrice (2 DDL)       :active, c1, 2026-05-18, 2026-05-21
    Nuancer Section 4.1                 :c2, 2026-05-18, 1d
    Rédaction Section 5 (Optimisation)  :c3, 2026-05-19, 2d
    Rédaction Section 6 (Conclusion)    :c4, 2026-05-20, 2d
    
    Dernier jet pour le superviseur     :milestone, m1, 2026-05-21, 0d
```

## ✅ Todo List Subdivisée

### 1. Implémentation & Expériences (Code)
- [x] #task Mettre à jour l'échantillonnage de la section 4.1 : utiliser des couronnes concentriques (boule $L_\infty$) autour de $A_\delta$. ⏫ 📅 2026-05-14
- [x] #task Implémenter le contrôle de densité : s'assurer d'avoir un nombre homogène de matrices par couronne. ⏫ 📅 2026-05-14
- [ ] #task Optimiser l'évaluation temporelle avec la fonction `cumsum`. 🔼 📅 2026-05-18
- [ ] #task Tracer l'évolution temporelle de la *Total Variation* sur une trajectoire unique et l'agréger. ⏫ 📅 2026-05-18

### 2. Visualisation & Figures (Graphes)
- [ ] #task Remplacer le Boxplot (Figure 2) par un Bar plot / Min plot (affichant moyenne et variance). ⏫ 📅 2026-05-18
- [ ] #task Produire les graphes finaux : variance des 10 % dernières itérations et *Total Variation*. 🔼 📅 2026-05-18

### 3. Réflexion Théorique & Rédaction (`Master_Thesis.typ`)
- [ ] #task **(PRIORITÉ MAJEURE)** Justifier la matrice paramétrique personnelle et démontrer mathématiquement la réduction à 2 degrés de liberté. ⏫ 📅 2026-05-21
- [ ] #task Nuancer la conclusion de l'exploration aléatoire (Section 4.1) : limiter le chaos à une anomalie locale. ⏫ 📅 2026-05-18
- [ ] #task Rédiger la Section 5 (Optimisation) : intégrer le modèle et les expériences OMWU/OGDA. ⏫ 📅 2026-05-20
- [ ] #task Rédiger la Section 6 (Conclusion). 🔼 📅 2026-05-21

## 💡 Notes de l'appel du 11 mai 2026
* **Lemme 5 vs $A_{\lambda, \gamma}$** : Julien Grand-Clément préfère la paramétrisation du Lemme 5 ($A_{sep}$) car une matrice de jeu $2 \times 2$ n'a mathématiquement que 2 degrés de liberté (invariance par translation et multiplication scalaire positive). -> **Mise à jour 18 mai :** Modélisation et démonstration d'une matrice paramétrique propre pour retrouver ces 2 DDL.
* **Nuance Section 4.1** : Ne pas généraliser le chaos à l'ensemble du simplexe, ce phénomène reste concentré localement autour de la région de $A_\delta$.
