# Description
Plutôt que de plotter les $\| \text{DG}(A_0) - \text{DG}((A_δ + δ(\epsilon_0, \epsilon_1)))\|_{l^2}$ en fonction des $(\epsilon_0,\epsilon_1)$, Julien propose de plotter, étant donné $N$ le nombre d'itération max, $\| \text{DG}(A_δ + δ(\epsilon_0, \epsilon_1))_{i > 90\% * N}\|_{l^\infty}$   en fonction des $(\epsilon_0,\epsilon_1)$. 

# Résultat attendu
Peut-être intéressant, fort pouvoir discriminatif entre des runs qui convergent et d'autres qui ne convergent pas, mais peut-être trop discriminatif justement, ce qui ne permettra pas d'apprécier une vraie structure. Et à nouveau, en perturbant on n'a plus de garantie sur l'équilibre de Nash vers lequel doit tendre Omwu, donc peut être que l'algorithme adopte un comportement chaotique, mais on ne saura pas trop pourquoi

