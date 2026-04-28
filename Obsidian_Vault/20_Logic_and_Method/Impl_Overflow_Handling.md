---
tags:
  - bug
  - résolu
---
# Description

Bug lors de l'utilisation de OmwuOftrl depuis Python, mais il y avait aussi des problèmes dans Rust. Overflow (ou underflow peut-être) lors des itérations, malgré le log-sum-exp trick.
Empêche de mener à bien les expériences faisant intervenir OmwuOftrl. 

# Pistes
- Passer en fp128, mais peut être pas supporté sur windows
	- Alors peut-être exécuter le script depuis WSL, mais est-ce que ça change quelque-chose? #question
- Utiliser des librairies pour des précisions arbitraires.
	- #question #gemini quelles sont les crates ou autres solutions qui existent pour des précisions plus élevées? 
	- #question #gemini À quel point est-ce que cela va être compliqué à mettre en place avec ndarray et rust-numpy, et le passage à python? 

# Résolution 
Erreur dans mon code : 
 L'algorithme FTRL (Follow The Regularized Leader) avec régularisation entropique calcule la nouvelle stratégie en se basant sur le gradient cumulé et une distribution uniforme a priori : $x_{t, i} \propto \exp(-\eta L_{t, i})$. Actuellement, l'implémentation Rust multiplie le résultat de l'exponentielle du gradient cumulé par la stratégie précédente $x_{t-1}$. Cela signifie que tu accumules exponentiellement l'exponentielle des gradients ($x_t \propto x_{t-1} \exp(-\eta L_t)$) ! C'est ce qui fait exploser tes valeurs (underflow/overflow) et annule la somme, provoquant une division par zéro. La formule de mise à jour FTRL ne doit pas dépendre de state.x.

## Code Corrigé
```rust
impl OptimizerStrategy for OmwuOftrl {
    fn step(&mut self,state: &mut GameState) -> f64 {
        let (grad_x, grad_y) = state.compute_gradient();

        // update the cumulative gradient
        self.cumulative_grad_x = &self.cumulative_grad_x + &grad_x;
        self.cumulative_grad_y = &self.cumulative_grad_y + &grad_y;

        // Add the current gradient again, the optimism part 
        let step_x = -self.eta * (&self.cumulative_grad_x + &grad_x);
        let step_y = -self.eta * (&self.cumulative_grad_y + &grad_y);

        let max_step_x = step_x.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));
        let max_step_y = step_y.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));

        // update the strategy
        let mut x = step_x.map(|&s| f64::exp(s - max_step_x));
        let mut y = step_y.map(|&s| f64::exp(s - max_step_y));

        x /= x.sum();
        y /= y.sum();

        // Check if they lie on the Simplex 
        state.x = S::build(x).expect("x doesn't lie on the simplex");
        state.y = S::build(y).expect("y doesn't lie on the simplex");
        
        state.duality_gap(&grad_x, &grad_y)        
    }
}
```


