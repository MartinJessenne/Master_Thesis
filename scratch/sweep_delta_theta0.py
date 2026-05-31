import numpy as np
import matplotlib.pyplot as plt

class OMWU:
    def __init__(self, x_init, y_init, A, eta):
        self.x = x_init.copy()
        self.y = y_init.copy()
        self.x_hat = self.x.copy()
        self.y_hat = self.y.copy()
        self.A = A
        self.eta = eta
        self.history_x = [self.x.copy()]
        self.history_y = [self.y.copy()]
        
        # Calculate initial gap
        val_max_y = np.max(self.x @ self.A) 
        val_min_x = np.min(self.A @ self.y)
        self.gaps = [val_max_y - val_min_x]

    def _compute_gradients(self, x, y):
        grad_x = self.A @ y
        grad_y = -self.A.T @ x 
        return grad_x, grad_y

    def step(self):
        grad_x, grad_y = self._compute_gradients(self.x, self.y)

        x_hat_unnorm = self.x_hat * np.exp(-self.eta * grad_x)
        self.x_hat = x_hat_unnorm / np.sum(x_hat_unnorm)

        y_hat_unnorm = self.y_hat * np.exp(-self.eta * grad_y)
        self.y_hat = y_hat_unnorm / np.sum(y_hat_unnorm)

        x_unnorm = self.x_hat * np.exp(-self.eta * grad_x)
        self.x = x_unnorm / np.sum(x_unnorm)

        y_unnorm = self.y_hat * np.exp(-self.eta * grad_y)
        self.y = y_unnorm / np.sum(y_unnorm)

        self.history_x.append(self.x.copy())
        self.history_y.append(self.y.copy())
        val_max_y = np.max(self.x @ self.A) 
        val_min_x = np.min(self.A @ self.y)
        self.gaps.append(val_max_y - val_min_x)

def run_simulation(delta, num_steps=100000, eta=0.1):
    # Center of circular sweep
    p_0 = 1 / (1 + delta)
    q_0 = 1 / (2 * (1 + delta))
    
    # Radius r = 1/2 * d_min
    r = 0.5 * (delta / (1 + delta))
    
    # theta = 0
    p_theta = p_0 + r
    q_theta = q_0
    
    # Construct canonical matrix elements
    S = 1.0
    a = 1.0 + S * (1.0 - p_theta - q_theta)
    b = 1.0 - q_theta * S
    c = 1.0 - p_theta * S
    d = 1.0
    
    matrix = np.array([[a, b], [c, d]], dtype=np.float64)
    
    # Affine normalization to [0, 1]
    m = np.min(matrix)
    M = np.max(matrix)
    if M - m > 1e-9:
        matrix = (matrix - m) / (M - m)
        
    x_init = np.array([0.5, 0.5])
    y_init = np.array([0.5, 0.5])
    
    optimizer = OMWU(x_init, y_init, matrix, eta)
    
    for _ in range(num_steps):
        optimizer.step()
        
    return np.array(optimizer.gaps)

def main():
    deltas = [0.1, 0.05, 0.01, 0.005]
    num_steps = 100000
    
    plt.figure(figsize=(10, 6), constrained_layout=True)
    
    steps = np.arange(num_steps + 1) + 1
    
    for delta in deltas:
        print(f"Running simulation for delta = {delta}...")
        gaps = run_simulation(delta, num_steps=num_steps)
        best_iterate = np.minimum.accumulate(gaps)
        
        plt.loglog(steps, best_iterate, label=rf"Best-Iterate ($\delta = {delta}$)")
        
    # Plot standard T^(-1/6) reference curve
    # We fit the constant C so that it starts near the initial gap
    C = 0.5
    theoretical_bound = C * (steps ** (-1/6))
    plt.loglog(steps, theoretical_bound, label=r"$T^{-1/6}$ Worst-Case Rate Reference", color='black', linestyle='--', alpha=0.8)
    
    # Plot T^(-1/2) reference curve as well to see if it converges at a faster rate
    plt.loglog(steps, 0.5 * (steps ** (-0.5)), label=r"$T^{-1/2}$ Rate Reference", color='gray', linestyle=':', alpha=0.8)
    
    plt.xlabel('Iteration Step $t$ (Log Scale)')
    plt.ylabel('Duality Gap (Log Scale)')
    plt.title(r'OMWU Best-Iterate Convergence for Various $\delta$ at $\theta = 0$', fontsize=14, weight='bold', pad=15)
    plt.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.savefig("images/OMWU_Best_Iterate_theta_0_deltas.png", dpi=300)
    print("Saved plot to images/OMWU_Best_Iterate_theta_0_deltas.png")
    
if __name__ == "__main__":
    main()
