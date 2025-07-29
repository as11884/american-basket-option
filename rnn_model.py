
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from longstaff_schwartz import CorrelatedGBM

# 1. data generator using existing CorrelatedGBM
def generate_american_option_data(M=5000, N=126, d=5, S0=100.0, K=100.0, r=0.04, T=0.5, sig=0.40, corr=0.50):
    print("Generating data using CorrelatedGBM...")
    
    # Create same covariance matrix as Longstaff-Schwartz
    S0_array = np.full(d, S0)
    cov = np.full((d, d), corr * sig * sig)
    np.fill_diagonal(cov, sig * sig)
    
    # Use existing CorrelatedGBM class
    gbm_model = CorrelatedGBM(
        S0=S0_array, 
        r=r, 
        cov=cov, 
        T=T, 
        step=N, 
        N=M
    )
    
    # Generate paths: shape (M, N+1, d) 
    paths = gbm_model.generate_paths(seed=42)
    
    # Convert to torch tensors
    S = torch.from_numpy(paths).float()
    
    # data standardization
    S_mean = S.mean(dim=(0,1), keepdim=True)
    S_std = S.std(dim=(0,1), keepdim=True) + 1e-8
    S_normalized = (S - S_mean) / S_std
    
    # Arithmetic basket (same as Longstaff-Schwartz)
    weights = torch.full((d,), 1.0/d)  # Equal weights
    basket_prices = (S * weights).sum(dim=-1)  # Arithmetic average
    
    # American put payoff: max(K - basket_price, 0)
    payoffs = torch.maximum(torch.tensor(K) - basket_prices[:,1:], torch.tensor(0.0))
    payoff_T = torch.maximum(torch.tensor(K) - basket_prices[:,-1], torch.tensor(0.0))
    
    # Exercise indicators (1 if should exercise, 0 otherwise)
    delta_T = (basket_prices[:,1:] < K).float()
    
    times = torch.linspace(0, T, N+1).unsqueeze(0).repeat(M,1)
    discount = torch.exp(-r * (T - times[:,1:]))
    
    print("Data generation complete!")
    return S_normalized[:,1:], payoffs.unsqueeze(-1), discount, payoff_T, delta_T, S_mean, S_std

# 2. network structure
class PriceGRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        self.embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, h_prev):
        gru_out, h_next = self.gru(x, h_prev)
        gru_out_flat = gru_out.contiguous().view(-1, self.hidden_dim)
        e = self.embed(gru_out_flat)
        e = e.view(gru_out.shape[0], gru_out.shape[1], -1)
        return self.output(e), h_next

class DeltaGRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        self.embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, h_prev):
        gru_out, h_next = self.gru(x, h_prev)
        gru_out_flat = gru_out.contiguous().view(-1, self.hidden_dim)
        e = self.embed(gru_out_flat)
        e = e.view(gru_out.shape[0], gru_out.shape[1], -1)
        return self.output(e), h_next

class AmericanOptionRNN(nn.Module):
    def __init__(self, d, hidden_dim=32, alpha=0.1, beta=0.1):
        super().__init__()
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.price_net = PriceGRUNet(2*d, hidden_dim)
        self.delta_net = DeltaGRUNet(2*d, hidden_dim, d)
        
    def init_hidden(self, batch_size, device):
        h_price = torch.zeros(
            self.price_net.num_layers, 
            batch_size, 
            self.price_net.hidden_dim,
            device=device
        ).uniform_(-0.1, 0.1)
        
        h_delta = torch.zeros(
            self.delta_net.num_layers,
            batch_size,
            self.delta_net.hidden_dim,
            device=device
        ).uniform_(-0.1, 0.1)
        
        return h_price, h_delta

    def forward(self, S, g_S, times, payoff_T, delta_T):
        batch_size, seq_len = S.shape[:2]
        
        if g_S.dim() == 3 and g_S.shape[-1] == 1:
            g_S = g_S.squeeze(-1)
        g_S = g_S.unsqueeze(-1).expand(-1, -1, self.d)
        
        X = torch.cat([S, g_S], dim=-1)
        h_price, h_delta = self.init_hidden(batch_size, S.device)
        
        prices, deltas = [], []
        for t in reversed(range(seq_len)):
            price_t, h_price = self.price_net(X[:,t:t+1], h_price)
            delta_t, h_delta = self.delta_net(X[:,t:t+1], h_delta)
            
            prices.append(price_t.squeeze(1))
            deltas.append(delta_t.squeeze(1))
        
        prices = torch.stack(prices[::-1], dim=1)
        deltas = torch.stack(deltas[::-1], dim=1)
        return prices, deltas

# 3. Loss function
def bsde_loss(y_pred, y_true, delta_pred, S, rho, sigma, dt, r):
    price_term = torch.mean((y_true - y_pred)**2)
    delta_term = torch.mean((delta_pred**2).sum(dim=1))
    return price_term + 0.1 * delta_term

# 4. Training function
def train_model():
    print("Initializing model...")
    # Use same parameters as Longstaff-Schwartz
    d = 5
    S0 = 100.0
    K = 100.0
    r = 0.04
    T = 0.5
    N = int(252/2)  # 126 time steps
    sig = 0.40
    corr = 0.50
    
    model = AmericanOptionRNN(d=d)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    print("Loading data...")
    print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, sig={sig}, corr={corr}")
    print(f"Assets: {d}, Time steps: {N}")
    
    S, g_S, discount, payoff_T, delta_T, S_mean, S_std = generate_american_option_data(
        M=5000, N=N, d=d, S0=S0, K=K, r=r, T=T, sig=sig, corr=corr
    )
    dataset = TensorDataset(S, g_S, discount, payoff_T, delta_T)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    print("Starting training...")

    for epoch in range(100):
        model.train()
        total_loss = 0
        for S_batch, g_batch, disc_batch, payoff_batch, delta_batch in loader:
            y_pred, delta_pred = model(S_batch, g_batch, disc_batch, payoff_batch, delta_batch)
            
            y_true = (payoff_batch.unsqueeze(1) * disc_batch).unsqueeze(-1)
            
            rho = torch.eye(d) * (1-corr) + corr  # Use same correlation structure
            loss = bsde_loss(y_pred, y_true, delta_pred, S_batch, rho, sigma=sig, dt=T/N, r=r)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss/len(loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1:03d}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    print("Training complete!")
    return model

# 5. Test function
def test_model(model):
    print("\nTesting model...")
    test_S = torch.randn(3, 100, 5)
    test_g = torch.randn(3, 100, 1)
    test_disc = torch.randn(3, 100)
    test_payoff = torch.randn(3)
    test_delta = torch.randn(3, 100, 5)
    
    with torch.no_grad():
        y_pred, delta_pred = model(test_S, test_g, None, test_payoff, test_delta)
    
    print(f"y_pred shape: {y_pred.shape}")
    print(f"delta_pred shape: {delta_pred.shape}")
    
    assert y_pred.shape == (3, 100, 1), f"Expected [3,100,1], got {y_pred.shape}"
    assert delta_pred.shape == (3, 100, 5), f"Expected [3,100,5], got {delta_pred.shape}"
    print("Test passed!")


def visualize_predictions(model, loader, num_examples=3, S_mean=None, S_std=None):
    model.eval()
    with torch.no_grad():
        S, g, disc, payoff, delta = next(iter(loader))
        y_pred, delta_pred = model(S, g, disc, payoff, delta)
        
        time_steps = np.linspace(0, 1, S.shape[1])
        
        print("\n=== Prediction Examples ===")
        for i in range(min(num_examples, S.size(0))):
            plt.figure(figsize=(12, 6))
            
            true_price = disc[i].numpy() * payoff[i].item()
            
            pred_price = y_pred[i].squeeze().numpy()
            if S_mean is not None and S_std is not None:
                pred_price = pred_price * S_std[0][0][0].item() + S_mean[0][0][0].item()
                true_price = true_price * S_std[0][0][0].item() + S_mean[0][0][0].item()
            
            plt.plot(time_steps, true_price, 'g-', label='True Price (Discounted Payoff)')
            plt.plot(time_steps, pred_price, 'r--', label='Predicted Price')
            
            plt.scatter(time_steps[0], true_price[0], c='green', s=100, label='Initial True')
            plt.scatter(time_steps[0], pred_price[0], c='red', s=100, label='Initial Pred')
            plt.scatter(time_steps[-1], true_price[-1], c='blue', s=100, label='Final True')
            plt.scatter(time_steps[-1], pred_price[-1], c='orange', s=100, label='Final Pred')
            
            plt.title(f"Example {i+1} - Price Prediction vs True")
            plt.xlabel("Normalized Time")
            plt.ylabel("Option Price")
            plt.legend()
            plt.grid(True)
            plt.show()

            print(f"\nExample {i+1}:")
            print(f"Final True Payoff: {payoff[i].item():.4f}")
            print(f"Initial Predicted Price: {pred_price[0]:.4f}")
            print(f"Final Predicted Price: {pred_price[-1]:.4f}")
            
            print("\nDelta Predictions (first dimension):")
            print("True:", delta[i, [0, 49, 99], 0].numpy().round(4))
            print("Pred:", delta_pred[i, [0, 49, 99], 0].numpy().round(4))


if __name__ == "__main__":
    # Use exact same parameters as Longstaff-Schwartz
    print("RNN American PUT Option Pricing (matching Longstaff-Schwartz parameters)")
    print("=" * 72)
    
    # Parameters (matching longstaff_schwartz.py main)
    n_assets = 5
    S0 = 100.0
    K = 100.0
    r = 0.04
    T = 0.5
    N = int(252/2)  # 126 time steps
    sig = 0.40      # volatility
    corr = 0.50     # correlation
    
    print(f"S0={S0}, K={K}, T={T}, r={r}, sig={sig}, corr={corr}")
    print(f"Assets: {n_assets}, Time steps: {N}")
    print("=" * 72)
    
    # Train the model
    trained_model = train_model()
    
    # Test model functionality
    test_model(trained_model)
    
    # Generate data for visualization with same parameters
    S, g_S, discount, payoff_T, delta_T, S_mean, S_std = generate_american_option_data(
        M=1000, N=N, d=n_assets, S0=S0, K=K, r=r, T=T, sig=sig, corr=corr
    )
    dataset = TensorDataset(S, g_S, discount, payoff_T, delta_T)
    loader = DataLoader(dataset, batch_size=100)

    # Visualize predictions
    visualize_predictions(trained_model, loader, num_examples=3, S_mean=S_mean, S_std=S_std)

    # Save the trained model
    torch.save(trained_model.state_dict(), "american_option_rnn.pth")
    print("Model saved to 'american_option_rnn.pth'")
    
    print("\n" + "=" * 72)
    print("RNN Training Complete - Model uses same parameters as Longstaff-Schwartz:")
    print(f"• {n_assets} assets, arithmetic basket American PUT")
    print(f"• S0={S0}, K={K}, T={T} years, r={r*100}%")
    print(f"• Volatility: {sig*100}%, Correlation: {corr*100}%")
    print(f"• Time steps: {N}")
    print("=" * 72)
