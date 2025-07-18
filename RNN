
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 1. data generator
def generate_american_option_data(M=5000, N=100, d=5, S0=100, K=100, r=0.05, sigma=0.2):
    print("Generating data...")
    dt = 1.0 / N
    S = torch.zeros(M, N+1, d)
    S[:,0] = S0
    
    for t in range(1, N+1):
        dW = torch.randn(M, d) * np.sqrt(dt)
        S[:,t] = S[:,t-1] * torch.exp((r-0.5*sigma**2)*dt + sigma*dW)
    
    # data standardization
    S_mean = S.mean(dim=(0,1), keepdim=True)
    S_std = S.std(dim=(0,1), keepdim=True) + 1e-8
    S = (S - S_mean) / S_std
    
    adjusted_K = (K - S_mean) / S_std
    g = torch.max(S[:,1:] - adjusted_K, dim=-1).values
    payoff_T = torch.max(S[:,-1] - adjusted_K.squeeze(1), dim=-1).values
    delta_T = (S[:,1:] > adjusted_K).float()
    
    times = torch.linspace(0, 1, N+1).unsqueeze(0).repeat(M,1)
    discount = torch.exp(-r * (1 - times[:,1:]))
    
    print("Data generation complete!")
    return S[:,1:], g.unsqueeze(-1), discount, payoff_T, delta_T, S_mean, S_std

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
    d = 5
    model = AmericanOptionRNN(d=d)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    print("Loading data...")
    S, g_S, discount, payoff_T, delta_T, S_mean, S_std = generate_american_option_data(d=d)
    dataset = TensorDataset(S, g_S, discount, payoff_T, delta_T)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    print("Starting training...")

    for epoch in range(100):
        model.train()
        total_loss = 0
        for S_batch, g_batch, disc_batch, payoff_batch, delta_batch in loader:
            y_pred, delta_pred = model(S_batch, g_batch, disc_batch, payoff_batch, delta_batch)
            
            y_true = (payoff_batch.unsqueeze(1) * disc_batch).unsqueeze(-1)
            
            rho = torch.eye(d) * 0.9 + 0.1
            loss = bsde_loss(y_pred, y_true, delta_pred, S_batch, rho, sigma=0.2, dt=1/100, r=0.05)
            
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
    trained_model = train_model()
    test_model(trained_model)
    S, g_S, discount, payoff_T, delta_T, S_mean, S_std = generate_american_option_data(d=5)
    dataset = TensorDataset(S, g_S, discount, payoff_T, delta_T)
    loader = DataLoader(dataset, batch_size=100)

    visualize_predictions(trained_model, loader, num_examples=3, S_mean=S_mean, S_std=S_std)

    torch.save(trained_model.state_dict(), "american_option_rnn.pth")
    print("Model saved to 'american_option_rnn.pth'")
