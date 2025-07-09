# American and European Basket Option Pricing with Longstaff-Schwartz Method

This project provides a comprehensive framework for pricing American and European basket options using the Longstaff-Schwartz (LSM) method with Monte Carlo simulation. It supports multiple correlated assets, flexible basket structures, and customizable regression bases for the LSM algorithm. The code is modular and extensible, making it easy to add new pricing methods, such as neural network-based regression, for comparison and research purposes.

## Features
- **Correlated Geometric Brownian Motion (GBM) Simulation**: Simulate price paths for multiple assets with customizable covariance/correlation structures.
- **Basket Option Pricing**: Price both American and European basket options (put/call) using the LSM method.
- **Flexible Regression Bases**: Use polynomial or custom regression bases for the continuation value estimation in LSM.
- **Sensitivity Analysis**: Compute basket deltas (option price sensitivity to each asset).
- **Visualization**: Plot asset and basket price paths, option values, and sensitivity results.
- **Extensible Design**: Easily add new regression methods (e.g., neural networks) for research and benchmarking.

## File Structure
- `longstaff_schwartz.py`  
  Core implementation of the correlated GBM simulator, regression bases, and LSM option pricer.
- `basket_option_pricing_demo.ipynb`  
  Jupyter notebook demonstrating usage, analysis, and visualization for basket option pricing and sensitivity.

## Getting Started
1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/basket-american-option.git
   cd basket-american-option/src
   ```
2. **Install dependencies**
   - Python 3.8+
   - numpy
   - matplotlib
   - (Optional for future: pytorch, tensorflow, or keras for neural network regression)
   ```sh
   pip install numpy matplotlib
   ```
3. **Run the notebook**
   Open `basket_option_pricing_demo.ipynb` in Jupyter or VS Code and run the cells to reproduce the results and plots.

## Planned Extensions
- **Neural Network Regression for LSM**: Implement a neural network-based regression for the continuation value in the LSM algorithm, and compare its performance to polynomial regression.
- **Benchmarking and Analysis**: Add experiments to compare pricing accuracy, computational efficiency, and stability between polynomial and neural network regression methods.

## Example Usage
See the notebook for a full workflow, including:
- Simulating correlated asset paths
- Building block correlation/covariance matrices
- Pricing American and European basket options
- Calculating and visualizing basket deltas

## References
- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: a simple least-squares approach. The Review of Financial Studies, 14(1), 113-147.
- Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering. Springer.

## License
MIT License

---

*This project is designed for research and educational purposes. Contributions and suggestions are welcome!*
