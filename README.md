# exploration_in_foraging
This project will study if a model that makes stochastic choices can provide a better account of how humans and other animals make stay-or-leave decisions. 

### Explanation:
1. **Initialization**:
    - `beta_values` and `intercept_values` are only set if the model is not `mellowmax`.
    - `omega_values` is only set if the model is `mellowmax`.

2. **Simulation**:
    - The `simulate` method selects actions based on the specified model (softmax, epsilon-greedy, or mellowmax).

3. **Results Preparation**:
    - For the `mellowmax` model, results are prepared with multiple `omega` values, similar to how `beta` and `intercept` values are handled for the other models.

4. **Plotting Results**:
    - Separate methods for plotting results are provided: one for `mellowmax` (`plot_mellowmax_results`) and another for `softmax` and `epsilon-greedy` (`plot_softmax_epsilon_results`).

5. **Main Function**:
    - The main function initializes the `Simulation` class with appropriate parameters and calls methods to prepare and plot results.

This code ensures the correct handling of different models with appropriate parameters and methods for simulations and result plotting.