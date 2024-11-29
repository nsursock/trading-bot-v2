import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Define the parameter space
space = [
    Real(1.0, 5.0, name='boost_factor'),
    Integer(1, 150, name='leverage_min'),
    Integer(1, 150, name='leverage_max'),
    Real(0.01, 0.1, name='risk_per_trade'),
    Real(0.1, 1.0, name='tp_mult_perc'),
    Real(0.1, 1.0, name='sl_mult_perc')
]

# Define the objective function
@use_named_args(space)
def objective(**params):
    # Here you would integrate your trading strategy or simulation
    # For demonstration, we'll use a dummy function
    # Replace this with your actual evaluation function
    boost_factor = params['boost_factor']
    leverage_min = params['leverage_min']
    leverage_max = params['leverage_max']
    risk_per_trade = params['risk_per_trade']
    tp_mult_perc = params['tp_mult_perc']
    sl_mult_perc = params['sl_mult_perc']
    
    # Dummy objective function (to be replaced with actual logic)
    return (boost_factor - 3.5)**2 + (leverage_min - 50)**2 + (leverage_max - 100)**2 + \
           (risk_per_trade - 0.05)**2 + (tp_mult_perc - 0.35)**2 + (sl_mult_perc - 0.2)**2

# Run Bayesian optimization
res = gp_minimize(objective, space, n_calls=50, random_state=0)

# Print the results
print("Best parameters found:")
print(f"Boost Factor: {res.x[0]}")
print(f"Leverage Min: {res.x[1]}")
print(f"Leverage Max: {res.x[2]}")
print(f"Risk Per Trade: {res.x[3]}")
print(f"TP Mult Perc: {res.x[4]}")
print(f"SL Mult Perc: {res.x[5]}")