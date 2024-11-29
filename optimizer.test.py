import unittest
from bayes_opt import BayesianOptimization
from optimizer import objective, pbounds, print_best_params

class TestOptimizer(unittest.TestCase):
    
    def setUp(self):
        self.pbounds = pbounds.copy()
        print(self.pbounds)
        
        self.pbounds['timesteps'] = (1000, 2000) # Set a low range for timesteps
        self.pbounds['num_episodes'] = (1, 5) # Set a low range for num_episodes
        print(self.pbounds)
    
    def test_optimizer(self):
        
        # Initialize Bayesian Optimization with the objective function and parameter bounds
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=self.pbounds,
            random_state=1,
            verbose=2,
            allow_duplicate_points=True,
        )

        # Perform optimization with init_points=1 and n_iter=2
        optimizer.maximize(
            init_points=1,
            n_iter=2
        )
        
        # # Print the best result
        # print("Best parameters found:")
        # for param, value in optimizer.max['params'].items():
        #     print(f"{param}: {value}")
        
        
        # Sort and list the best parameters by performance score
        results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)
        top_n = 3  # Number of top configurations to list

        print_best_params(results, top_n)

        # Check if the optimizer has results
        self.assertTrue(len(optimizer.res) > 0, "Optimizer did not produce any results.")

        # Optionally, check if the best result has a valid target
        best_result = optimizer.max
        self.assertIn('target', best_result, "Best result does not contain a target.")
        self.assertIsInstance(best_result['target'], float, "Target is not a float.")

if __name__ == '__main__':
    unittest.main()