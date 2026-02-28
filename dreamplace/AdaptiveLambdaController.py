##
# @file   AdaptiveLambdaController.py
# @author BeyondPPA/MPC Integration
# @brief  Sensitivity-based adaptive weight tuning for the five structural
#         cost features: wirelength, density balance, IO keepout, grid
#         alignment, and notch avoidance.
#         Based on Section III-B.5 of BeyondPPA (2025 MLCAD).

import torch
import logging

logger = logging.getLogger(__name__)


class AdaptiveLambdaController(object):
    """
    @brief Dynamically adjusts the weight vector lambda_1-5 during MPC
    refinement using sensitivity-based heuristics.
    Features that improve less receive higher weight, shifting focus
    toward under-optimized objectives.

    Update rule (every K iterations):
        delta_i = J_i^{prev} - J_i^{current}   (improvement per feature)
        sensitivity_i = 1 / (delta_i + epsilon)  (inverse improvement)
        lambda_i <- lambda_i + eta * sensitivity_i  (shift weight)
        normalize lambdas to maintain consistent scale
    """

    def __init__(self, num_features=5, eta=0.05, epsilon=1e-6,
                 lambda_min=0.1, lambda_max=5.0):
        """
        @brief initialization
        @param num_features number of cost features (default 5)
        @param eta learning rate for weight updates
        @param epsilon small constant to avoid division by zero
        @param lambda_min minimum allowed weight
        @param lambda_max maximum allowed weight
        """
        self.num_features = num_features
        self.eta = eta
        self.epsilon = epsilon
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.prev_costs = None

    def update(self, lambdas, current_costs):
        """
        @brief update lambda weights based on per-feature cost improvement
        @param lambdas current weight tensor [num_features]
        @param current_costs tensor of current per-feature costs [num_features]
        @return updated lambda tensor
        """
        if self.prev_costs is None:
            self.prev_costs = current_costs.clone()
            return lambdas

        # Per-feature improvement (positive = improved)
        delta = self.prev_costs - current_costs

        # Sensitivity: features that improved less get higher sensitivity
        sensitivity = 1.0 / (delta.abs() + self.epsilon)

        # Normalize sensitivity to sum to num_features
        sensitivity = sensitivity / (sensitivity.sum() + self.epsilon) * self.num_features

        # Update weights
        new_lambdas = lambdas + self.eta * sensitivity

        # Clamp to valid range
        new_lambdas = new_lambdas.clamp(min=self.lambda_min, max=self.lambda_max)

        # Re-normalize to maintain consistent total scale
        new_lambdas = new_lambdas / (new_lambdas.sum() + self.epsilon) * self.num_features

        # Store current costs for next update
        self.prev_costs = current_costs.clone()

        logger.info("Lambda update: [%s] -> [%s]" % (
            ", ".join(["%.3f" % l for l in lambdas]),
            ", ".join(["%.3f" % l for l in new_lambdas])
        ))

        return new_lambdas

    def reset(self):
        """
        @brief reset controller state
        """
        self.prev_costs = None
