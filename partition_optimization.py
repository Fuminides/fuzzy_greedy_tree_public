"""
Fuzzy Partition Optimization Module
====================================

This module provides methods to optimize fuzzy partitions for classification tasks.
It offers multiple optimization strategies:

1. Gradient-based optimization (using differentiable metrics)
2. Grid search over quantile configurations
3. Heuristic coordinate descent
4. Hybrid approaches

The optimization aims to improve class separability before tree induction.

Author: Javier Fumanal-Idocin
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings

# Import ex_fuzzy for partition construction
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import utils


@dataclass
class PartitionConfig:
    """Configuration for fuzzy partition parameters."""
    quantiles: List[float]  # e.g., [0, 20, 40, 60, 80, 100]
    n_partitions: int = 3  # Low, Medium, High
    
    def to_percentiles(self):
        """Convert to percentile format for numpy."""
        return [q for q in self.quantiles]


class FuzzyPartitionOptimizer:
    """
    Optimizes fuzzy partition parameters to maximize class separability.
    
    This optimizer works by adjusting the quantile positions that define
    trapezoidal membership functions, aiming to improve the quality of
    fuzzy partitions before tree induction.
    
    Parameters
    ----------
    optimization_method : str, default='separability'
        Method to use for optimization:
        - 'separability': Maximize separability index
        - 'gini': Minimize weighted Gini impurity
        - 'fisher': Maximize Fisher discriminant ratio
    
    search_strategy : str, default='grid'
        Search strategy to use:
        - 'grid': Grid search over quantile configurations
        - 'coordinate': Coordinate descent on quantile positions
        - 'gradient': Gradient-based optimization (if applicable)
        - 'hybrid': Combine grid search with local refinement
    
    verbose : bool, default=True
        Whether to print optimization progress
    """
    
    def __init__(
        self,
        optimization_method: str = 'separability',
        search_strategy: str = 'grid',
        verbose: bool = True
    ):
        self.optimization_method = optimization_method
        self.search_strategy = search_strategy
        self.verbose = verbose
        
        # Metric functions
        self._metric_functions = {
            'separability': self._compute_separability_index,
            'gini': self._compute_weighted_gini_loss,
            'fisher': self._compute_fisher_ratio
        }
        
        # Search functions
        self._search_functions = {
            'grid': self._grid_search,
            'coordinate': self._coordinate_descent,
            'gradient': self._gradient_descent,
            'hybrid': self._hybrid_search
        }
        
    def optimize_partitions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_partitions: Optional[np.ndarray] = None,
        max_iterations: int = 50,
        categorical_mask: Optional[np.ndarray] = None
    ) -> list:
        """
        Optimize fuzzy partitions for all features.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target labels
        initial_partitions : np.ndarray, optional
            Initial partition parameters as array of shape (n_features, 3, 4)
            where each (3, 4) slice contains trapezoid parameters [a,b,c,d]
            for [Low, Medium, High]. If None, uses default quantile-based partitions.
        max_iterations : int, default=50
            Maximum number of optimization iterations
        categorical_mask : array-like, optional
            Boolean mask indicating which features are categorical
            
        Returns
        -------
        optimized_partitions : list[fs.fuzzyVariable]
            List of fuzzy variables, one for each feature, in ex_fuzzy format.
            Compatible with FuzzyCART and other ex_fuzzy classifiers.
        """
        # Convert initial_partitions to quantiles for compatibility with existing code
        # or use default quantiles if not provided
        
        n_features = X.shape[1]
        optimized_partitions = {}
        
        if self.verbose:
            print(f"\nOptimizing fuzzy partitions using {self.optimization_method} "
                  f"metric with {self.search_strategy} search...")
            if initial_partitions is not None:
                print(f"Using provided initial partitions")
                partition_parameters = []
                for feature_idx in range(n_features):
                    params_feature = np.zeros((4 * len(initial_partitions[feature_idx]),))
                    for ix, logic_set in enumerate(initial_partitions[feature_idx]):
                        params_feature[ix*4:(ix+1)*4] = logic_set.membership_parameters

                    partition_parameters.append(params_feature)
            else:
                raise RuntimeError("Initial partitions must be provided for optimization.")
            

        
        # Optimize each feature independently
        for feature_idx in range(n_features):
            if self.verbose:
                print(f"\nOptimizing feature {feature_idx + 1}/{n_features}...")
            
            X_feature = X[:, feature_idx].reshape(-1, 1)
            
            initial_partition = initial_partitions[feature_idx]
            
            # Run optimization for this feature
            best_partition, best_score = self._optimize_single_feature_from_partition(
                X_feature, y, initial_partition, max_iterations
            )
            
            optimized_partitions[feature_idx] = best_partition

        
        if self.verbose:
            print(f"\n✓ Partition optimization complete!")
            print(f"Converting to ex_fuzzy format...")
        
        # Convert optimized dict to ex_fuzzy format using utils.construct_partitions
        # We'll use the optimized partition parameters to create custom fuzzy variables
        fuzzy_vars = []
        
        for feature_idx in range(n_features):
            partition_params = optimized_partitions[feature_idx]
            linguistic_terms = ['Low', 'Medium', 'High']
            
            # Create fuzzy sets for this feature using ex_fuzzy API
            # fs.FS(name, membership_parameters, linguistic_variable_domain)
            fuzzy_sets_list = []
            for term_idx, (term_name, params) in enumerate(zip(linguistic_terms, partition_params)):
                # params = [a, b, c, d] for trapezoid
                fuzzy_set = fs.FS(term_name, params, None)
                fuzzy_sets_list.append(fuzzy_set)
            
            # Create fuzzy variable with optimized sets
            fuzzy_var = fs.fuzzyVariable(
                f"Feature_{feature_idx}",
                fuzzy_sets_list
            )
            fuzzy_vars.append(fuzzy_var)
        
        if self.verbose:
            print(f"✓ Conversion complete! Returning {len(fuzzy_vars)} fuzzy variables")
        
        return fuzzy_vars
    
    def _optimize_single_feature(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        initial_quantiles: List[float],
        max_iterations: int
    ) -> Tuple[List[float], float]:
        """
        Optimize partition for a single feature (legacy method using quantiles).
        
        Returns
        -------
        best_quantiles : List[float]
            Optimized quantile positions
        best_score : float
            Best score achieved
        """
        search_func = self._search_functions[self.search_strategy]
        return search_func(X_feature, y, initial_quantiles, max_iterations)
    
    def _optimize_single_feature_from_partition(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        initial_partition: np.ndarray,
        max_iterations: int
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize partition for a single feature starting from partition parameters.
        
        Parameters
        ----------
        X_feature : np.ndarray
            Feature values
        y : np.ndarray
            Target labels
        initial_partition : np.ndarray of shape (3, 4)
            Initial trapezoid parameters [a,b,c,d] for [Low, Medium, High]
        max_iterations : int
            Maximum iterations
        
        Returns
        -------
        best_partition : np.ndarray
            Optimized partition parameters
        best_score : float
            Best score achieved
        """
        # Encode initial partition
        best_encoded = self._encode_partitions_direct(initial_partition)
        best_score = self._evaluate_partition_encoded(X_feature, y, best_encoded)
        
        # Run search strategy on encoded parameters
        if self.search_strategy == 'grid':
            best_encoded, best_score = self._grid_search_encoded(
                X_feature, y, best_encoded, best_score, max_iterations
            )
        else:
            # For other strategies, we could implement encoded versions
            # For now, just use the initial encoding (coordinate descent, gradient, hybrid not yet adapted to encoded space)
            if self.verbose:
                print(f"    Warning: {self.search_strategy} strategy not yet adapted to direct encoding. Using initial partition.")
        
        # Decode to get final partition
        best_partition = self._decode_partitions(best_encoded, X_feature)
        
        if self.verbose:
            initial_score = self._evaluate_partition_direct(X_feature, y, initial_partition)
            print(f"  Initial score: {initial_score:.4f}")
            print(f"  Optimized score: {best_score:.4f}")
        
        return best_partition, best_score
    
    def _encode_partitions_direct(
        self,
        partition_params: fs.fuzzyVariable
    ) -> np.ndarray:
        """
        Encode partition parameters directly (without going through quantiles).
        
        Parameters
        ----------
        partition_params : np.ndarray of shape (3, 4)
            Trapezoid parameters [a,b,c,d] for [Low, Medium, High]
        
        Returns
        -------
        encoded : np.ndarray
            Encoded incremental parameters (12 values)
        """
        total_parameters = np.zeros((4*len(partition_params),))
        for ix, linguistic_label in enumerate(partition_params):
            if ix == 0:
                total_parameters[0] = linguistic_label.membership_parameters[0]
                total_parameters[1] = linguistic_label.membership_parameters[1]
                total_parameters[2] = linguistic_label.membership_parameters[2]
                total_parameters[4] = linguistic_label.membership_parameters[3]
            elif ix == len(partition_params) - 1:
                total_parameters[-5] = linguistic_label.membership_parameters[0]
                total_parameters[-3] = linguistic_label.membership_parameters[1]
                total_parameters[-2] = linguistic_label.membership_parameters[2]
                total_parameters[-1] = linguistic_label.membership_parameters[3]
            else:
                total_parameters[ix*4 - 1] = linguistic_label.membership_parameters[0]
                total_parameters[ix*4 + 1] = linguistic_label.membership_parameters[1]
                total_parameters[ix*4 + 2] = linguistic_label.membership_parameters[2]
                total_parameters[ix*4 + 4] = linguistic_label.membership_parameters[3]
        
        flattened_partition_params = np.array(total_parameters).flatten()

        # Now, we need to make sure that the intertwines are respected.

        encoded = np.concatenate(([0.0], np.diff(flattened_partition_params)))
        
        return encoded
    
    def _grid_search_encoded(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        initial_encoded: np.ndarray,
        initial_score: float,
        max_iterations: int
    ) -> Tuple[np.ndarray, float]:
        """Grid search over encoded parameters."""
        best_encoded = initial_encoded.copy()
        best_score = initial_score
        
        feature_range = np.max(X_feature) - np.min(X_feature)
        increment_candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        
        evaluated = 0
        for inc_low_c in increment_candidates:
            for inc_low_d in increment_candidates:
                for inc_med_g in increment_candidates:
                    test_encoded = best_encoded.copy()
                    test_encoded[2] = inc_low_c * feature_range
                    test_encoded[4] = inc_low_d * feature_range
                    test_encoded[6] = inc_med_g * feature_range
                    test_encoded[1:] = np.maximum(test_encoded[1:], 0.0)
                    
                    score = self._evaluate_partition_encoded(X_feature, y, test_encoded)
                    evaluated += 1
                    
                    if score > best_score:
                        best_encoded = test_encoded
                        best_score = score
        
        if self.verbose:
            print(f"    Grid search: evaluated {evaluated} configurations")
        
        return best_encoded, best_score
    
    def _evaluate_partition_direct(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        partition_params: np.ndarray
    ) -> float:
        """Evaluate partition quality directly from trapezoid parameters."""
        metric_func = self._metric_functions[self.optimization_method]
        return metric_func(X_feature, y, partition_params)
    
    def _grid_search(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        initial_quantiles: List[float],
        max_iterations: int
    ) -> Tuple[List[float], float]:
        """
        Grid search over encoded partition parameters.
        
        Optimizes the interpretable encoding to ensure ordered partitions.
        """
        # Encode initial configuration
        best_encoded = self._encode_partitions_from_quantiles(X_feature, initial_quantiles)
        best_score = self._evaluate_partition_encoded(X_feature, y, best_encoded)
        
        # Define search space for encoded parameters (12 params in interleaved order)
        # Search ranges for normalized increments (as fractions of feature range)
        feature_range = np.max(X_feature) - np.min(X_feature)
        increment_candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        
        # Key parameters to search:
        # enc[2]: Low[c] - controls Low plateau start
        # enc[4]: Low[d] - controls Low end / overlap with Medium
        # enc[6]: Medium[g] - controls Medium plateau end
        # enc[8]: Medium[h] - controls Medium end / overlap with High
        # enc[10]: High[k] - controls High plateau end
        evaluated = 0
        
        # Grid search over critical increments
        for inc_low_c in increment_candidates:
            for inc_low_d in increment_candidates:
                for inc_med_g in increment_candidates:
                    # Construct encoded vector with modified increments
                    test_encoded = best_encoded.copy()
                    test_encoded[2] = inc_low_c * feature_range  # Low[c] increment
                    test_encoded[4] = inc_low_d * feature_range  # Low[d] increment
                    test_encoded[6] = inc_med_g * feature_range  # Medium[g] increment
                    
                    # Ensure non-negative increments
                    test_encoded[1:] = np.maximum(test_encoded[1:], 0.0)
                    
                    score = self._evaluate_partition_encoded(X_feature, y, test_encoded)
                    evaluated += 1
                    
                    if score > best_score:
                        best_encoded = test_encoded
                        best_score = score
        
        if self.verbose:
            print(f"    Grid search: evaluated {evaluated} encoded configurations")
        
        return best_encoded, best_score
    
    def _evaluate_partition_encoded(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        encoded: np.ndarray
    ) -> float:
        """
        Evaluate partition quality from encoded parameters.
        
        Parameters
        ----------
        X_feature : np.ndarray
            Feature values
        y : np.ndarray
            Target labels
        encoded : np.ndarray
            Encoded partition parameters
            
        Returns
        -------
        score : float
            Quality metric value
        """
        # Decode to get trapezoid parameters
        partition_params = self._decode_partitions(encoded, X_feature)
        
        # Evaluate using existing metric
        metric_func = self._metric_functions[self.optimization_method]
        score = metric_func(X_feature, y, partition_params)
        
        return score
    
    def _coordinate_descent(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        initial_quantiles: List[float],
        max_iterations: int
    ) -> Tuple[List[float], float]:
        """
        Coordinate descent: optimize one quantile at a time.
        """
        current_quantiles = initial_quantiles.copy()
        current_score = self._evaluate_partition(X_feature, y, current_quantiles)
        
        step_sizes = [5, 2, 1]  # Progressively smaller steps
        
        for iteration in range(max_iterations):
            improved = False
            
            for step_size in step_sizes:
                # Try to improve Q1 (index 1)
                for delta in [-step_size, step_size]:
                    new_q1 = current_quantiles[1] + delta
                    if 5 <= new_q1 < current_quantiles[2]:  # Valid range
                        test_quantiles = current_quantiles.copy()
                        test_quantiles[1] = new_q1
                        score = self._evaluate_partition(X_feature, y, test_quantiles)
                        
                        if score > current_score:
                            current_quantiles = test_quantiles
                            current_score = score
                            improved = True
                
                # Try to improve Q2 (index 2)
                for delta in [-step_size, step_size]:
                    new_q2 = current_quantiles[2] + delta
                    if current_quantiles[1] < new_q2 < current_quantiles[3]:
                        test_quantiles = current_quantiles.copy()
                        test_quantiles[2] = new_q2
                        score = self._evaluate_partition(X_feature, y, test_quantiles)
                        
                        if score > current_score:
                            current_quantiles = test_quantiles
                            current_score = score
                            improved = True
                
                # Try to improve Q3 (index 3)
                for delta in [-step_size, step_size]:
                    new_q3 = current_quantiles[3] + delta
                    if current_quantiles[2] < new_q3 <= 95:
                        test_quantiles = current_quantiles.copy()
                        test_quantiles[3] = new_q3
                        score = self._evaluate_partition(X_feature, y, test_quantiles)
                        
                        if score > current_score:
                            current_quantiles = test_quantiles
                            current_score = score
                            improved = True
                
                if improved:
                    break  # Found improvement with this step size
            
            if not improved:
                break  # Converged
        
        if self.verbose:
            print(f"    Coordinate descent: converged after {iteration + 1} iterations")
        
        return current_quantiles, current_score
    
    def _gradient_descent(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        initial_quantiles: List[float],
        max_iterations: int
    ) -> Tuple[List[float], float]:
        """
        Gradient-based optimization using numerical gradients.
        
        This approximates gradients using finite differences.
        """
        # Convert quantiles to continuous parameters (Q1, Q2, Q3)
        # We optimize in the space [0, 100] with constraints
        params = np.array([initial_quantiles[1], initial_quantiles[2], initial_quantiles[3]])
        
        learning_rate = 2.0
        epsilon = 0.5  # For numerical gradient
        
        best_params = params.copy()
        best_score = self._evaluate_partition_from_params(X_feature, y, params)
        
        for iteration in range(max_iterations):
            # Compute numerical gradient
            gradient = np.zeros(3)
            
            for i in range(3):
                # Forward difference
                params_plus = params.copy()
                params_plus[i] += epsilon
                
                # Check validity
                if self._is_valid_params(params_plus):
                    score_plus = self._evaluate_partition_from_params(X_feature, y, params_plus)
                else:
                    score_plus = -np.inf
                
                # Backward difference
                params_minus = params.copy()
                params_minus[i] -= epsilon
                
                if self._is_valid_params(params_minus):
                    score_minus = self._evaluate_partition_from_params(X_feature, y, params_minus)
                else:
                    score_minus = -np.inf
                
                # Central difference
                if score_plus != -np.inf and score_minus != -np.inf:
                    gradient[i] = (score_plus - score_minus) / (2 * epsilon)
                elif score_plus != -np.inf:
                    gradient[i] = (score_plus - best_score) / epsilon
                elif score_minus != -np.inf:
                    gradient[i] = (best_score - score_minus) / epsilon
                else:
                    gradient[i] = 0.0
            
            # Gradient ascent step (maximizing score)
            new_params = params + learning_rate * gradient
            
            # Project to valid space
            new_params = self._project_to_valid_params(new_params)
            
            # Evaluate new parameters
            new_score = self._evaluate_partition_from_params(X_feature, y, new_params)
            
            # Update if improved
            if new_score > best_score:
                best_score = new_score
                best_params = new_params
                params = new_params
            else:
                # Reduce learning rate
                learning_rate *= 0.5
                if learning_rate < 0.1:
                    break
        
        # Convert back to quantiles
        best_quantiles = [0, best_params[0], best_params[1], best_params[2], 100]
        
        if self.verbose:
            print(f"    Gradient descent: converged after {iteration + 1} iterations")
        
        return best_quantiles, best_score
    
    def _hybrid_search(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        initial_quantiles: List[float],
        max_iterations: int
    ) -> Tuple[List[float], float]:
        """
        Hybrid approach: coarse grid search followed by local refinement.
        """
        # Phase 1: Coarse grid search
        q1_candidates = [15, 20, 25]
        q2_candidates = [40, 50, 60]
        q3_candidates = [75, 80, 85]
        
        best_quantiles = initial_quantiles.copy()
        best_score = self._evaluate_partition(X_feature, y, best_quantiles)
        
        for q1 in q1_candidates:
            for q2 in q2_candidates:
                for q3 in q3_candidates:
                    if q1 < q2 < q3:
                        candidate_quantiles = [0, q1, q2, q3, 100]
                        score = self._evaluate_partition(X_feature, y, candidate_quantiles)
                        
                        if score > best_score:
                            best_score = score
                            best_quantiles = candidate_quantiles
        
        # Phase 2: Local coordinate descent refinement
        best_quantiles, best_score = self._coordinate_descent(
            X_feature, y, best_quantiles, max_iterations // 2
        )
        
        if self.verbose:
            print(f"    Hybrid search: grid + coordinate descent")
        
        return best_quantiles, best_score
    
    def _evaluate_partition(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        quantiles: List[float]
    ) -> float:
        """
        Evaluate partition quality using the selected metric.
        """
        # Convert quantiles to partition params
        partition_params = self._quantiles_to_partitions(X_feature, quantiles)
        metric_func = self._metric_functions[self.optimization_method]
        return metric_func(X_feature, y, partition_params)
    
    def _evaluate_partition_from_params(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        params: np.ndarray
    ) -> float:
        """Evaluate from continuous parameters [Q1, Q2, Q3]."""
        quantiles = [0, params[0], params[1], params[2], 100]
        return self._evaluate_partition(X_feature, y, quantiles)
    
    def _compute_separability_index(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        partition_params: np.ndarray
    ) -> float:
        """
        Compute separability index for fuzzy partition.
        
        SI = Σ_v Σ_c [Σ_i μ_v(x_i) * I(y_i = c)]² / Σ_i μ_v(x_i)
        
        Higher values indicate better class separation.
        """
        # partition_params is already (3, 4) array
        
        total_separability = 0.0
        n_classes = len(np.unique(y))
        
        # For each linguistic term (Low, Medium, High)
        for v in range(3):
            memberships = self._compute_memberships(X_feature, partition_params[v])
            
            # For each class
            for c in range(n_classes):
                class_mask = (y == c).astype(float)
                numerator = np.sum(memberships * class_mask) ** 2
                denominator = np.sum(memberships) + 1e-10
                
                total_separability += numerator / denominator
        
        return total_separability
    
    def _compute_weighted_gini_loss(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        partition_params: np.ndarray
    ) -> float:
        """
        Compute negative weighted Gini impurity (to maximize = minimize Gini).
        
        Lower Gini = better partition, so we return negative for maximization.
        """
        # partition_params is already a (3, 4) array
        
        total_gini = 0.0
        
        # For each linguistic term
        for v in range(3):
            memberships = self._compute_memberships(X_feature, partition_params[v])
            
            # Compute weighted Gini for this partition
            total_membership = np.sum(memberships)
            
            if total_membership < 1e-10:
                continue
            
            # Weighted class proportions
            unique_classes = np.unique(y)
            gini = 1.0
            
            for c in unique_classes:
                class_mask = (y == c).astype(float)
                weighted_count = np.sum(memberships * class_mask)
                proportion = weighted_count / total_membership
                gini -= proportion ** 2
            
            total_gini += gini
        
        # Return negative (we want to maximize = minimize Gini)
        return -total_gini
    
    def _compute_fisher_ratio(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        partition_params: np.ndarray
    ) -> float:
        """
        Compute Fisher discriminant ratio for the partition.
        
        Higher values indicate better class separability.
        """
        # partition_params is already a (3, 4) array
        
        total_fisher = 0.0
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        if n_classes < 2:
            return 0.0
        
        # For each linguistic term
        for v in range(3):
            memberships = self._compute_memberships(X_feature, partition_params[v])
            
            # Compute class means weighted by memberships
            class_means = []
            class_vars = []
            
            for c in unique_classes:
                class_mask = (y == c).astype(float)
                weighted_membership = memberships * class_mask
                total_weight = np.sum(weighted_membership)
                
                if total_weight > 1e-10:
                    weighted_mean = np.sum(weighted_membership * X_feature.flatten()) / total_weight
                    weighted_var = np.sum(weighted_membership * (X_feature.flatten() - weighted_mean) ** 2) / total_weight
                    class_means.append(weighted_mean)
                    class_vars.append(weighted_var)
            
            if len(class_means) >= 2:
                # Between-class variance
                overall_mean = np.mean(class_means)
                between_var = np.var(class_means)
                
                # Within-class variance
                within_var = np.mean(class_vars) if class_vars else 1e-10
                
                # Fisher ratio
                fisher = between_var / (within_var + 1e-10)
                total_fisher += fisher
        
        return total_fisher
    
    def _encode_partitions_from_quantiles(
        self,
        X_feature: np.ndarray,
        quantiles: List[float]
    ) -> np.ndarray:
        """
        Encode partition parameters using the CORRECT interleaved ordering.
        
        This encoding scheme ensures monotonicity by construction through interleaving
        parameters from adjacent trapezoids. The key insight is that we encode the
        TOUCHING POINTS between adjacent trapezoids together, ensuring they remain ordered.
        
        For 3 partitions, the encoding order is:
        1-2: Low[a,b] (first two params of Low)
        3: Low[c] (third param of Low)
        4: Medium[e] (first param of Medium) -- touches after Low[c]
        5: Low[d] (last param of Low) -- touches before Medium[f]
        6: Medium[f] (second param of Medium) -- touches after Low[d]
        7: Medium[g] (third param of Medium)
        8: High[i] (first param of High) -- touches after Medium[g]
        9: Medium[h] (last param of Medium) -- touches before High[j]
        10: High[j] (second param of High) -- touches after Medium[h]
        11-12: High[k,l] (last two params of High)
        
        GENERALIZATION TO N PARTITIONS:
        --------------------------------
        The pattern extends naturally to N linguistic terms by continuing the interleaving:
        
        For N trapezoids, we encode 4*N parameters in an interleaved order that mixes
        parameters from different trapezoids. The key principle:
        
        1. All encoded values are POSITIVE (or zero)
        2. During decoding, we ACCUMULATE these values sequentially
        3. The accumulated sequence is guaranteed to be monotonically increasing
        4. We extract parameters following the interleaved order
        5. Finally, we NORMALIZE to fit the feature domain [feature_min, feature_max]
        
        This approach ensures:
        - Each trapezoid is valid (a≤b≤c≤d) by construction
        - Trapezoids are interpretably ordered (centers increase)
        - Domain is fully covered
        - Optimization can work with unconstrained positive values
        
        Total parameters: 4*N for N linguistic terms (each trapezoid has 4 points)
        
        Parameters
        ----------
        X_feature : np.ndarray
            Feature values
        quantiles : List[float]
            Quantile positions [0, 20, 40, 60, 80, 100]
        
        Returns
        -------
        encoded : np.ndarray
            Encoded incremental parameters (10 values for 3 linguistic terms)
        """
        # Get trapezoid parameters directly (no quantiles involved!)
        partition_params = self._quantiles_to_partitions(X_feature, quantiles)
        
        low_a, low_b, low_c, low_d = partition_params[0]
        med_e, med_f, med_g, med_h = partition_params[1]
        high_i, high_j, high_k, high_l = partition_params[2]
        
        # Encode as positive increments in the interleaved order
        # Just map the parameters following your specification
        encoded = np.array([
            low_a,              # Position 1
            low_b - low_a,      # Position 2
            low_c - low_b,      # Position 3
            med_e - low_c,      # Position 4
            low_d - med_e,      # Position 5
            med_f - low_d,      # Position 6
            med_g - med_f,      # Position 7
            high_i - med_g,     # Position 8
            med_h - high_i,     # Position 9
            high_j - med_h,     # Position 10
            high_k - high_j,    # Position 11
            high_l - high_k     # Position 12
        ])
        
        return np.maximum(encoded, 0.0)  # Ensure positive
    
    def _decode_partitions(
        self,
        encoded: np.ndarray,
        X_feature: np.ndarray
    ) -> np.ndarray:
        """
        Decode incremental parameters back into trapezoid parameters.
        Uses the CORRECT interleaved ordering that ensures proper monotonicity.
        
        The encoding order for 3 partitions is:
        1-2: Low[a,b], 3: Low[c], 4: Medium[e], 5: Low[d],
        6: Medium[f], 7: Medium[g], 8: High[i], 9: Medium[h],
        10: High[j], 11-12: High[k,l]
        
        By accumulating positive increments and then normalizing to the feature domain,
        we guarantee:
        - All 12 accumulated values form a monotonically increasing sequence
        - Each trapezoid is valid (a≤b≤c≤d)
        - Trapezoids overlap appropriately
        - The partition covers the full feature range
        - Interpretability is maintained (Low < Medium < High by center ordering)
        
        This encoding allows optimization algorithms to work with unconstrained
        positive values while automatically ensuring valid, interpretable partitions
        
        Parameters
        ----------
        encoded : np.ndarray
            Encoded incremental parameters (12 values for 3 partitions)
        X_feature : np.ndarray
            Feature values (for domain bounds)
        
        Returns
        -------
        partition_params : np.ndarray of shape (3, 4)
            Trapezoid parameters [a, b, c, d] for [Low, Medium, High]
        """
        partition_params = np.zeros((len(encoded) // 4, 4))

        # Accumulate to get absolute positions
        accumulated = np.cumsum(encoded)

        # Map accumulated values back to trapezoid parameters
        for ix in range(len(partition_params)):
            if ix == 0:
                partition_params[ix, 0] = accumulated[0]         # a
                partition_params[ix, 1] = accumulated[1]         # b
                partition_params[ix, 2] = accumulated[2]         # c
                partition_params[ix, 3] = accumulated[4]         # d
            elif ix == len(partition_params) - 1:
                partition_params[ix, 0] = accumulated[-5]        # a
                partition_params[ix, 1] = accumulated[-3]        # b
                partition_params[ix, 2] = accumulated[-2]        # c
                partition_params[ix, 3] = accumulated[-1]        # d
            else:
                partition_params[ix, 0] = accumulated[ix*4 - 1]  # a
                partition_params[ix, 1] = accumulated[ix*4 + 1]  # b
                partition_params[ix, 2] = accumulated[ix*4 + 2]  # c
                partition_params[ix, 3] = accumulated[ix*4 + 4]  # d

        # Normalize from [0, accumulated.max()] to [feature_min, feature_max]
        feature_min = X_feature.min()
        feature_max = X_feature.max()
        accumulated_max = accumulated.max()
        
        if accumulated_max > 0:
            partition_params = partition_params / accumulated_max * (feature_max - feature_min) + feature_min
        else:
            # Fallback if all encoded values are zero
            partition_params = partition_params + feature_min
        
        return partition_params
    
    def _quantiles_to_partitions(
        self,
        X_feature: np.ndarray,
        quantiles: List[float]
    ) -> np.ndarray:
        """
        Convert quantile positions to trapezoid parameters.
        Direct conversion - simple and clear.
        
        Returns
        -------
        partition_params : np.ndarray of shape (3, 4)
            Trapezoid parameters [a, b, c, d] for each linguistic term
        """
        # Compute actual quantile values
        Q = [np.percentile(X_feature, q) for q in quantiles]
        
        # Build trapezoid parameters directly
        partition_params = np.array([
            # Low: [Q0, Q0, Q1, Q2]
            [Q[0], Q[0], Q[1], Q[2]],
            # Medium: [Q1, (Q1+Q2)/2, (Q2+Q3)/2, Q3]
            [Q[1], (Q[1] + Q[2]) / 2, (Q[2] + Q[3]) / 2, Q[3]],
            # High: [Q2, Q3, Q4, Q4]
            [Q[2], Q[3], Q[4], Q[4]]
        ])
        
        return partition_params
    
    def _compute_memberships(
        self,
        X: np.ndarray,
        trapezoid_params: np.ndarray
    ) -> np.ndarray:
        """
        Compute trapezoidal membership values.
        
        Parameters
        ----------
        X : np.ndarray
            Input values
        trapezoid_params : np.ndarray of shape (4,)
            Parameters [a, b, c, d] of the trapezoid
            
        Returns
        -------
        memberships : np.ndarray
            Membership degrees in [0, 1]
        """
        a, b, c, d = trapezoid_params
        X = X.flatten()
        
        memberships = np.zeros_like(X)
        
        # Left slope: (x - a) / (b - a) for x in [a, b]
        mask1 = (X >= a) & (X < b)
        if b > a:
            memberships[mask1] = (X[mask1] - a) / (b - a)
        
        # Plateau: 1 for x in [b, c]
        mask2 = (X >= b) & (X <= c)
        memberships[mask2] = 1.0
        
        # Right slope: (d - x) / (d - c) for x in (c, d]
        mask3 = (X > c) & (X <= d)
        if d > c:
            memberships[mask3] = (d - X[mask3]) / (d - c)
        
        return memberships
    
    def _is_valid_params(self, params: np.ndarray) -> bool:
        """Check if parameters satisfy ordering constraints."""
        if len(params) != 3:
            return False
        return 5 <= params[0] < params[1] < params[2] <= 95
    
    def _project_to_valid_params(self, params: np.ndarray) -> np.ndarray:
        """Project parameters to valid space."""
        # Ensure ordering
        params = np.sort(params)
        
        # Ensure bounds
        params[0] = np.clip(params[0], 5, 35)
        params[1] = np.clip(params[1], params[0] + 5, 65)
        params[2] = np.clip(params[2], params[1] + 5, 95)
        
        return params


def optimize_partitions_for_gfrt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    initial_partitions: list=None,
    method: str = 'separability',
    strategy: str = 'hybrid',
    verbose: bool = True,
    categorical_mask: Optional[np.ndarray] = None
) -> list:
    """
    Convenience function to optimize partitions for GFRT.
    
    Parameters
    ----------
    X_train : np.ndarray of shape (n_samples, n_features)
        Training data
    y_train : np.ndarray of shape (n_samples,)
        Training labels
    method : str, default='separability'
        Optimization metric: 'separability', 'gini', or 'fisher'
    strategy : str, default='hybrid'
        Search strategy: 'grid', 'coordinate', 'gradient', or 'hybrid'
    verbose : bool, default=True
        Print progress information
    categorical_mask : array-like, optional
        Boolean mask indicating which features are categorical
        
    Returns
    -------
    optimized_partitions : list[fs.fuzzyVariable]
        List of fuzzy variables in ex_fuzzy format, ready for GFRT
        
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from tree_learning import FuzzyCART
    >>> 
    >>> # Load data
    >>> X, y = load_iris(return_X_y=True)
    >>> 
    >>> # Optimize partitions
    >>> partitions = optimize_partitions_for_gfrt(
    ...     X, y, method='separability', strategy='hybrid'
    ... )
    >>> 
    >>> # Use optimized partitions in GFRT
    >>> model = FuzzyCART(fuzzy_partitions=partitions)
    >>> model.fit(X, y)
    """
    optimizer = FuzzyPartitionOptimizer(
        optimization_method=method,
        search_strategy=strategy,
        verbose=verbose
    )
    
    return optimizer.optimize_partitions(X_train, y_train, initial_partitions=initial_partitions, categorical_mask=categorical_mask)


def plot_fuzzy_partitions(
    X_feature: np.ndarray,
    initial_partitions: list,
    optimized_partitions: list,
    feature_idx: int = 0,
    save_path: str = None
):
    """
    Visualize initial vs optimized fuzzy partitions for a feature.
    
    Parameters
    ----------
    X_feature : np.ndarray
        Feature values (1D array)
    initial_partitions : list[fs.fuzzyVariable]
        Initial fuzzy partitions
    optimized_partitions : list[fs.fuzzyVariable]
        Optimized fuzzy partitions
    feature_idx : int
        Which feature to plot
    save_path : str, optional
        Path to save the figure
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Create x values for plotting
    x_min, x_max = np.min(X_feature), np.max(X_feature)
    x_range = x_max - x_min
    x_plot = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 500)
    
    colors = ['blue', 'green', 'red']
    labels = ['Low', 'Medium', 'High']
    
    # Plot initial partitions
    ax1.set_title('Initial Fuzzy Partitions', fontsize=14, fontweight='bold')
    for idx, (fuzzy_set, color, label) in enumerate(zip(initial_partitions[feature_idx], colors, labels)):
        memberships = fuzzy_set.membership(x_plot)
        ax1.plot(x_plot, memberships, color=color, linewidth=2, label=label, alpha=0.7)
        ax1.fill_between(x_plot, 0, memberships, color=color, alpha=0.2)
    
    ax1.set_xlabel('Feature Value', fontsize=12)
    ax1.set_ylabel('Membership Degree', fontsize=12)
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot optimized partitions
    ax2.set_title('Optimized Fuzzy Partitions', fontsize=14, fontweight='bold')
    for idx, (fuzzy_set, color, label) in enumerate(zip(optimized_partitions[feature_idx], colors, labels)):
        memberships = fuzzy_set.membership(x_plot)
        ax2.plot(x_plot, memberships, color=color, linewidth=2, label=label, alpha=0.7)
        ax2.fill_between(x_plot, 0, memberships, color=color, alpha=0.2)
    
    ax2.set_xlabel('Feature Value', fontsize=12)
    ax2.set_ylabel('Membership Degree', fontsize=12)
    ax2.set_ylim([-0.05, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved partition plot to: {save_path}")
    
    plt.show()
    
    return fig


if __name__ == '__main__':
    # Example usage
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    print("="*80)
    print("Fuzzy Partition Optimization Demo")
    print("="*80)
    
    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\nDataset: Iris")
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Test different optimization strategies
    strategies = ['grid', 'coordinate', 'gradient', 'hybrid']
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing strategy: {strategy}")
        print(f"{'='*60}")
        
        optimized_partitions = optimize_partitions_for_gfrt(
            X_train, y_train,
            method='separability',
            strategy=strategy,
            verbose=True
        )
        
        print(f"\nOptimized {len(optimized_partitions)} feature partitions")
