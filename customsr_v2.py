import numpy as np
import random
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Any
from sklearn.linear_model import LinearRegression


@dataclass
class Dimension:
    """
    Represents physical dimensions using dimensional analysis.
    Currently only tracks length dimension [Length^L].
    
    Args:
        length: The power of the length dimension
    """
    length: int = 0
    
    def __add__(self, other: 'Dimension') -> 'Dimension':
        """Addition requires same dimensions, returns the same dimension."""
        return Dimension(self.length + other.length)
    
    def __sub__(self, other: 'Dimension') -> 'Dimension':
        """Subtraction requires same dimensions, returns the same dimension."""
        return Dimension(self.length - other.length)
    
    def __mul__(self, other: 'Dimension') -> 'Dimension':
        """Multiplication adds dimension powers."""
        return Dimension(self.length + other.length)
    
    def __truediv__(self, other: 'Dimension') -> 'Dimension':
        """Division subtracts dimension powers."""
        return Dimension(self.length - other.length)
    
    def __eq__(self, other: object) -> bool:
        """Check if two dimensions are equal."""
        if not isinstance(other, Dimension):
            return False
        return self.length == other.length
    
    def is_dimensionless(self) -> bool:
        """Check if dimension is dimensionless (all powers are zero)."""
        return self.length == 0


class Node:
    """
    Base class for expression tree nodes in symbolic regression.
    Each node can evaluate itself given input data and track its physical dimensions.
    """
    
    def __init__(self) -> None:
        self.dimension: Dimension = Dimension()
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the node given input data.
        
        Args:
            X: Input data matrix of shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples,) with evaluation results
        """
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def get_dimension(self) -> Dimension:
        """Get the physical dimension of this node's output."""
        return self.dimension
    
    def __str__(self) -> str:
        """String representation of the node."""
        raise NotImplementedError("Subclasses must implement __str__ method")
    
    def collect_features(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Collect features for linear regression optimization.
        Base implementation returns the node's evaluation.
        
        Args:
            X: Input data matrix
            
        Returns:
            List of feature arrays for coefficient optimization
        """
        return [self.evaluate(X)]


class VariableNode(Node):
    """
    Node representing an input variable.
    
    Args:
        index: Column index in the input data matrix
        name: Human-readable name of the variable
        dimension: Physical dimension of this variable
    """
    
    def __init__(self, index: int, name: str, dimension: Dimension) -> None:
        super().__init__()
        self.index: int = index
        self.name: str = name
        self.dimension: Dimension = dimension
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Return the specified column from input data."""
        if self.index >= X.shape[1]:
            raise IndexError(f"Variable index {self.index} out of bounds for data with {X.shape[1]} features")
        return X[:, self.index]
    
    def __str__(self) -> str:
        return self.name


class ConstantNode(Node):
    """
    Node representing a constant value.
    Constants are always dimensionless.
    
    Args:
        value: The constant value
    """
    
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value: float = value
        self.dimension: Dimension = Dimension(0)  # Constants are dimensionless
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Return array filled with the constant value."""
        return np.full(X.shape[0], self.value, dtype=np.float64)
    
    def __str__(self) -> str:
        return f"{self.value:.3f}"

class PiecewiseOpNode(Node):
    """
    Node representing a piecewise operation (if-then-else).
    Evaluates condition and returns true_expr if condition > 0, else false_expr.
    
    Args:
        condition: Node that evaluates to the condition
        true_expr: Node to evaluate when condition > 0
        false_expr: Node to evaluate when condition <= 0
    """
    
    def __init__(self, condition: Node, true_expr: Node, false_expr: Node) -> None:
        super().__init__()
        self.condition: Node = condition
        self.true_expr: Node = true_expr
        self.false_expr: Node = false_expr
        self._calculate_dimension()
    
    def _calculate_dimension(self) -> None:
        """Calculate the resulting dimension. Both branches must have same dimension."""
        # Both true and false expressions must have the same dimension
        if self.true_expr.dimension == self.false_expr.dimension:
            self.dimension = self.true_expr.dimension
        else:
            # Mark as invalid dimension for dimensional analysis violations
            self.dimension = Dimension(-999)  # Sentinel value for invalid
    
    def is_dimensionally_valid(self) -> bool:
        """Check if this operation is dimensionally valid."""
        return self.dimension.length != -999
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the piecewise operation."""
        condition_val: np.ndarray = self.condition.evaluate(X)
        true_val: np.ndarray = self.true_expr.evaluate(X)
        false_val: np.ndarray = self.false_expr.evaluate(X)
        
        # Use numpy.where for vectorized conditional evaluation
        return np.where(condition_val > 0, true_val, false_val)
    
    def collect_features(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Collect features from both branches for linear regression optimization.
        """
        # For piecewise operations, we treat the entire operation as a single feature
        # since the branching logic makes it difficult to decompose linearly
        return [self.evaluate(X)]
    
    def __str__(self) -> str:
        return f"if({self.condition} > 0, {self.true_expr}, {self.false_expr})"

class BinaryOpNode(Node):
    """
    Node representing a binary operation (+, -, *, /).
    
    Args:
        left: Left operand node
        right: Right operand node
        op: Operation string ('+', '-', '*', '/')
    """
    
    def __init__(self, left: Node, right: Node, op: str) -> None:
        super().__init__()
        self.left: Node = left
        self.right: Node = right
        self.op: str = op
        self._calculate_dimension()
    
    def _calculate_dimension(self) -> None:
        """Calculate the resulting dimension based on the operation."""
        if self.op in ['+', '-']:
            # Addition/subtraction requires same dimensions
            if self.left.dimension == self.right.dimension:
                self.dimension = self.left.dimension
            else:
                # Mark as invalid dimension for dimensional analysis violations
                self.dimension = Dimension(-999)  # Sentinel value for invalid
        elif self.op == '*':
            self.dimension = self.left.dimension * self.right.dimension
        elif self.op == '/':
            self.dimension = self.left.dimension / self.right.dimension
        else:
            raise ValueError(f"Unknown binary operation: {self.op}")
    
    def is_dimensionally_valid(self) -> bool:
        """Check if this operation is dimensionally valid."""
        return self.dimension.length != -999
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the binary operation."""
        left_val: np.ndarray = self.left.evaluate(X)
        right_val: np.ndarray = self.right.evaluate(X)
        
        if self.op == '+':
            return left_val + right_val
        elif self.op == '-':
            return left_val - right_val
        elif self.op == '*':
            return left_val * right_val
        elif self.op == '/':
            # Avoid division by zero
            return np.divide(left_val, right_val, 
                           out=np.ones_like(left_val), 
                           where=np.abs(right_val) > 1e-10)
        else:
            raise ValueError(f"Unknown binary operation: {self.op}")
    
    def collect_features(self, X: np.ndarray) -> List[np.ndarray]:
        """
        For additive operations, collect features from both sides.
        For multiplicative operations, treat as single feature.
        """
        if self.op in ['+', '-']:
            left_features = self.left.collect_features(X)
            right_features = self.right.collect_features(X)
            return left_features + right_features
        else:
            return [self.evaluate(X)]
    
    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class UnaryOpNode(Node):
    """
    Node representing a unary operation (sqrt, log, exp).
    
    Args:
        child: The operand node
        op: Operation string ('sqrt', 'log', 'exp')
    """
    
    def __init__(self, child: Node, op: str) -> None:
        super().__init__()
        self.child: Node = child
        self.op: str = op
        self._calculate_dimension()
    
    def _calculate_dimension(self) -> None:
        """Calculate the resulting dimension. Most functions require dimensionless input."""
        if self.op in ['sqrt', 'log', 'exp']:
            # These functions require dimensionless input and produce dimensionless output
            if self.child.dimension.is_dimensionless():
                self.dimension = Dimension(0)
            else:
                # Mark as invalid for dimensional analysis violations
                self.dimension = Dimension(-999)
        elif self.op == 'square':
            # Square doubles the dimension
            self.dimension = self.child.dimension * self.child.dimension
        else:
            raise ValueError(f"Unknown unary operation: {self.op}")
    
    def is_dimensionally_valid(self) -> bool:
        """Check if this operation is dimensionally valid."""
        return self.dimension.length != -999
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the unary operation."""
        child_val: np.ndarray = self.child.evaluate(X)
        
        if self.op == 'sqrt':
            # Take square root of absolute value to avoid complex numbers
            return np.sqrt(np.abs(child_val))
        elif self.op == 'log':
            # Add small epsilon to avoid log(0)
            return np.log(np.abs(child_val) + 1e-10)
        elif self.op == 'exp':
            # Clip to avoid overflow
            return np.exp(np.clip(child_val, -10, 10))
        elif self.op == 'square':
            return child_val ** 2
        else:
            raise ValueError(f"Unknown unary operation: {self.op}")
    
    def __str__(self) -> str:
        return f"{self.op}({self.child})"


class OptimizedExpression:
    """
    Wrapper for an expression tree with optimized linear coefficients.
    
    Args:
        expression: The base expression tree
        coefficients: Optional array of linear coefficients for each feature
    """
    
    def __init__(self, expression: Node, coefficients: Optional[np.ndarray] = None) -> None:
        self.expression: Node = expression
        self.coefficients: Optional[np.ndarray] = coefficients
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the optimized expression.
        
        Args:
            X: Input data matrix
            
        Returns:
            Predicted values
        """
        if self.coefficients is None:
            return self.expression.evaluate(X)
        
        try:
            features: List[np.ndarray] = self.expression.collect_features(X)
            if len(features) == 0:
                return np.zeros(X.shape[0])
            
            result: np.ndarray = np.zeros(X.shape[0])
            
            for i, feature in enumerate(features):
                if i < len(self.coefficients):
                    result += self.coefficients[i] * feature
            
            return result
        except Exception:
            # Fallback to unoptimized evaluation if something goes wrong
            return self.expression.evaluate(X)
    
    def __str__(self) -> str:
        if self.coefficients is None:
            return str(self.expression)
        
        try:
            # Build string representation with optimized coefficients
            features_str: List[str] = self._get_feature_strings(self.expression)
            terms: List[str] = []
            
            for i, (coef, feat_str) in enumerate(zip(self.coefficients, features_str)):
                if abs(coef) > 1e-6:  # Only include significant terms
                    if feat_str == "1":  # Constant term
                        terms.append(f"{coef:.3f}")
                    else:
                        if coef >= 0 and len(terms) > 0:
                            terms.append(f" + {coef:.3f}*{feat_str}")
                        else:
                            terms.append(f"{coef:.3f}*{feat_str}")
            
            return "".join(terms) if terms else "0"
        except Exception:
            return str(self.expression)
    
    def _get_feature_strings(self, node: Node) -> List[str]:
        """
        Get string representations of features for display purposes.
        
        Args:
            node: Node to extract feature strings from
            
        Returns:
            List of feature string representations
        """
        if isinstance(node, ConstantNode):
            return ["1"]
        elif isinstance(node, VariableNode):
            return [node.name]
        elif isinstance(node, BinaryOpNode):
            if node.op in ['+', '-']:
                left_strs = self._get_feature_strings(node.left)
                right_strs = self._get_feature_strings(node.right)
                return left_strs + right_strs
            else:
                return [str(node)]
        elif isinstance(node, UnaryOpNode):
            return [str(node)]
        return [str(node)]


class DimensionalSymbolicRegressor:
    """
    Genetic Programming-based Symbolic Regressor with dimensional analysis.
    
    This regressor evolves mathematical expressions that respect physical dimensions
    and uses linear regression to optimize coefficients.
    """
    
    def __init__(self, 
                 population_size: int = 100, 
                 generations: int = 50, 
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8, 
                 max_depth: int = 6, 
                 tournament_size: int = 3, 
                 fitness_threshold: Optional[float] = None) -> None:
        """
        Initialize the symbolic regressor.
        
        Args:
            population_size: Number of individuals in each generation
            generations: Maximum number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            max_depth: Maximum depth of expression trees
            tournament_size: Size of tournament for selection
            fitness_threshold: Early stopping threshold (optional)
        """
        self.population_size: int = population_size
        self.generations: int = generations
        self.mutation_rate: float = mutation_rate
        self.crossover_rate: float = crossover_rate
        self.max_depth: int = max_depth
        self.tournament_size: int = tournament_size
        self.fitness_threshold: Optional[float] = fitness_threshold
        
        self.population: List[Node] = []
        self.best_individual: Optional[OptimizedExpression] = None
        
        # Define variable dimensions for perovskite tolerance factor prediction
        self.variable_dims: dict[str, Dimension] = {
            'nA': Dimension(0),      # dimensionless (coordination number)
            'nB': Dimension(0),      # dimensionless (coordination number)
            'nX': Dimension(0),      # dimensionless (coordination number)
            'rA': Dimension(1),      # length (ionic radius)
            'rB': Dimension(1),      # length (ionic radius)
            'rX': Dimension(1),      # length (ionic radius)
            'lattice constant': Dimension(1)  # length
        }
        self.variable_names: List[str] = list(self.variable_dims.keys())
        self.target_dimension: Dimension = Dimension(0)  # tolerance factor is dimensionless
    
    def create_random_tree(self, max_depth: int) -> Node:
        """
        Create a random expression tree with dimensional constraints.
        
        Args:
            max_depth: Maximum depth of the tree
            
        Returns:
            Random Node representing an expression
        """
        if max_depth <= 1 or random.random() < 0.3:
            # Create terminal node (leaf)
            if random.random() < 0.7:  # Variable
                var_idx: int = random.randint(0, len(self.variable_names) - 1)
                var_name: str = self.variable_names[var_idx]
                return VariableNode(var_idx, var_name, self.variable_dims[var_name])
            else:  # Constant
                return ConstantNode(random.uniform(-3, 3))
        else:
            # Create function node (internal node)
            node_type = random.random()
            if node_type < 0.6:  # Binary operator
                op: str = random.choice(['+', '-', '*', '/'])
                left: Node = self.create_random_tree(max_depth - 1)
                right: Node = self.create_random_tree(max_depth - 1)
                binary_node: BinaryOpNode = BinaryOpNode(left, right, op)
                
                # Check dimensional validity and retry if invalid
                if not binary_node.is_dimensionally_valid():
                    return self.create_random_tree(max_depth)
                return binary_node
            elif node_type < 0.7:  # Unary operator
                op: str = random.choice(['sqrt', 'log'])
                child: Node = self.create_random_tree(max_depth - 1)
                unary_node: UnaryOpNode = UnaryOpNode(child, op)
                
                # Check dimensional validity and retry if invalid
                if not unary_node.is_dimensionally_valid():
                    return self.create_random_tree(max_depth)
                return unary_node
            else:  # Piecewise operator
                condition: Node = self.create_random_tree(max_depth - 1)
                true_expr: Node = self.create_random_tree(max_depth - 1)
                false_expr: Node = self.create_random_tree(max_depth - 1)
                
                # Create the piecewise node
                piecewise_node = PiecewiseOpNode(condition, true_expr, false_expr)
                
                # Check dimensional validity and retry if invalid
                if not piecewise_node.is_dimensionally_valid():
                    return self.create_random_tree(max_depth)
                return piecewise_node
    
    def count_variables(self, node: Node) -> int:
        """
        Count the number of variable nodes in the expression tree.
        
        Args:
            node: Root node of the expression tree
            
        Returns:
            Number of variable nodes
        """
        if isinstance(node, VariableNode):
            return 1
        elif isinstance(node, ConstantNode):
            return 0
        elif isinstance(node, BinaryOpNode):
            return self.count_variables(node.left) + self.count_variables(node.right)
        elif isinstance(node, UnaryOpNode):
            return self.count_variables(node.child)
        elif isinstance(node, PiecewiseOpNode):
            return (self.count_variables(node.condition) + 
                   self.count_variables(node.true_expr) + 
                   self.count_variables(node.false_expr))
        return 0
    
    def count_nodes(self, node: Node) -> int:
        """
        Count the total number of nodes in the expression tree.
        
        Args:
            node: Root node of the expression tree
            
        Returns:
            Total number of nodes
        """
        if isinstance(node, (VariableNode, ConstantNode)):
            return 1
        elif isinstance(node, BinaryOpNode):
            return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)
        elif isinstance(node, UnaryOpNode):
            return 1 + self.count_nodes(node.child)
        elif isinstance(node, PiecewiseOpNode):
            return (1 + self.count_nodes(node.condition) + 
                   self.count_nodes(node.true_expr) + 
                   self.count_nodes(node.false_expr))
        return 1
    
    def is_trivial_expression(self, individual: Node) -> bool:
        """
        Check if expression is trivial (e.g., constant only or no meaningful variables).
        
        Args:
            individual: Expression tree to check
            
        Returns:
            True if the expression is considered trivial
        """
        var_count: int = self.count_variables(individual)
        total_nodes: int = self.count_nodes(individual)
        
        # Expression is trivial if:
        # 1. It has no variables at all
        # 2. It's just a single constant
        # 3. Variables make up less than 20% of the expression (mostly constants)
        if var_count == 0:
            return True
        if total_nodes == 1 and isinstance(individual, ConstantNode):
            return True
        if total_nodes > 1 and var_count / total_nodes < 0.2:
            return True
        
        return False
    
    def optimize_coefficients(self, expression: Node, X: np.ndarray, y: np.ndarray) -> OptimizedExpression:
        """
        Optimize linear coefficients for the expression using least squares regression.
        
        Args:
            expression: Expression tree to optimize
            X: Input data matrix
            y: Target values
            
        Returns:
            OptimizedExpression with fitted coefficients
        """
        try:
            features: List[np.ndarray] = expression.collect_features(X)
            
            if len(features) == 0:
                return OptimizedExpression(expression)
            
            # Stack features into matrix
            X_features: np.ndarray = np.column_stack(features)
            
            # Handle edge cases
            if X_features.shape[1] == 0:
                return OptimizedExpression(expression)
            
            # Check for constant features or perfect multicollinearity
            if np.linalg.matrix_rank(X_features) < X_features.shape[1]:
                # Use pseudo-inverse for rank-deficient matrices
                coefficients, _, _, _ = np.linalg.lstsq(X_features, y, rcond=None)
                return OptimizedExpression(expression, coefficients)
            
            # Use linear regression to find optimal coefficients
            lr: LinearRegression = LinearRegression(fit_intercept=False)
            lr.fit(X_features, y)
            
            return OptimizedExpression(expression, lr.coef_)
            
        except Exception:
            # If optimization fails, return unoptimized expression
            return OptimizedExpression(expression)
    
    def fitness(self, individual: Node, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate fitness score for an individual (lower is better).
        
        Args:
            individual: Expression tree to evaluate
            X: Input data matrix
            y: Target values
            
        Returns:
            Fitness score (MSE + penalties)
        """
        try:
            # Check if expression is dimensionally valid for target
            if not individual.dimension.is_dimensionless():
                return float('inf')  # Invalid dimensions
            
            # Heavy penalty for trivial expressions
            if self.is_trivial_expression(individual):
                return float('inf')
            
            # Optimize coefficients for this expression
            optimized_expr: OptimizedExpression = self.optimize_coefficients(individual, X, y)
            pred: np.ndarray = optimized_expr.evaluate(X)
            
            # Handle invalid values
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return float('inf')
            
            # Check for constant predictions (no variation)
            if np.std(pred) < 1e-10:
                return float('inf')  # Penalize constant outputs
            
            # Mean squared error
            mse: float = float(np.mean((pred - y) ** 2))
            
            # Add complexity penalty (encourages simpler expressions)
            complexity: int = self.count_nodes(individual)
            complexity_penalty: float = 0.001 * complexity
            
            # Add diversity penalty (encourages variable usage)
            var_count: int = self.count_variables(individual)
            diversity_penalty: float = 0.1 if var_count < 2 else 0.0
            
            return mse + complexity_penalty + diversity_penalty
            
        except Exception:
            return float('inf')
    
    def tournament_selection(self, fitnesses: List[float]) -> int:
        """
        Select an individual using tournament selection.
        
        Args:
            fitnesses: List of fitness scores for the population
            
        Returns:
            Index of selected individual
        """
        tournament_indices: List[int] = random.sample(range(len(fitnesses)), 
                                                     min(self.tournament_size, len(fitnesses)))
        best_idx: int = min(tournament_indices, key=lambda i: fitnesses[i])
        return best_idx
    
    def crossover(self, parent1: Node, parent2: Node) -> Tuple[Node, Node]:
        """
        Perform crossover between two parent nodes.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two offspring
        """
        # Simple subtree crossover (for now, just return deep copies)
        # TODO: Implement proper subtree crossover
        child1: Node = copy.deepcopy(parent1)
        child2: Node = copy.deepcopy(parent2)
        return child1, child2
    
    def mutate(self, individual: Node) -> Node:
        """
        Mutate an individual by replacing it with a new random subtree.
        
        Args:
            individual: Individual to potentially mutate
            
        Returns:
            Mutated individual (or original if no mutation occurred)
        """
        if random.random() < self.mutation_rate:
            return self.create_random_tree(self.max_depth)
        return copy.deepcopy(individual)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the symbolic regressor to training data.
        
        Args:
            X: Input features matrix of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
        """
        # Validate input dimensions
        if X.shape[1] != len(self.variable_names):
            raise ValueError(f"Expected {len(self.variable_names)} features, got {X.shape[1]}")
        
        # Initialize population
        self.population = []
        for _ in range(self.population_size):
            individual: Node = self.create_random_tree(self.max_depth)
            self.population.append(individual)
        
        best_fitness: float = float('inf')
        
        print(f"Starting evolution with {self.population_size} individuals for {self.generations} generations")
        
        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            fitnesses: List[float] = [self.fitness(ind, X, y) for ind in self.population]
            
            # Track best individual
            min_fitness: float = min(fitnesses)
            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_idx: int = fitnesses.index(min_fitness)
                best_expr: Node = self.population[best_idx]
                # Store optimized version of best expression
                self.best_individual = self.optimize_coefficients(best_expr, X, y)
            
            # Progress reporting
            print(f"Generation {generation:3d}: Best fitness = {best_fitness:.6f}")
            if self.best_individual:
                var_count: int = self.count_variables(self.best_individual.expression)
                print(f"  Best expression: {self.best_individual} (Variables: {var_count})")
            
            # Check for early stopping
            if self.fitness_threshold is not None and best_fitness <= self.fitness_threshold:
                print(f"Early stopping at generation {generation}: "
                      f"fitness {best_fitness:.6f} <= threshold {self.fitness_threshold}")
                break
            
            # Create new population through evolution
            new_population: List[Node] = []
            
            # Elitism: keep best individual's expression structure
            if self.best_individual is not None:
                new_population.append(copy.deepcopy(self.best_individual.expression))
            
            # Generate rest of population through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                # Selection
                parent1_idx: int = self.tournament_selection(fitnesses)
                parent2_idx: int = self.tournament_selection(fitnesses)
                
                parent1: Node = self.population[parent1_idx]
                parent2: Node = self.population[parent2_idx]
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Ensure exact population size
            self.population = new_population[:self.population_size]
        
        print(f"\nEvolution completed!")
        print(f"Final best fitness: {best_fitness:.6f}")
        if self.best_individual:
            var_count = self.count_variables(self.best_individual.expression)
            print(f"Final best expression: {self.best_individual} (Variables: {var_count})")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the best evolved expression.
        
        Args:
            X: Input features matrix
            
        Returns:
            Predicted values
            
        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if self.best_individual is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if X.shape[1] != len(self.variable_names):
            raise ValueError(f"Expected {len(self.variable_names)} features, got {X.shape[1]}")
        
        return self.best_individual.evaluate(X)
    
    def get_best_expression(self) -> Optional[str]:
        """
        Get string representation of the best expression found.
        
        Returns:
            String representation of best expression, or None if not fitted
        """
        if self.best_individual is None:
            return None
        return str(self.best_individual)
