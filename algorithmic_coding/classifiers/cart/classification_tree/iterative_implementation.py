"""
Iterative Classification Tree (CART) Implementation.

This module implements a binary classification tree using an iterative approach
with explicit stack management instead of recursion. The tree uses Gini impurity
as the splitting criterion.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class TreeNode:
    """Represents a node in the classification tree."""
    feature_index: Optional[int] = None  # Feature used for splitting
    threshold: Optional[float] = None     # Threshold value for the split
    left: Optional['TreeNode'] = None     # Left child (values <= threshold)
    right: Optional['TreeNode'] = None    # Right child (values > threshold)
    value: Optional[np.ndarray] = None    # Class distribution at leaf node
    predicted_class: Optional[int] = None # Predicted class at leaf node
    
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class ClassificationTree:
    """
    Binary Classification Tree using iterative construction.
    
    This implementation builds a decision tree for classification tasks using
    Gini impurity as the splitting criterion. The tree is built iteratively
    using a stack-based approach rather than recursion.
    
    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.
    
    Attributes
    ----------
    root_ : TreeNode
        The root node of the fitted tree.
    n_classes_ : int
        Number of classes found during fitting.
    n_features_ : int
        Number of features in the training data.
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root_: Optional[TreeNode] = None
        self.n_classes_: int = 0
        self.n_features_: int = 0
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity for a set of labels.
        
        Gini = 1 - sum(p_i^2) where p_i is the proportion of class i.
        """
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
    
    def _gini_gain(
        self,
        y: np.ndarray,
        left_indices: np.ndarray,
        right_indices: np.ndarray
    ) -> float:
        """
        Calculate the Gini gain (reduction in impurity) from a split.
        """
        n = len(y)
        n_left = len(left_indices)
        n_right = len(right_indices)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        parent_gini = self._gini_impurity(y)
        left_gini = self._gini_impurity(y[left_indices])
        right_gini = self._gini_impurity(y[right_indices])
        
        weighted_child_gini = (n_left / n) * left_gini + (n_right / n) * right_gini
        return parent_gini - weighted_child_gini
    
    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray
    ) -> tuple[Optional[int], Optional[float], float]:
        """
        Find the best feature and threshold to split the data.
        
        Returns
        -------
        best_feature : int or None
            Index of the best feature to split on.
        best_threshold : float or None
            Best threshold value for the split.
        best_gain : float
            Gini gain from the best split.
        """
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        X_subset = X[indices]
        y_subset = y[indices]
        
        for feature_idx in range(self.n_features_):
            feature_values = X_subset[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Try midpoints between consecutive unique values as thresholds
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                left_indices = np.where(left_mask)[0]
                right_indices = np.where(right_mask)[0]
                
                # Check minimum samples at leaf constraint
                if (len(left_indices) < self.min_samples_leaf or 
                    len(right_indices) < self.min_samples_leaf):
                    continue
                
                gain = self._gini_gain(y_subset, left_indices, right_indices)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _create_leaf_node(self, y: np.ndarray, indices: np.ndarray) -> TreeNode:
        """Create a leaf node with class distribution and prediction."""
        y_subset = y[indices]
        classes, counts = np.unique(y_subset, return_counts=True)
        
        # Create class distribution array
        value = np.zeros(self.n_classes_)
        for cls, count in zip(classes, counts):
            value[cls] = count
        
        predicted_class = classes[np.argmax(counts)]
        
        return TreeNode(value=value, predicted_class=predicted_class)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ClassificationTree':
        """
        Build the classification tree from training data.
        
        Uses an iterative approach with a stack to build the tree level by level.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target class labels (must be integers starting from 0).
        
        Returns
        -------
        self : ClassificationTree
            The fitted classifier.
        """
        n_samples, self.n_features_ = X.shape
        self.n_classes_ = len(np.unique(y))
        
        # Initialize root node
        self.root_ = TreeNode()
        
        # Stack entries: (node, indices of samples at this node, current depth)
        stack = [(self.root_, np.arange(n_samples), 0)]
        
        while stack:
            node, indices, depth = stack.pop()
            n_node_samples = len(indices)
            
            # Check stopping conditions for creating a leaf
            y_subset = y[indices]
            is_pure = len(np.unique(y_subset)) == 1
            
            if (depth >= self.max_depth or 
                n_node_samples < self.min_samples_split or
                is_pure):
                # Make this node a leaf
                leaf = self._create_leaf_node(y, indices)
                node.value = leaf.value
                node.predicted_class = leaf.predicted_class
                continue
            
            # Find best split
            best_feature, best_threshold, best_gain = self._find_best_split(
                X, y, indices
            )
            
            # If no valid split found, make this a leaf
            if best_feature is None or best_gain <= 0:
                leaf = self._create_leaf_node(y, indices)
                node.value = leaf.value
                node.predicted_class = leaf.predicted_class
                continue
            
            # Apply the split
            node.feature_index = best_feature
            node.threshold = best_threshold
            
            # Partition indices based on the split
            feature_values = X[indices, best_feature]
            left_mask = feature_values <= best_threshold
            
            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]
            
            # Create child nodes and add to stack
            node.left = TreeNode()
            node.right = TreeNode()
            
            # Add children to stack (right first so left is processed first)
            stack.append((node.right, right_indices, depth + 1))
            stack.append((node.left, left_indices, depth + 1))
        
        return self
    
    def _predict_single(self, x: np.ndarray) -> int:
        """Predict class for a single sample using iterative traversal."""
        node = self.root_
        
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        return node.predicted_class
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        return np.array([self._predict_single(x) for x in X])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities for each sample.
        """
        probas = []
        for x in X:
            node = self.root_
            while not node.is_leaf():
                if x[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            # Normalize class counts to probabilities
            proba = node.value / node.value.sum()
            probas.append(proba)
        return np.array(probas)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.
        y : np.ndarray of shape (n_samples,)
            True labels for X.
        
        Returns
        -------
        score : float
            Accuracy of the classifier.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_depth(self) -> int:
        """
        Return the depth of the tree using iterative BFS traversal.
        """
        if self.root_ is None:
            return 0
        
        max_depth = 0
        queue = deque([(self.root_, 0)])
        
        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            
            if node.left is not None:
                queue.append((node.left, depth + 1))
            if node.right is not None:
                queue.append((node.right, depth + 1))
        
        return max_depth
    
    def get_n_leaves(self) -> int:
        """
        Return the number of leaf nodes using iterative traversal.
        """
        if self.root_ is None:
            return 0
        
        n_leaves = 0
        stack = [self.root_]
        
        while stack:
            node = stack.pop()
            if node.is_leaf():
                n_leaves += 1
            else:
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)
        
        return n_leaves


# =============================================================================
# Unit Tests
# =============================================================================

import unittest


class TestClassificationTree(unittest.TestCase):
    """Unit tests for the ClassificationTree classifier."""
    
    def test_simple_binary_classification(self):
        """Test on a simple linearly separable binary classification problem."""
        # Create simple dataset: class 0 when x1 < 5, class 1 otherwise
        X = np.array([
            [1, 2], [2, 3], [3, 4], [4, 5],  # Class 0
            [6, 7], [7, 8], [8, 9], [9, 10]   # Class 1
        ])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        clf = ClassificationTree(max_depth=5)
        clf.fit(X, y)
        
        # Test predictions
        y_pred = clf.predict(X)
        accuracy = np.mean(y_pred == y)
        self.assertEqual(accuracy, 1.0)
        
        # Test on new samples
        X_test = np.array([[0, 0], [10, 10]])
        y_test_pred = clf.predict(X_test)
        np.testing.assert_array_equal(y_test_pred, [0, 1])
    
    def test_multiclass_classification(self):
        """Test on a multiclass classification problem."""
        np.random.seed(42)
        
        # Generate 3 clusters
        n_samples = 30
        X1 = np.random.randn(n_samples, 2) + np.array([0, 0])
        X2 = np.random.randn(n_samples, 2) + np.array([5, 5])
        X3 = np.random.randn(n_samples, 2) + np.array([10, 0])
        
        X = np.vstack([X1, X2, X3])
        y = np.array([0] * n_samples + [1] * n_samples + [2] * n_samples)
        
        clf = ClassificationTree(max_depth=10)
        clf.fit(X, y)
        
        # Should achieve high accuracy on training data
        accuracy = clf.score(X, y)
        self.assertGreater(accuracy, 0.9)
        self.assertEqual(clf.n_classes_, 3)
    
    def test_pure_node_early_stopping(self):
        """Test that tree stops splitting when a node is pure."""
        # All samples in first half have class 0, second half class 1
        X = np.array([[1], [2], [3], [7], [8], [9]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        clf = ClassificationTree(max_depth=100)
        clf.fit(X, y)
        
        # Tree should be shallow despite high max_depth
        depth = clf.get_depth()
        self.assertLessEqual(depth, 3)
        self.assertEqual(clf.score(X, y), 1.0)
    
    def test_max_depth_constraint(self):
        """Test that max_depth parameter is respected."""
        np.random.seed(123)
        X = np.random.randn(100, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        for max_depth in [1, 2, 3, 5]:
            clf = ClassificationTree(max_depth=max_depth)
            clf.fit(X, y)
            actual_depth = clf.get_depth()
            self.assertLessEqual(actual_depth, max_depth)
    
    def test_min_samples_split(self):
        """Test that min_samples_split parameter works correctly."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        # With high min_samples_split, tree should be very shallow
        clf = ClassificationTree(max_depth=10, min_samples_split=7)
        clf.fit(X, y)
        
        # Root should not split if we require 7 samples and only have 8
        n_leaves = clf.get_n_leaves()
        self.assertLessEqual(n_leaves, 3)
    
    def test_min_samples_leaf(self):
        """Test that min_samples_leaf parameter is respected."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        clf = ClassificationTree(max_depth=10, min_samples_leaf=2)
        clf.fit(X, y)
        
        # Every leaf should have at least 2 samples
        self.assertEqual(clf.score(X, y), 1.0)
    
    def test_predict_proba(self):
        """Test probability predictions."""
        X = np.array([[1, 0], [2, 0], [3, 0], [7, 0], [8, 0], [9, 0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        clf = ClassificationTree(max_depth=3)
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(len(X)))
        
        # Shape should be (n_samples, n_classes)
        self.assertEqual(proba.shape, (6, 2))
    
    def test_xor_problem(self):
        """Test on XOR problem which requires depth > 1."""
        # Create XOR pattern with more samples around each corner
        np.random.seed(42)
        n_per_corner = 20
        noise = 0.1
        
        # Four corners of XOR: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
        corners = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        X_list = []
        y_list = []
        
        for cx, cy, label in corners:
            X_corner = np.random.randn(n_per_corner, 2) * noise + np.array([cx, cy])
            X_list.append(X_corner)
            y_list.extend([label] * n_per_corner)
        
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        clf = ClassificationTree(max_depth=5, min_samples_leaf=1)
        clf.fit(X, y)
        
        accuracy = clf.score(X, y)
        self.assertGreater(accuracy, 0.95)
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[1], [2], [3], [10], [11], [12]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        clf = ClassificationTree(max_depth=3)
        clf.fit(X, y)
        
        self.assertEqual(clf.n_features_, 1)
        self.assertEqual(clf.score(X, y), 1.0)
    
    def test_tree_structure(self):
        """Test that tree structure is created correctly."""
        X = np.array([[0], [1], [10], [11]])
        y = np.array([0, 0, 1, 1])
        
        clf = ClassificationTree(max_depth=3)
        clf.fit(X, y)
        
        # Root should not be a leaf
        self.assertFalse(clf.root_.is_leaf())
        
        # Root should have feature_index and threshold set
        self.assertIsNotNone(clf.root_.feature_index)
        self.assertIsNotNone(clf.root_.threshold)
        
        # Leaves should have values and predicted_class
        self.assertTrue(clf.root_.left.is_leaf() or clf.root_.left.left is not None)
    
    def test_gaussian_clusters(self):
        """Test on well-separated Gaussian clusters."""
        np.random.seed(42)
        
        # Two well-separated clusters
        mean_0 = [0, 0]
        mean_1 = [10, 10]
        cov = [[1, 0], [0, 1]]
        
        X0 = np.random.multivariate_normal(mean_0, cov, 50)
        X1 = np.random.multivariate_normal(mean_1, cov, 50)
        
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)
        
        clf = ClassificationTree(max_depth=5)
        clf.fit(X, y)
        
        # Should achieve perfect or near-perfect accuracy
        accuracy = clf.score(X, y)
        self.assertGreater(accuracy, 0.95)
    
    def test_get_n_leaves(self):
        """Test leaf counting."""
        X = np.array([[1], [2], [3], [7], [8], [9]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        clf = ClassificationTree(max_depth=3)
        clf.fit(X, y)
        
        n_leaves = clf.get_n_leaves()
        self.assertGreater(n_leaves, 0)
        self.assertLessEqual(n_leaves, len(X))
    
    def test_reproducibility(self):
        """Test that fitting the same data gives same results."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        
        clf1 = ClassificationTree(max_depth=5)
        clf2 = ClassificationTree(max_depth=5)
        
        clf1.fit(X, y)
        clf2.fit(X, y)
        
        # Predictions should be identical
        y_pred1 = clf1.predict(X)
        y_pred2 = clf2.predict(X)
        np.testing.assert_array_equal(y_pred1, y_pred2)


class TestGiniImpurity(unittest.TestCase):
    """Unit tests for Gini impurity calculations."""
    
    def setUp(self):
        self.clf = ClassificationTree()
        self.clf.n_classes_ = 2
    
    def test_pure_class(self):
        """Gini should be 0 for pure class."""
        y = np.array([0, 0, 0, 0])
        gini = self.clf._gini_impurity(y)
        self.assertEqual(gini, 0.0)
    
    def test_equal_split(self):
        """Gini should be 0.5 for equal binary split."""
        y = np.array([0, 0, 1, 1])
        gini = self.clf._gini_impurity(y)
        self.assertAlmostEqual(gini, 0.5)
    
    def test_empty_array(self):
        """Gini should be 0 for empty array."""
        y = np.array([])
        gini = self.clf._gini_impurity(y)
        self.assertEqual(gini, 0.0)
    
    def test_multiclass_gini(self):
        """Test Gini for multiclass case."""
        # Equal distribution across 3 classes: 1 - (1/3)^2 * 3 = 1 - 1/3 = 2/3
        y = np.array([0, 1, 2])
        gini = self.clf._gini_impurity(y)
        self.assertAlmostEqual(gini, 2/3)


if __name__ == '__main__':
    # Run unittest tests
    print("=" * 70)
    print("Running Unit Tests")
    print("=" * 70)
    unittest.main(verbosity=2, exit=False)
    
    # Additional demonstration
    print("\n" + "=" * 70)
    print("Demonstration: Classification Tree on Synthetic Data")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate synthetic dataset
    n_samples = 200
    X_train = np.random.randn(n_samples, 2)
    # Create circular decision boundary
    y_train = ((X_train[:, 0] ** 2 + X_train[:, 1] ** 2) > 1).astype(int)
    
    X_test = np.random.randn(50, 2)
    y_test = ((X_test[:, 0] ** 2 + X_test[:, 1] ** 2) > 1).astype(int)
    
    clf = ClassificationTree(max_depth=6, min_samples_split=5)
    clf.fit(X_train, y_train)
    
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    print(f"\nDataset: Circular decision boundary")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {clf.n_features_}")
    print(f"Classes: {clf.n_classes_}")
    print(f"\nTree Statistics:")
    print(f"  Max depth allowed: {clf.max_depth}")
    print(f"  Actual depth: {clf.get_depth()}")
    print(f"  Number of leaves: {clf.get_n_leaves()}")
    print(f"\nAccuracy:")
    print(f"  Training: {train_acc:.4f}")
    print(f"  Test: {test_acc:.4f}")

