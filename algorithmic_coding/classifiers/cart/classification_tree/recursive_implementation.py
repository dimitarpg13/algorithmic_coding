"""
Recursive Classification Tree (CART) Implementation.

This module implements a binary classification tree using a recursive approach.
The tree uses Gini impurity as the splitting criterion.
"""

import numpy as np
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
    Binary Classification Tree using recursive construction.
    
    This implementation builds a decision tree for classification tasks using
    Gini impurity as the splitting criterion. The tree is built recursively.
    
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
        y: np.ndarray
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
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
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
                
                gain = self._gini_gain(y, left_indices, right_indices)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _create_leaf_node(self, y: np.ndarray) -> TreeNode:
        """Create a leaf node with class distribution and prediction."""
        classes, counts = np.unique(y, return_counts=True)
        
        # Create class distribution array
        value = np.zeros(self.n_classes_)
        for cls, count in zip(classes, counts):
            value[cls] = count
        
        predicted_class = classes[np.argmax(counts)]
        
        return TreeNode(value=value, predicted_class=predicted_class)
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        """
        Recursively build the classification tree.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix for current node.
        y : np.ndarray
            Labels for current node.
        depth : int
            Current depth in the tree.
        
        Returns
        -------
        node : TreeNode
            The constructed tree node.
        """
        n_samples = len(y)
        is_pure = len(np.unique(y)) == 1
        
        # Check stopping conditions for creating a leaf
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            is_pure):
            return self._create_leaf_node(y)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no valid split found, make this a leaf
        if best_feature is None or best_gain <= 0:
            return self._create_leaf_node(y)
        
        # Apply the split
        feature_values = X[:, best_feature]
        left_mask = feature_values <= best_threshold
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ClassificationTree':
        """
        Build the classification tree from training data.
        
        Uses a recursive approach to build the tree.
        
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
        
        # Build the tree recursively starting from root
        self.root_ = self._build_tree(X, y, depth=0)
        
        return self
    
    def _predict_single(self, x: np.ndarray, node: TreeNode) -> int:
        """Recursively predict class for a single sample."""
        if node.is_leaf():
            return node.predicted_class
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
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
        return np.array([self._predict_single(x, self.root_) for x in X])
    
    def _predict_proba_single(self, x: np.ndarray, node: TreeNode) -> np.ndarray:
        """Recursively get class probabilities for a single sample."""
        if node.is_leaf():
            return node.value / node.value.sum()
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_proba_single(x, node.left)
        else:
            return self._predict_proba_single(x, node.right)
    
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
        return np.array([self._predict_proba_single(x, self.root_) for x in X])
    
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
    
    def _get_depth(self, node: TreeNode) -> int:
        """Recursively calculate the depth of the tree."""
        if node is None or node.is_leaf():
            return 0
        
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))
    
    def get_depth(self) -> int:
        """Return the depth of the tree."""
        if self.root_ is None:
            return 0
        return self._get_depth(self.root_)
    
    def _count_leaves(self, node: TreeNode) -> int:
        """Recursively count the number of leaf nodes."""
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)
    
    def get_n_leaves(self) -> int:
        """Return the number of leaf nodes."""
        return self._count_leaves(self.root_)
    
    def get_feature_importances(self) -> np.ndarray:
        """
        Calculate feature importances based on Gini decrease.
        
        Returns
        -------
        importances : np.ndarray of shape (n_features,)
            Feature importances (normalized to sum to 1).
        """
        importances = np.zeros(self.n_features_)
        
        def _accumulate_importances(node: TreeNode, n_samples: int):
            if node is None or node.is_leaf():
                return
            
            # Calculate samples in left and right children
            n_left = int(node.left.value.sum()) if node.left.is_leaf() else 0
            n_right = int(node.right.value.sum()) if node.right.is_leaf() else 0
            
            if n_left == 0 and node.left is not None:
                # Internal node - estimate from children
                _accumulate_importances(node.left, n_samples)
            if n_right == 0 and node.right is not None:
                _accumulate_importances(node.right, n_samples)
            
            # Accumulate importance for this feature
            importances[node.feature_index] += 1  # Simple count-based importance
            
            _accumulate_importances(node.left, n_samples)
            _accumulate_importances(node.right, n_samples)
        
        if self.root_ is not None and not self.root_.is_leaf():
            _accumulate_importances(self.root_, 1)
        
        # Normalize
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        return importances


if __name__ == '__main__':
    # Quick sanity test
    np.random.seed(42)
    
    # Generate simple dataset
    X = np.array([
        [1, 2], [2, 3], [3, 4], [4, 5],  # Class 0
        [6, 7], [7, 8], [8, 9], [9, 10]   # Class 1
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    clf = ClassificationTree(max_depth=5)
    clf.fit(X, y)
    
    print("Recursive Classification Tree - Quick Test")
    print("=" * 50)
    print(f"Training accuracy: {clf.score(X, y):.4f}")
    print(f"Tree depth: {clf.get_depth()}")
    print(f"Number of leaves: {clf.get_n_leaves()}")
    print(f"Predictions: {clf.predict(X)}")
    print(f"Expected:    {y}")

