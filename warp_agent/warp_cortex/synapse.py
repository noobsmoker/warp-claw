"""
Topological Synapse implementation using Gudhi for landmark selection.
"""

import numpy as np
from typing import List, Tuple
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


class TopologicalSynapse:
    """Shared buffer storing landmarks selected via TDA."""

    def __init__(self, landmark_count: int = 64):
        self.landmark_count = landmark_count
        self.landmarks: List[np.ndarray] = []
        self._buffer: List[np.ndarray] = []

    def select_landmarks(self, embeddings: np.ndarray) -> List[int]:
        """Select landmark indices using TDA-based sampling."""
        if not GUDHI_AVAILABLE:
            # Fallback to random sampling if Gudhi not available
            n_tokens = len(embeddings)
            return np.random.choice(n_tokens, size=min(self.landmark_count, n_tokens), replace=False).tolist()

        # Use Gudhi for topological landmarking
        # This is a simplified implementation - full TDA would be more complex
        n_tokens = len(embeddings)
        if n_tokens <= self.landmark_count:
            return list(range(n_tokens))

        # Simple farthest point sampling as proxy for TDA landmarking
        selected = [0]  # Start with first point

        for _ in range(min(self.landmark_count - 1, n_tokens - 1)):
            distances = np.array([
                min(np.linalg.norm(embeddings[i] - embeddings[j]) for j in selected)
                for i in range(n_tokens)
            ])
            next_point = np.argmax(distances)
            selected.append(int(next_point))

        return selected

    def update_landmarks(self, embeddings: np.ndarray) -> None:
        """Update the landmark buffer with new embeddings."""
        landmark_indices = self.select_landmarks(embeddings)
        self.landmarks = [embeddings[i] for i in landmark_indices]
        self._buffer = embeddings.tolist()