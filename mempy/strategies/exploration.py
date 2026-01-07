"""
Relation exploration strategies for memory management.

This module provides strategies for discovering potential relationships
between memories that aren't explicitly connected.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from mempy.core.memory import Memory
from mempy.core.interfaces import RelationType


class RelationExplorationStrategy(ABC):
    """
    Abstract base class for relation exploration strategies.

    A relation exploration strategy discovers potential relationships
    between memories that may not be explicitly connected, enabling
    "subconscious" memory association.
    """

    @abstractmethod
    async def explore(
        self,
        memories: List[Memory],
        similarity_threshold: float
    ) -> List[Tuple[Memory, Memory, RelationType]]:
        """
        Explore and discover new relations between memories.

        Args:
            memories: List of memories to explore for potential relations
            similarity_threshold: Minimum similarity/confidence to establish a relation

        Returns:
            List of tuples (memory1, memory2, relation_type) representing
            newly discovered relations
        """
        pass


class CosineSimilarityExplorer(RelationExplorationStrategy):
    """
    Cosine similarity-based relation exploration strategy.

    Discovers relations by computing semantic similarity between memory
    embeddings using cosine similarity. Pairs with similarity above
    the threshold are connected with a SIMILAR relation type.

    This is a simple but effective strategy that leverages existing
    embeddings without requiring additional computation or external services.
    """

    def __init__(self, max_relations: int = 100):
        """
        Initialize the cosine similarity explorer.

        Args:
            max_relations: Maximum number of new relations to discover
                          in a single exploration run
        """
        self.max_relations = max_relations

    async def explore(
        self,
        memories: List[Memory],
        similarity_threshold: float
    ) -> List[Tuple[Memory, Memory, RelationType]]:
        """
        Explore relations using cosine similarity.

        Args:
            memories: List of memories to explore
            similarity_threshold: Minimum cosine similarity [0.0, 1.0]

        Returns:
            List of (memory1, memory2, RelationType.SIMILAR) tuples
        """
        if not memories:
            return []

        relations = []
        count = 0

        # Compare all pairs (avoiding duplicates)
        for i, mem1 in enumerate(memories):
            if count >= self.max_relations:
                break

            for mem2 in memories[i+1:]:
                if count >= self.max_relations:
                    break

                # Calculate cosine similarity
                similarity = self._cosine_similarity(
                    mem1.embedding,
                    mem2.embedding
                )

                # Establish relation if similarity exceeds threshold
                if similarity >= similarity_threshold:
                    relations.append((mem1, mem2, RelationType.SIMILAR))
                    count += 1

        return relations

    def _cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            float: Cosine similarity in range [-1.0, 1.0]
                   (typically [0.0, 1.0] for normalized embeddings)
        """
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = dot_product / (norm1 * norm2)

        # Ensure result is in valid range
        return max(-1.0, min(1.0, float(cosine_sim)))


class AdaptiveSimilarityExplorer(RelationExplorationStrategy):
    """
    Adaptive similarity-based exploration with confidence filtering.

    Similar to CosineSimilarityExplorer but with additional intelligence:
    - Only explores memories with confidence below a threshold
    - Uses different relation types based on similarity ranges
    - Limits the number of relations per memory to prevent over-connecting

    Relation type mapping:
    - similarity >= 0.95: EQUIVALENT
    - similarity >= 0.85: SIMILAR
    - similarity >= threshold: RELATED
    """

    # Similarity thresholds for relation types
    EQUIVALENT_THRESHOLD = 0.95
    SIMILAR_THRESHOLD = 0.85

    def __init__(
        self,
        max_relations: int = 100,
        max_relations_per_memory: int = 5,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the adaptive similarity explorer.

        Args:
            max_relations: Maximum total relations to discover
            max_relations_per_memory: Maximum relations for any single memory
            confidence_threshold: Only explore memories with confidence below this
        """
        self.max_relations = max_relations
        self.max_relations_per_memory = max_relations_per_memory
        self.confidence_threshold = confidence_threshold

    async def explore(
        self,
        memories: List[Memory],
        similarity_threshold: float
    ) -> List[Tuple[Memory, Memory, RelationType]]:
        """
        Explore relations with adaptive strategy.

        Args:
            memories: List of memories to explore
            similarity_threshold: Minimum similarity for RELATED type

        Returns:
            List of (memory1, memory2, relation_type) tuples
        """
        if not memories:
            return []

        # Filter memories by confidence threshold
        candidate_memories = [
            m for m in memories
            if m.confidence < self.confidence_threshold
        ]

        if not candidate_memories:
            return []

        relations = []
        relations_count = {}  # Track relations per memory

        # Compare pairs
        for i, mem1 in enumerate(candidate_memories):
            if len(relations) >= self.max_relations:
                break

            # Check if this memory has too many relations already
            if relations_count.get(mem1.memory_id, 0) >= self.max_relations_per_memory:
                continue

            for mem2 in candidate_memories[i+1:]:
                if len(relations) >= self.max_relations:
                    break

                # Check if mem2 has too many relations
                if relations_count.get(mem2.memory_id, 0) >= self.max_relations_per_memory:
                    continue

                # Calculate similarity
                similarity = self._cosine_similarity(mem1.embedding, mem2.embedding)

                # Determine relation type based on similarity
                if similarity >= self.EQUIVALENT_THRESHOLD:
                    relation_type = RelationType.EQUIVALENT
                elif similarity >= self.SIMILAR_THRESHOLD:
                    relation_type = RelationType.SIMILAR
                elif similarity >= similarity_threshold:
                    relation_type = RelationType.RELATED
                else:
                    continue  # Below threshold, skip

                # Add relation
                relations.append((mem1, mem2, relation_type))

                # Update counts
                relations_count[mem1.memory_id] = relations_count.get(mem1.memory_id, 0) + 1
                relations_count[mem2.memory_id] = relations_count.get(mem2.memory_id, 0) + 1

        return relations

    def _cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = dot_product / (norm1 * norm2)
        return max(-1.0, min(1.0, float(cosine_sim)))
