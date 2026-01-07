"""Relation builders for automatic graph construction.

This module provides strategy classes for automatically constructing relations
between memories when new memories are added to the system.

RelationBuilders are invoked during the Memory.add() operation and differ from
RelationExplorationStrategy, which is used for batch exploration of existing
memories.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any

from mempy.core.memory import Memory, RelationType


class RelationBuilder(ABC):
    """Abstract base class for automatic relation construction.

    RelationBuilders are invoked automatically when new memories are added
    to the system, allowing for dynamic construction of graph relationships
    based on customizable strategies.

    This differs from RelationExplorationStrategy, which is designed for
    batch exploration and discovery of relations across existing memories.

    Example:
        >>> class MyBuilder(RelationBuilder):
        ...     async def build(self, new_memory, existing_memories):
        ...         # Custom relation building logic
        ...         return [(new_id, existing_id, RelationType.RELATED, None)]
    """

    @abstractmethod
    async def build(
        self,
        new_memory: Memory,
        existing_memories: List["Memory"]
    ) -> List[Tuple[str, str, RelationType, Optional[Dict[str, Any]]]]:
        """Build relations from new memory to existing memories.

        Args:
            new_memory: The newly added memory object
            existing_memories: List of existing memories in the user's collection

        Returns:
            List of tuples representing relations to create:
            (from_id, to_id, relation_type, metadata)

            - from_id: Source memory ID (should be new_memory.memory_id)
            - to_id: Target memory ID (from existing_memories)
            - relation_type: Type of relation (e.g., RelationType.RELATED)
            - metadata: Optional dict with additional information about the relation

        Example:
            >>> return [
            ...     ("mem_123", "mem_456", RelationType.SIMILAR, {"similarity": 0.95}),
            ...     ("mem_123", "mem_789", RelationType.RELATED, {"source": "semantic"}),
            ... ]
        """
        pass


class RandomRelationBuilder(RelationBuilder):
    """Random relation builder for testing and demonstration.

    This builder creates random relations between a new memory and existing
    memories, useful for testing graph functionality or demonstration purposes.

    Attributes:
        max_relations: Maximum number of relations to create per new memory
        relation_types: List of relation types to randomly choose from
        build_probability: Probability (0-1) that any relations will be built
        seed: Optional random seed for reproducibility

    Example:
        >>> from mempy.strategies import RandomRelationBuilder, RelationType
        >>>
        >>> # Create builder that creates 1-3 random relations 80% of the time
        >>> builder = RandomRelationBuilder(
        ...     max_relations=3,
        ...     relation_types=[RelationType.RELATED, RelationType.SIMILAR],
        ...     build_probability=0.8,
        ...     seed=42  # For reproducibility
        ... )
        >>>
        >>> # Use with Memory
        >>> from mempy import Memory, Embedder
        >>> memory = Memory(embedder=MyEmbedder(), relation_builder=builder)
        >>> await memory.add("I like blue", user_id="alice")
        # May automatically create random relations to existing memories
    """

    def __init__(
        self,
        max_relations: int = 3,
        relation_types: Optional[List[RelationType]] = None,
        build_probability: float = 1.0,
        seed: Optional[int] = None
    ):
        """Initialize the RandomRelationBuilder.

        Args:
            max_relations: Maximum number of relations to create
                (actual number will be random between 1 and max_relations)
            relation_types: List of relation types to choose from
                (defaults to [RelationType.RELATED])
            build_probability: Probability that relations will be built
                (0.0 = never, 1.0 = always)
            seed: Random seed for reproducible behavior
                (None for non-deterministic)

        Raises:
            ValueError: If max_relations < 1 or build_probability not in [0, 1]
        """
        if max_relations < 1:
            raise ValueError("max_relations must be at least 1")
        if not 0.0 <= build_probability <= 1.0:
            raise ValueError("build_probability must be between 0.0 and 1.0")

        self.max_relations = max_relations
        self.relation_types = relation_types or [RelationType.RELATED]
        self.build_probability = build_probability

        # Use instance-level random state to avoid interference between builders
        self._random = random.Random(seed)

    async def build(
        self,
        new_memory: Memory,
        existing_memories: List[Memory]
    ) -> List[Tuple[str, str, RelationType, Optional[Dict[str, Any]]]]:
        """Build random relations to existing memories.

        Args:
            new_memory: The newly added memory
            existing_memories: List of existing memories to potentially connect to

        Returns:
            List of relation tuples. May be empty if:
            - No existing memories
            - Probability check fails
            - max_relations is 0
        """
        # Probability check - skip building relations
        if self._random.random() > self.build_probability:
            return []

        # No existing memories to connect to
        if not existing_memories:
            return []

        # Determine number of relations (random between 1 and max_relations)
        num_relations = min(
            self.max_relations,
            len(existing_memories),
            self._random.randint(1, self.max_relations)
        )

        # Randomly select target memories
        selected = self._random.sample(existing_memories, num_relations)

        # Create relations
        relations = []
        for target in selected:
            relation_type = self._random.choice(self.relation_types)
            metadata = {
                "source": "RandomRelationBuilder",
                "random": True,
                "build_probability": self.build_probability
            }
            relations.append((
                new_memory.memory_id,
                target.memory_id,
                relation_type,
                metadata
            ))

        return relations

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RandomRelationBuilder("
            f"max_relations={self.max_relations}, "
            f"relation_types={len(self.relation_types)}, "
            f"probability={self.build_probability})"
        )
