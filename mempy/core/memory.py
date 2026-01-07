"""Core data classes for memory and relations."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class RelationType(Enum):
    """
    Memory relation types, designed based on relational algebra concepts.

    These relations model how memories connect to each other, supporting
    both semantic relationships and structural organization.
    """

    # Basic relations (core relational algebra operations)
    RELATED = "related"  # Correlation (Selection/Join semantics)
    EQUIVALENT = "equivalent"  # Equivalence (Union semantics)
    CONTRADICTORY = "contradictory"  # Contradiction (Difference semantics)

    # Hierarchical relations (Cartesian product decomposition)
    GENERALIZATION = "generalization"  # Generalization (superclass)
    SPECIALIZATION = "specialization"  # Specialization (subclass)
    PART_OF = "part_of"  # Composition / part-whole relation

    # Temporal / Causal
    PRECEDES = "precedes"  # Comes before (temporal sequence)
    FOLLOWS = "follows"  # Comes after
    CAUSES = "causes"  # Causality
    CAUSED_BY = "caused_by"  # Reverse causality

    # Semantic associations
    SIMILAR = "similar"  # Similarity
    PROPERTY_OF = "property_of"  # Attribute / property relation
    INSTANCE_OF = "instance_of"  # Instance relation
    CONTEXT_FOR = "context_for"  # Context / background relation


@dataclass
class Memory:
    """
    A memory unit with content, embedding, and metadata.

    Attributes:
        memory_id: Unique identifier for this memory
        content: The text content of the memory
        embedding: Vector embedding of the content
        user_id: Optional user identifier for scoping
        agent_id: Optional agent identifier for scoping
        run_id: Optional session/conversation identifier
        metadata: Additional flexible metadata
        created_at: Creation timestamp
        updated_at: Last update timestamp (None if never updated)
        priority: Importance score (0.0 to 1.0)
        confidence: Confidence score (0.0 to 1.0)
        last_accessed_at: Last access timestamp for tracking (None if never accessed)
        access_count: Number of times this memory has been accessed
        importance: Importance score for forgetting threshold (0.0 to 1.0)
    """

    memory_id: str
    content: str
    embedding: List[float]
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    priority: float = 0.5
    confidence: float = 1.0
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    importance: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "embedding": self.embedding,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "priority": self.priority,
            "confidence": self.confidence,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "access_count": self.access_count,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create from dictionary representation."""
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class Relation:
    """
    A relation between two memories.

    Attributes:
        relation_id: Unique identifier for this relation
        from_id: Source memory ID
        to_id: Target memory ID
        type: Relation type (RelationType enum)
        metadata: Additional metadata about the relation
        created_at: Creation timestamp
    """

    relation_id: str
    from_id: str
    to_id: str
    type: RelationType
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "relation_id": self.relation_id,
            "from_id": self.from_id,
            "to_id": self.to_id,
            "type": self.type.value if isinstance(self.type, RelationType) else self.type,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """Create from dictionary representation."""
        if isinstance(data.get("type"), str):
            data["type"] = RelationType(data["type"])
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class ProcessorResult:
    """
    Result from memory processor deciding what operation to perform.

    Attributes:
        action: The action to take ("add", "update", "delete", "none")
        memory_id: Target memory ID for update/delete operations
        content: New content for update operations
        reason: Explanation for the decision
    """

    action: str  # "add" | "update" | "delete" | "none"
    memory_id: Optional[str] = None
    content: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "action": self.action,
            "memory_id": self.memory_id,
            "content": self.content,
            "reason": self.reason,
        }
