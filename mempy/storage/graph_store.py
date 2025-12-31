"""NetworkX-based graph storage implementation."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from mempy.core.memory import Memory, Relation, RelationType
from mempy.core.exceptions import StorageError


class NetworkXGraphStore:
    """
    NetworkX-based graph storage for memory relations.

    This store handles:
    - Memory nodes with attributes
    - Relations between memories
    - Graph traversal (BFS/DFS)
    - Persistence to disk
    """

    def __init__(self, persist_path: Path):
        """
        Initialize the NetworkX graph store.

        Args:
            persist_path: Directory path for persistence
        """
        self.persist_path = Path(persist_path)
        self.graph_path = self.persist_path / "graph.pkl"

        # Load existing graph or create new one
        self.graph = self._load_graph()

    def _load_graph(self) -> nx.DiGraph:
        """Load graph from disk or create new one."""
        if self.graph_path.exists():
            try:
                with open(self.graph_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                # If load fails, start fresh
                return nx.DiGraph()
        return nx.DiGraph()

    def _save_graph(self) -> None:
        """Save graph to disk."""
        try:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            with open(self.graph_path, "wb") as f:
                pickle.dump(self.graph, f)
        except Exception as e:
            raise StorageError(f"Failed to save graph: {e}") from e

    def _memory_to_node_attrs(self, memory: Memory) -> Dict[str, Any]:
        """Convert Memory to node attributes."""
        return {
            "memory_id": memory.memory_id,
            "content": memory.content,
            "user_id": memory.user_id,
            "agent_id": memory.agent_id,
            "run_id": memory.run_id,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
            "priority": memory.priority,
            "confidence": memory.confidence,
            "metadata": json.dumps(memory.metadata),
        }

    def _node_attrs_to_memory(self, memory_id: str, attrs: Dict[str, Any]) -> Memory:
        """Convert node attributes back to Memory."""
        from datetime import datetime

        metadata = {}
        if "metadata" in attrs:
            try:
                metadata = json.loads(attrs["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

        return Memory(
            memory_id=memory_id,
            content=attrs.get("content", ""),
            embedding=[],  # Embeddings stored separately in vector store
            user_id=attrs.get("user_id"),
            agent_id=attrs.get("agent_id"),
            run_id=attrs.get("run_id"),
            metadata=metadata,
            created_at=datetime.fromisoformat(attrs["created_at"]) if attrs.get("created_at") else None,
            updated_at=datetime.fromisoformat(attrs["updated_at"]) if attrs.get("updated_at") else None,
            priority=attrs.get("priority", 0.5),
            confidence=attrs.get("confidence", 1.0),
        )

    async def add_node(self, memory_id: str, memory: Memory) -> None:
        """
        Add a memory as a node in the graph.

        Args:
            memory_id: The memory ID
            memory: The memory object

        Raises:
            StorageError: If add fails
        """
        try:
            self.graph.add_node(memory_id, **self._memory_to_node_attrs(memory))
            self._save_graph()
        except Exception as e:
            raise StorageError(f"Failed to add node to graph: {e}") from e

    async def get_node(self, memory_id: str) -> Optional[Memory]:
        """
        Get a memory node by ID.

        Args:
            memory_id: The memory ID

        Returns:
            The memory if found, None otherwise
        """
        if memory_id not in self.graph.nodes:
            return None

        attrs = self.graph.nodes[memory_id]
        return self._node_attrs_to_memory(memory_id, attrs)

    async def update_node(self, memory_id: str, memory: Memory) -> None:
        """
        Update a memory node.

        Args:
            memory_id: The memory ID
            memory: The updated memory

        Raises:
            StorageError: If update fails or node not found
        """
        if memory_id not in self.graph.nodes:
            raise StorageError(f"Memory node {memory_id} not found in graph")

        try:
            # Update node attributes
            for key, value in self._memory_to_node_attrs(memory).items():
                self.graph.nodes[memory_id][key] = value
            self._save_graph()
        except Exception as e:
            raise StorageError(f"Failed to update node: {e}") from e

    async def delete_node(self, memory_id: str) -> None:
        """
        Delete a memory node from the graph.

        Args:
            memory_id: The memory ID

        Raises:
            StorageError: If delete fails
        """
        try:
            if memory_id in self.graph.nodes:
                self.graph.remove_node(memory_id)
                self._save_graph()
        except Exception as e:
            raise StorageError(f"Failed to delete node: {e}") from e

    async def add_edge(
        self,
        from_id: str,
        to_id: str,
        relation_type: RelationType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a relation edge between two memories.

        Args:
            from_id: Source memory ID
            to_id: Target memory ID
            relation_type: Type of relation
            metadata: Optional metadata

        Raises:
            StorageError: If add fails
        """
        try:
            # Ensure nodes exist
            if from_id not in self.graph.nodes:
                raise StorageError(f"Source memory {from_id} not found")
            if to_id not in self.graph.nodes:
                raise StorageError(f"Target memory {to_id} not found")

            edge_attrs = {
                "type": relation_type.value if isinstance(relation_type, RelationType) else relation_type,
                "metadata": json.dumps(metadata or {}),
            }
            self.graph.add_edge(from_id, to_id, **edge_attrs)
            self._save_graph()
        except Exception as e:
            raise StorageError(f"Failed to add edge: {e}") from e

    async def get_relations(
        self,
        memory_id: str,
        direction: str = "both",
        max_depth: int = 1
    ) -> List[Relation]:
        """
        Get relations for a memory with optional graph traversal.

        Args:
            memory_id: The memory ID
            direction: "out" (outgoing), "in" (incoming), or "both"
            max_depth: Maximum depth to traverse (1 = direct neighbors only)

        Returns:
            List of relations
        """
        if memory_id not in self.graph.nodes:
            return []

        relations = []
        seen = set()

        if direction in ["out", "both"]:
            # Traverse outgoing edges
            for target in nx.descendants_at_distance(self.graph, memory_id, max_depth):
                if target not in seen:
                    for _, v, data in self.graph.out_edges(memory_id, data=True):
                        if max_depth == 1 or nx.has_path(self.graph, memory_id, v):
                            rel = Relation(
                                relation_id=f"{memory_id}-{v}",
                                from_id=memory_id,
                                to_id=v,
                                type=RelationType(data["type"]),
                                metadata=json.loads(data.get("metadata", "{}"))
                            )
                            relations.append(rel)
                            seen.add(v)

        if direction in ["in", "both"]:
            # Traverse incoming edges
            for source in nx.ancestors_at_distance(self.graph, memory_id, max_depth):
                if source not in seen:
                    for u, _, data in self.graph.in_edges(memory_id, data=True):
                        if max_depth == 1 or nx.has_path(self.graph, u, memory_id):
                            rel = Relation(
                                relation_id=f"{u}-{memory_id}",
                                from_id=u,
                                to_id=memory_id,
                                type=RelationType(data["type"]),
                                metadata=json.loads(data.get("metadata", "{}"))
                            )
                            relations.append(rel)
                            seen.add(u)

        return relations

    async def get_neighbors(
        self,
        memory_id: str,
        direction: str = "both"
    ) -> List[str]:
        """
        Get direct neighbor memory IDs.

        Args:
            memory_id: The memory ID
            direction: "out", "in", or "both"

        Returns:
            List of neighbor memory IDs
        """
        if memory_id not in self.graph.nodes:
            return []

        neighbors = set()

        if direction in ["out", "both"]:
            neighbors.update(self.graph.successors(memory_id))

        if direction in ["in", "both"]:
            neighbors.update(self.graph.predecessors(memory_id))

        return list(neighbors)

    async def find_path(self, from_id: str, to_id: str) -> Optional[List[str]]:
        """
        Find shortest path between two memories.

        Args:
            from_id: Source memory ID
            to_id: Target memory ID

        Returns:
            List of memory IDs forming the path, or None if no path exists
        """
        try:
            if from_id not in self.graph.nodes or to_id not in self.graph.nodes:
                return None
            return nx.shortest_path(self.graph, from_id, to_id)
        except nx.NetworkXNoPath:
            return None

    async def reset(self) -> None:
        """
        Reset the graph, deleting all data.

        WARNING: This is irreversible.
        """
        try:
            self.graph.clear()
            self._save_graph()
        except Exception as e:
            raise StorageError(f"Failed to reset graph: {e}") from e
