"""NetworkX-based graph storage implementation."""

import fcntl
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

    Persistence Strategy:
    - By default (auto_save=False), persistence is manual via save() or context manager
    - Set auto_save=True to enable automatic saving after each write operation
    - Set save_interval > 0 to save every N operations (requires auto_save=True)
    - Set enable_file_lock=True for multi-process safety (Linux only)
    """

    def __init__(
        self,
        persist_path: Path,
        auto_save: bool = False,
        save_interval: int = 0,
        enable_file_lock: bool = False,
    ):
        """
        Initialize the NetworkX graph store.

        Args:
            persist_path: Directory path for persistence
            auto_save: If True, automatically save after write operations
            save_interval: If > 0, save every N operations (requires auto_save=True)
            enable_file_lock: If True, use fcntl file locks (Linux only)
        """
        self.persist_path = Path(persist_path)
        self.graph_path = self.persist_path / "graph.pkl"
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.enable_file_lock = enable_file_lock

        # State tracking for lazy persistence
        self._dirty = False
        self._pending_writes = 0

        # Load existing graph or create new one
        self.graph = self._load_graph()

    def _load_graph(self) -> nx.DiGraph:
        """Load graph from disk or create new one."""
        if self.graph_path.exists():
            try:
                with open(self.graph_path, "rb") as f:
                    if self.enable_file_lock:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for read
                    data = pickle.load(f)
                    if self.enable_file_lock:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return data
            except Exception as e:
                # If load fails, start fresh
                return nx.DiGraph()
        return nx.DiGraph()

    def _save_graph(self) -> None:
        """Save graph to disk with optional file locking."""
        try:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            with open(self.graph_path, "wb") as f:
                if self.enable_file_lock:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for write
                pickle.dump(self.graph, f)
                if self.enable_file_lock:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            self._dirty = False
            self._pending_writes = 0
        except Exception as e:
            raise StorageError(f"Failed to save graph: {e}") from e

    def _mark_dirty(self) -> None:
        """
        Mark the graph as dirty and trigger save if conditions are met.

        This method is called internally after write operations.
        The actual save occurs based on auto_save and save_interval settings.
        """
        self._dirty = True
        self._pending_writes += 1

        if self.auto_save:
            if self.save_interval == 0:
                # Save immediately after each operation
                self._save_graph()
            elif self._pending_writes >= self.save_interval:
                # Save after N operations
                self._save_graph()

    async def save(self) -> None:
        """
        Manually trigger graph persistence to disk.

        This method is useful when auto_save=False. It writes the current
        graph state to disk, including all nodes and edges.
        """
        self._save_graph()

    async def __aenter__(self):
        """Context manager entry - return self."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save pending changes."""
        if self._dirty:
            self._save_graph()
        return False

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
            "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else None,
            "access_count": memory.access_count,
            "importance": memory.importance,
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

        # Parse last_accessed_at if present
        last_accessed_at = None
        if attrs.get("last_accessed_at"):
            try:
                last_accessed_at = datetime.fromisoformat(attrs["last_accessed_at"])
            except (ValueError, TypeError):
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
            last_accessed_at=last_accessed_at,
            access_count=attrs.get("access_count", 0),
            importance=attrs.get("importance", 0.5),
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
            self._mark_dirty()
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
            self._mark_dirty()
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
                self._mark_dirty()
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
            self._mark_dirty()
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

        if direction in ["out", "both"]:
            # Get direct outgoing neighbors
            if max_depth == 1:
                for _, v, data in self.graph.out_edges(memory_id, data=True):
                    rel = Relation(
                        relation_id=f"{memory_id}-{v}",
                        from_id=memory_id,
                        to_id=v,
                        type=RelationType(data["type"]),
                        metadata=json.loads(data.get("metadata", "{}"))
                    )
                    relations.append(rel)
            else:
                # Traverse up to max_depth using BFS
                visited = set()
                current_level = {memory_id}
                for _ in range(max_depth):
                    next_level = set()
                    for node in current_level:
                        for _, v, data in self.graph.out_edges(node, data=True):
                            if v not in visited:
                                rel = Relation(
                                    relation_id=f"{node}-{v}",
                                    from_id=node,
                                    to_id=v,
                                    type=RelationType(data["type"]),
                                    metadata=json.loads(data.get("metadata", "{}"))
                                )
                                relations.append(rel)
                                visited.add(v)
                                next_level.add(v)
                    current_level = next_level

        if direction in ["in", "both"]:
            # Get direct incoming neighbors
            if max_depth == 1:
                for u, _, data in self.graph.in_edges(memory_id, data=True):
                    rel = Relation(
                        relation_id=f"{u}-{memory_id}",
                        from_id=u,
                        to_id=memory_id,
                        type=RelationType(data["type"]),
                        metadata=json.loads(data.get("metadata", "{}"))
                    )
                    relations.append(rel)
            else:
                # Traverse up to max_depth using BFS
                visited = set()
                current_level = {memory_id}
                for _ in range(max_depth):
                    next_level = set()
                    for node in current_level:
                        for u, _, data in self.graph.in_edges(node, data=True):
                            if u not in visited:
                                rel = Relation(
                                    relation_id=f"{u}-{node}",
                                    from_id=u,
                                    to_id=node,
                                    type=RelationType(data["type"]),
                                    metadata=json.loads(data.get("metadata", "{}"))
                                )
                                relations.append(rel)
                                visited.add(u)
                                next_level.add(u)
                    current_level = next_level

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
            self._mark_dirty()
        except Exception as e:
            raise StorageError(f"Failed to reset graph: {e}") from e
