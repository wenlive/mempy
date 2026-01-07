"""Main Memory API - the primary user interface for mempy."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from mempy.core.interfaces import Embedder, MemoryProcessor
from mempy.core.memory import Memory as MemoryData, RelationType
from mempy.core.exceptions import StorageError
from mempy.storage.backend import DualStorageBackend
from mempy.config import get_storage_path


@dataclass
class _ProcessorDecision:
    """Internal decision result from processor strategy.

    Attributes:
        action: The action to take ("add", "update", "delete", "none")
        memory_id: Target memory ID for update/delete operations
        content: Modified content (for update operations)
    """
    action: str  # "add", "update", "delete", "none"
    memory_id: Optional[str]
    content: Optional[str]


class Memory:
    """
    Main Memory class - the primary user interface for mempy.

    This class provides a pluggable strategy system for advanced memory management:
    - add(): Add new memories with optional strategy support
    - search(): Semantic search through memories
    - get(), get_all(): Retrieve memories
    - update(): Update existing memories
    - delete(), delete_all(): Delete memories
    - add_relation(), get_relations(): Manage memory relations
    - reset(): Clear all data

    Strategy Pipeline:
        When adding memories, the system follows a three-stage pipeline:
        1. Ingest Strategy (processor): Decide what to do with new content
        2. Storage: Save the memory to vector and graph stores
        3. Graph Strategy (relation_builder): Build relations automatically

    Strategies are optional and can be plugged in for advanced behavior.

    Example:
        ```python
        import mempy
        from mempy.strategies import RandomRelationBuilder, RelationType

        # Implement embedder (user must provide dimension)
        class MyEmbedder(mempy.Embedder):
            def __init__(self):
                self._dimension = 768  # Must declare dimension

            @property
            def dimension(self) -> int:
                return self._dimension

            async def embed(self, text: str) -> List[float]:
                # Call your LLM service
                return await my_llm.embed(text)

        # Basic usage (no strategies)
        memory = mempy.Memory(embedder=MyEmbedder(), verbose=True)
        await memory.add("I like blue", user_id="alice")

        # With graph strategy (automatic relation building)
        memory = mempy.Memory(
            embedder=MyEmbedder(),
            relation_builder=RandomRelationBuilder(max_relations=3)
        )
        await memory.add("I like blue", user_id="alice")
        # Relations are automatically built!

        # Search
        results = await memory.search("color preference", user_id="alice")
        ```
    """

    def __init__(
        self,
        embedder: Embedder,
        processor: Optional[MemoryProcessor] = None,
        relation_builder: Optional["RelationBuilder"] = None,
        storage_path: Optional[str] = None,
        verbose: bool = False,
        # Evolution strategies (optional, use defaults if not provided)
        confidence_strategy=None,
        firmness_calculator=None
    ):
        """
        Initialize the Memory instance.

        Args:
            embedder: User-provided embedder (must declare dimension property)
            processor: Optional LLM processor for intelligent memory operations
            relation_builder: Optional relation builder for automatic graph construction
            storage_path: Optional custom storage path (default: ~/.mempy/data)
            verbose: Enable verbose logging output
            confidence_strategy: Optional custom confidence evolution strategy
            firmness_calculator: Optional custom firmness calculator
        """
        self.embedder = embedder
        self.processor = processor
        self.relation_builder = relation_builder
        self.verbose = verbose

        # Setup storage
        path = get_storage_path(storage_path)
        self.storage = DualStorageBackend(path)

        # Setup logging
        self._setup_logging()

        # Setup evolution strategies (lazy import to avoid circular dependency)
        if confidence_strategy is None or firmness_calculator is None:
            from mempy.strategies import (
                SimpleConfidenceStrategy,
                WeightedFirmnessCalculator
            )

        self.confidence_strategy = confidence_strategy or SimpleConfidenceStrategy()
        self.firmness_calculator = firmness_calculator or WeightedFirmnessCalculator()

    def _setup_logging(self):
        """Setup logging for observability."""
        self.logger = logging.getLogger("mempy")

        if self.verbose and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[mempy] %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def _apply_processor_strategy(
        self,
        content: str,
        user_id: Optional[str]
    ) -> _ProcessorDecision:
        """Apply processor strategy to decide memory operation.

        This method encapsulates all processor logic, making the strategy
        call explicit and pluggable.

        Args:
            content: The content to process
            user_id: Optional user identifier for scoping

        Returns:
            _ProcessorDecision with action, memory_id, and content
        """
        if not self.processor:
            return _ProcessorDecision(action="add", memory_id=None, content=content)

        try:
            existing = await self.search(content, user_id=user_id, limit=5)
            result = await self.processor.process(content, existing)

            if self.verbose:
                self.logger.info(f"[PROCESSOR] Decision: {result.action}")

            return _ProcessorDecision(
                action=result.action,
                memory_id=result.memory_id,
                content=result.content
            )
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"[PROCESSOR] Failed: {e}, defaulting to ADD")
            return _ProcessorDecision(action="add", memory_id=None, content=content)

    async def _apply_relation_builder_strategy(
        self,
        memory: MemoryData,
        user_id: Optional[str]
    ) -> int:
        """Apply relation builder strategy to construct graph relations.

        This method encapsulates all relation building logic, making the
        strategy call explicit and pluggable.

        Args:
            memory: The newly added memory object
            user_id: Optional user identifier for scoping

        Returns:
            Number of relations created
        """
        if not self.relation_builder:
            return 0

        try:
            # Search for existing memories to connect to
            existing = await self.search(
                memory.content,
                user_id=user_id,
                limit=20
            )
            existing = [m for m in existing if m.memory_id != memory.memory_id]

            if not existing:
                return 0

            # Build relations using the configured strategy
            relations = await self.relation_builder.build(memory, existing)

            # Create the relations
            created = 0
            for from_id, to_id, rel_type, metadata in relations:
                try:
                    await self.add_relation(from_id, to_id, rel_type, metadata)
                    created += 1
                except Exception as e:
                    if self.verbose:
                        self.logger.warning(f"[RELATIONS] Failed to add relation: {e}")

            if self.verbose and created > 0:
                self.logger.info(f"[RELATIONS] Created {created} relations")

            return created

        except Exception as e:
            if self.verbose:
                self.logger.warning(f"[RELATIONS] Strategy failed: {e}")
            return 0

    async def add(
        self,
        content: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None
    ) -> Optional[str]:
        """
        Add a new memory with pluggable strategy support.

        Strategy Pipeline:
            1. Ingest Strategy (processor): Decide what to do with new content
            2. Storage: Save the memory
            3. Graph Strategy (relation_builder): Build relations

        Args:
            content: The memory content
            user_id: Optional user identifier for scoping
            agent_id: Optional agent identifier for scoping
            run_id: Optional session/conversation identifier
            metadata: Optional additional metadata
            importance: Optional importance score (0.0-1.0, default: 0.5)

        Returns:
            The memory ID if successful, None if skipped

        Raises:
            StorageError: If storage operation fails
            EmbedderError: If embedding generation fails
        """
        if self.verbose:
            self.logger.info(f"Processing: {content[:50]}{'...' if len(content) > 50 else ''}")

        # ========== STRATEGY 1: Ingest (Processor) ==========
        decision = await self._apply_processor_strategy(content, user_id)

        # Handle non-add decisions
        if decision.action == "update" and decision.memory_id:
            memory_id = await self.update(decision.memory_id, decision.content or content)
            if self.verbose:
                self.logger.info(f"Updated memory: {memory_id}")
            return memory_id

        elif decision.action == "delete" and decision.memory_id:
            await self.delete(decision.memory_id)
            if self.verbose:
                self.logger.info(f"Deleted memory: {decision.memory_id}")
            return None

        elif decision.action == "none":
            if self.verbose:
                self.logger.info("Skipped (processor decision)")
            return None

        # ========== STORAGE: Save Memory ==========
        embedding = await self.embedder.embed(decision.content or content)

        memory = MemoryData(
            memory_id=uuid4().hex,
            content=decision.content or content,
            embedding=embedding,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata or {},
            importance=importance if importance is not None else 0.5
        )

        memory_id = await self.storage.add(memory)

        if self.verbose:
            self.logger.info(f"Saved: {memory_id}")

        # ========== STRATEGY 2: Graph (RelationBuilder) ==========
        await self._apply_relation_builder_strategy(memory, user_id)

        return memory_id

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryData]:
        """
        Search for memories by semantic similarity.

        Args:
            query: The search query text
            user_id: Optional user filter
            agent_id: Optional agent filter
            limit: Maximum number of results

        Returns:
            List of memories ranked by similarity
        """
        query_vector = await self.embedder.embed(query)

        filters = {}
        if user_id is not None:
            filters["user_id"] = user_id
        if agent_id is not None:
            filters["agent_id"] = agent_id

        results = await self.storage.search(query_vector, filters, limit)

        if self.verbose:
            self.logger.info(f"Found {len(results)} memories for: {query[:50]}")

        return results

    async def get(self, memory_id: str) -> Optional[MemoryData]:
        """
        Get a specific memory by ID.

        Automatically updates access statistics (hotness tracking).

        Args:
            memory_id: The memory ID

        Returns:
            The memory if found, None otherwise
        """
        memory = await self.storage.get(memory_id)
        if memory:
            # Update access statistics
            memory.last_accessed_at = datetime.utcnow()
            memory.access_count += 1
            await self.storage.update(memory_id, memory)
        return memory

    async def get_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MemoryData]:
        """
        Get all memories matching filters.

        Args:
            user_id: Optional user filter
            agent_id: Optional agent filter
            limit: Optional maximum number of results

        Returns:
            List of matching memories
        """
        filters = {}
        if user_id is not None:
            filters["user_id"] = user_id
        if agent_id is not None:
            filters["agent_id"] = agent_id

        return await self.storage.get_all(filters, limit)

    async def update(self, memory_id: str, content: str) -> str:
        """
        Update an existing memory's content.

        Args:
            memory_id: The ID of the memory to update
            content: The new content

        Returns:
            The memory ID

        Raises:
            StorageError: If memory not found or update fails
        """
        existing = await self.storage.get(memory_id)
        if existing is None:
            raise StorageError(f"Memory {memory_id} not found")

        # Generate new embedding
        embedding = await self.embedder.embed(content)

        # Update memory object
        existing.content = content
        existing.embedding = embedding
        existing.updated_at = datetime.utcnow()

        # Save to storage
        await self.storage.update(memory_id, existing)

        if self.verbose:
            self.logger.info(f"Updated: {memory_id}")

        return memory_id

    async def delete(self, memory_id: str) -> None:
        """
        Delete a memory by ID.

        Args:
            memory_id: The ID of the memory to delete

        Raises:
            StorageError: If deletion fails
        """
        await self.storage.delete(memory_id)

        if self.verbose:
            self.logger.info(f"Deleted: {memory_id}")

    async def delete_all(self, user_id: str) -> None:
        """
        Delete all memories for a user.

        Args:
            user_id: The user ID

        Raises:
            StorageError: If deletion fails
        """
        await self.storage.delete_all({"user_id": user_id})

        if self.verbose:
            self.logger.info(f"Deleted all memories for user: {user_id}")

    async def add_relation(
        self,
        from_id: str,
        to_id: str,
        relation_type: RelationType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a relation between two memories.

        Automatically reinforces confidence for both memories when a relation is created.

        Args:
            from_id: Source memory ID
            to_id: Target memory ID
            relation_type: Type of relation
            metadata: Optional metadata about the relation

        Raises:
            StorageError: If relation cannot be added
        """
        await self.storage.add_relation(from_id, to_id, relation_type, metadata)

        # Reinforce confidence for both memories
        try:
            await self.reinforce_confidence(from_id, reason="relation")
            await self.reinforce_confidence(to_id, reason="relation")
        except StorageError:
            # If reinforcement fails, still log the relation was added
            pass

        if self.verbose:
            self.logger.info(
                f"Added relation: {from_id} --[{relation_type.value}]--> {to_id}"
            )

    async def get_relations(
        self,
        memory_id: str,
        direction: str = "both",
        max_depth: int = 2
    ) -> List:
        """
        Get relations for a memory.

        Args:
            memory_id: The memory ID
            direction: "out", "in", or "both"
            max_depth: Maximum depth to traverse in graph

        Returns:
            List of relations
        """
        return await self.storage.get_relations(memory_id, direction, max_depth)

    async def reset(self) -> None:
        """
        Reset all data, clearing all memories and relations.

        WARNING: This operation is irreversible.
        """
        await self.storage.reset()

        if self.verbose:
            self.logger.info("Reset complete - all data cleared")

    # ========================================================================
    # Memory Evolution Methods
    # ========================================================================

    async def evolve_confidence(
        self,
        decay_rate: float = 0.01,
        days_threshold: int = 30,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Manually trigger confidence time decay for long-unaccessed memories.

        Args:
            decay_rate: Daily decay rate (default: 0.01 = 1% per day)
            days_threshold: Only decay memories not accessed in this many days
            user_id: Optional user filter (None = all users)

        Returns:
            Dict with statistics:
                - updated_count: Number of memories updated
                - avg_decay: Average confidence decay applied
                - total_count: Total memories processed
        """
        from datetime import timedelta

        # Get all memories
        filters = {"user_id": user_id} if user_id else {}
        memories = await self.storage.get_all(filters, limit=None)

        if not memories:
            return {
                "updated_count": 0,
                "avg_decay": 0.0,
                "total_count": 0
            }

        updated_count = 0
        total_decay = 0.0
        threshold_time = datetime.utcnow() - timedelta(days=days_threshold)

        for memory in memories:
            # Skip if recently accessed
            if memory.last_accessed_at and memory.last_accessed_at > threshold_time:
                continue

            # Calculate days since last access (or since creation)
            reference_time = memory.last_accessed_at or memory.created_at
            days_since_access = (datetime.utcnow() - reference_time).days

            if days_since_access <= 0:
                continue

            # Calculate decay using strategy
            decay = await self.confidence_strategy.decay_over_time(memory, days_since_access)

            # Apply decay
            if decay > 0:
                memory.confidence = max(0.0, memory.confidence - decay)
                await self.storage.update(memory.memory_id, memory)
                updated_count += 1
                total_decay += decay

        avg_decay = total_decay / updated_count if updated_count > 0 else 0.0

        if self.verbose:
            self.logger.info(
                f"Confidence evolution: {updated_count}/{len(memories)} updated, "
                f"avg decay: {avg_decay:.3f}"
            )

        return {
            "updated_count": updated_count,
            "avg_decay": avg_decay,
            "total_count": len(memories)
        }

    async def reinforce_confidence(
        self,
        memory_id: str,
        reason: str = "manual"
    ) -> float:
        """
        Reinforce confidence for a specific memory.

        Args:
            memory_id: The memory ID to reinforce
            reason: Reason for reinforcement ("reference", "relation", or "manual")

        Returns:
            The new confidence value

        Raises:
            StorageError: If memory not found
        """
        memory = await self.storage.get(memory_id)
        if memory is None:
            raise StorageError(f"Memory {memory_id} not found")

        # Calculate increment based on reason
        if reason == "reference":
            increment = await self.confidence_strategy.reinforce_on_reference(memory)
        elif reason == "relation":
            increment = await self.confidence_strategy.reinforce_on_relation(memory)
        else:  # manual
            # Use reference increment as default for manual reinforcement
            increment = await self.confidence_strategy.reinforce_on_reference(memory)

        # Apply increment (cap at 1.0)
        old_confidence = memory.confidence
        memory.confidence = min(1.0, memory.confidence + increment)
        await self.storage.update(memory_id, memory)

        if self.verbose:
            self.logger.info(
                f"Reinforced {memory_id}: {old_confidence:.3f} -> {memory.confidence:.3f} "
                f"(+{increment:.3f}, reason: {reason})"
            )

        return memory.confidence

    async def cleanup_forgotten(
        self,
        threshold: float = 0.3,
        dry_run: bool = False,
        user_id: Optional[str] = None,
        threshold_strategy=None
    ) -> Dict[str, Any]:
        """
        Clean up memories with low firmness (forgotten memories).

        Args:
            threshold: Firmness threshold below which memories are deleted
            dry_run: If True, only report what would be deleted without actually deleting
            user_id: Optional user filter (None = all users)
            threshold_strategy: Optional custom forgetting threshold strategy

        Returns:
            Dict with statistics:
                - deleted_count: Number of memories deleted (or would be deleted in dry_run)
                - deleted_ids: List of memory IDs that were/would be deleted
                - preserved_count: Number of memories preserved
                - avg_firmness: Average firmness of all memories
        """
        from mempy.strategies import FixedThresholdStrategy

        # Use provided strategy or default
        forgetting_strategy = threshold_strategy or FixedThresholdStrategy(threshold)

        # Get all memories
        filters = {"user_id": user_id} if user_id else {}
        memories = await self.storage.get_all(filters, limit=None)

        if not memories:
            return {
                "deleted_count": 0,
                "deleted_ids": [],
                "preserved_count": 0,
                "avg_firmness": 0.0
            }

        # Calculate firmness for each memory and decide which to delete
        to_delete = []
        firmness_sum = 0.0

        for memory in memories:
            # Get relations for this memory
            try:
                relations = await self.storage.get_relations(memory.memory_id, "both", max_depth=1)
                relation_count = len(relations)

                # Calculate average confidence of related memories
                if relation_count > 0:
                    related_confidences = []
                    for rel in relations:
                        # Get the related memory (either source or target)
                        related_id = rel.to_id if rel.from_id == memory.memory_id else rel.from_id
                        related_mem = await self.storage.get(related_id)
                        if related_mem:
                            related_confidences.append(related_mem.confidence)

                    avg_relation_confidence = sum(related_confidences) / len(related_confidences) if related_confidences else 0.0
                else:
                    avg_relation_confidence = 0.0

            except Exception:
                # If getting relations fails, assume no relations
                relation_count = 0
                avg_relation_confidence = 0.0

            # Calculate firmness
            firmness = self.firmness_calculator.calculate(
                memory=memory,
                relation_count=relation_count,
                avg_relation_confidence=avg_relation_confidence
            )
            firmness_sum += firmness

            # Check if should forget
            if forgetting_strategy.should_forget(memory, firmness):
                to_delete.append(memory.memory_id)

        avg_firmness = firmness_sum / len(memories) if memories else 0.0

        if dry_run:
            if self.verbose:
                self.logger.info(
                    f"Dry run: would delete {len(to_delete)}/{len(memories)} memories "
                    f"(threshold: {threshold}, avg firmness: {avg_firmness:.3f})"
                )

            return {
                "deleted_count": len(to_delete),
                "deleted_ids": to_delete,
                "preserved_count": len(memories) - len(to_delete),
                "avg_firmness": avg_firmness
            }

        # Actually delete the memories
        for memory_id in to_delete:
            await self.storage.delete(memory_id)

        if self.verbose:
            self.logger.info(
                f"Deleted {len(to_delete)}/{len(memories)} forgotten memories "
                f"(threshold: {threshold}, avg firmness: {avg_firmness:.3f})"
            )

        return {
            "deleted_count": len(to_delete),
            "deleted_ids": to_delete,
            "preserved_count": len(memories) - len(to_delete),
            "avg_firmness": avg_firmness
        }

    async def explore_relations(
        self,
        confidence_threshold: float = 0.5,
        max_new_relations: int = 10,
        similarity_threshold: float = 0.8,
        user_id: Optional[str] = None,
        explorer=None
    ) -> Dict[str, Any]:
        """
        Explore and discover potential relations between memories (subconscious exploration).

        This method finds memories with low confidence and tries to establish connections
        to other memories based on semantic similarity, then reinforces their confidence.

        Args:
            confidence_threshold: Only explore memories with confidence below this threshold
            max_new_relations: Maximum number of new relations to create
            similarity_threshold: Minimum similarity for establishing a relation
            user_id: Optional user filter (None = all users)
            explorer: Optional custom relation exploration strategy

        Returns:
            Dict with statistics:
                - processed_count: Number of memories processed
                - new_relations: Number of new relations created
                - confidence_boosted: Number of memories whose confidence was boosted
        """
        from mempy.strategies import CosineSimilarityExplorer

        # Use provided explorer or default
        exploration_strategy = explorer or CosineSimilarityExplorer(max_new_relations)

        # Get candidate memories (low confidence)
        filters = {"user_id": user_id} if user_id else {}
        all_memories = await self.storage.get_all(filters, limit=None)

        # Filter by confidence threshold
        candidate_memories = [
            m for m in all_memories
            if m.confidence < confidence_threshold
        ]

        if not candidate_memories:
            return {
                "processed_count": 0,
                "new_relations": 0,
                "confidence_boosted": 0
            }

        # Explore for new relations
        new_relations = await exploration_strategy.explore(
            candidate_memories,
            similarity_threshold
        )

        # Create the new relations and boost confidence
        relations_created = 0
        memories_boosted = set()

        for mem1, mem2, relation_type in new_relations:
            try:
                # Check if relation already exists
                existing_relations = await self.storage.get_relations(mem1.memory_id, "out", max_depth=1)
                already_exists = any(
                    r.to_id == mem2.memory_id and r.type == relation_type
                    for r in existing_relations
                )

                if not already_exists:
                    # Create the relation
                    await self.add_relation(mem1.memory_id, mem2.memory_id, relation_type)
                    relations_created += 1

                    # Track boosted memories
                    memories_boosted.add(mem1.memory_id)
                    memories_boosted.add(mem2.memory_id)

                    # Stop if we've reached the max
                    if relations_created >= max_new_relations:
                        break

            except Exception as e:
                # If creating a relation fails, continue with others
                if self.verbose:
                    self.logger.warning(f"Failed to create relation: {e}")
                continue

        if self.verbose:
            self.logger.info(
                f"Exploration: processed {len(candidate_memories)} memories, "
                f"created {relations_created} new relations, "
                f"boosted {len(memories_boosted)} memories"
            )

        return {
            "processed_count": len(candidate_memories),
            "new_relations": relations_created,
            "confidence_boosted": len(memories_boosted)
        }

    async def compress_memories(
        self,
        start_date: datetime,
        end_date: datetime,
        summary: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> MemoryData:
        """
        Compress memories within a time range into a single summary memory.

        Original memories are marked as compressed (metadata["compressed"] = True)
        but not deleted.

        Args:
            start_date: Start of time range (inclusive)
            end_date: End of time range (inclusive)
            summary: User-provided summary of the memories
            user_id: Optional user filter
            agent_id: Optional agent filter

        Returns:
            The newly created compressed memory

        Raises:
            StorageError: If no memories found in time range
        """
        # Find memories in time range
        filters = {}
        if user_id is not None:
            filters["user_id"] = user_id
        if agent_id is not None:
            filters["agent_id"] = agent_id

        all_memories = await self.storage.get_all(filters, limit=None)

        # Filter by time range and not already compressed
        memories_in_range = [
            m for m in all_memories
            if start_date <= m.created_at <= end_date
            and not m.metadata.get("compressed", False)
        ]

        if not memories_in_range:
            raise StorageError(f"No memories found in time range {start_date} to {end_date}")

        # Generate embedding for summary
        summary_embedding = await self.embedder.embed(summary)

        # Create compressed memory
        compressed_memory = MemoryData(
            memory_id=uuid4().hex,
            content=summary,
            embedding=summary_embedding,
            user_id=user_id,
            agent_id=agent_id,
            metadata={
                "compressed": True,
                "time_range_start": start_date.isoformat(),
                "time_range_end": end_date.isoformat(),
                "source_memory_count": len(memories_in_range)
            }
        )

        # Save compressed memory
        await self.storage.add(compressed_memory)

        # Mark original memories as compressed
        for memory in memories_in_range:
            memory.metadata["compressed"] = True
            memory.metadata["compressed_into"] = compressed_memory.memory_id
            await self.storage.update(memory.memory_id, memory)

        if self.verbose:
            self.logger.info(
                f"Compressed {len(memories_in_range)} memories into {compressed_memory.memory_id} "
                f"({start_date.date()} to {end_date.date()})"
            )

        return compressed_memory

    async def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        report_type: str = "summary"
    ) -> str:
        """
        Generate a report of memories within a time range.

        Args:
            start_date: Start of time range (inclusive)
            end_date: End of time range (inclusive)
            user_id: Optional user filter
            agent_id: Optional agent filter
            report_type: Type of report ("summary", "detailed", or "timeline")

        Returns:
            A string report of the memories
        """
        # Find memories in time range
        filters = {}
        if user_id is not None:
            filters["user_id"] = user_id
        if agent_id is not None:
            filters["agent_id"] = agent_id

        all_memories = await self.storage.get_all(filters, limit=None)

        # Filter by time range
        memories_in_range = [
            m for m in all_memories
            if start_date <= m.created_at <= end_date
        ]

        if not memories_in_range:
            return f"No memories found in time range {start_date.date()} to {end_date.date()}"

        # Sort by creation date
        memories_in_range.sort(key=lambda m: m.created_at)

        # Generate report based on type
        if report_type == "summary":
            return self._generate_summary_report(memories_in_range, start_date, end_date)
        elif report_type == "detailed":
            return self._generate_detailed_report(memories_in_range, start_date, end_date)
        elif report_type == "timeline":
            return self._generate_timeline_report(memories_in_range, start_date, end_date)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    def _generate_summary_report(self, memories: List[MemoryData], start_date: datetime, end_date: datetime) -> str:
        """Generate a summary report."""
        lines = [
            f"# Memory Summary Report",
            f"**Period**: {start_date.date()} to {end_date.date()}",
            f"**Total Memories**: {len(memories)}",
            f"",
            f"## Statistics",
        ]

        # Calculate statistics
        avg_confidence = sum(m.confidence for m in memories) / len(memories)
        total_accesses = sum(m.access_count for m in memories)
        avg_importance = sum(m.importance for m in memories) / len(memories)

        lines.extend([
            f"- Average Confidence: {avg_confidence:.3f}",
            f"- Average Importance: {avg_importance:.3f}",
            f"- Total Accesses: {total_accesses}",
            f"",
            f"## Top Memories (by confidence)",
        ])

        # Top 5 by confidence
        top_confidence = sorted(memories, key=lambda m: m.confidence, reverse=True)[:5]
        for i, mem in enumerate(top_confidence, 1):
            lines.append(f"{i}. [{mem.confidence:.2f}] {mem.content[:60]}...")

        return "\n".join(lines)

    def _generate_detailed_report(self, memories: List[MemoryData], start_date: datetime, end_date: datetime) -> str:
        """Generate a detailed report."""
        lines = [
            f"# Detailed Memory Report",
            f"**Period**: {start_date.date()} to {end_date.date()}",
            f"**Total Memories**: {len(memories)}",
            f"",
        ]

        for i, mem in enumerate(memories, 1):
            lines.extend([
                f"## Memory {i}",
                f"- **ID**: {mem.memory_id}",
                f"- **Content**: {mem.content}",
                f"- **Created**: {mem.created_at.isoformat()}",
                f"- **Confidence**: {mem.confidence:.3f}",
                f"- **Importance**: {mem.importance:.3f}",
                f"- **Access Count**: {mem.access_count}",
                f"- **Last Accessed**: {mem.last_accessed_at.isoformat() if mem.last_accessed_at else 'Never'}",
                f"",
            ])

        return "\n".join(lines)

    def _generate_timeline_report(self, memories: List[MemoryData], start_date: datetime, end_date: datetime) -> str:
        """Generate a timeline report."""
        lines = [
            f"# Memory Timeline",
            f"**Period**: {start_date.date()} to {end_date.date()}",
            f"**Total Memories**: {len(memories)}",
            f"",
        ]

        for mem in memories:
            timestamp = mem.created_at.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- [{timestamp}] {mem.content[:80]}...")

        return "\n".join(lines)
