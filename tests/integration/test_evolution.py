"""
Integration tests for memory evolution features.

Tests all evolution methods including:
- Confidence evolution
- Forgetting mechanism
- Subconscious relation exploration
- Time aggregation and reporting
- Access tracking
"""

from datetime import datetime, timedelta
import pytest

from mempy.core.memory import Memory as MemoryData, RelationType
from mempy.memory import Memory as MemoryAPI
from mempy.core.exceptions import StorageError


# ============================================================================
# Access Tracking Tests
# ============================================================================

class TestAccessTracking:
    """Test access tracking functionality."""

    @pytest.mark.asyncio
    async def test_get_increments_access_count(self, memory_api):
        """Test that get() increments access count."""
        memory_id = await memory_api.add("Test memory")

        # First access
        mem1 = await memory_api.get(memory_id)
        assert mem1.access_count == 1
        assert mem1.last_accessed_at is not None

        # Second access
        mem2 = await memory_api.get(memory_id)
        assert mem2.access_count == 2

    @pytest.mark.asyncio
    async def test_get_updates_last_accessed_at(self, memory_api):
        """Test that get() updates last_accessed_at."""
        memory_id = await memory_api.add("Test memory")

        # First access
        mem1 = await memory_api.get(memory_id)
        first_access_time = mem1.last_accessed_at

        # Wait a bit and access again
        import time
        time.sleep(0.01)  # Small delay to ensure time difference

        # Second access
        mem2 = await memory_api.get(memory_id)
        second_access_time = mem2.last_accessed_at

        assert second_access_time > first_access_time

    @pytest.mark.asyncio
    async def test_get_nonexistent_memory(self, memory_api):
        """Test that getting nonexistent memory returns None."""
        result = await memory_api.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_add_supports_importance(self, memory_api):
        """Test that add() supports importance parameter."""
        # Add with high importance
        mem_id = await memory_api.add("Important", importance=1.0)
        mem = await memory_api.get(mem_id)
        assert mem.importance == 1.0

        # Add with low importance
        mem_id2 = await memory_api.add("Unimportant", importance=0.1)
        mem2 = await memory_api.get(mem_id2)
        assert mem2.importance == 0.1

        # Default importance
        mem_id3 = await memory_api.add("Normal")
        mem3 = await memory_api.get(mem_id3)
        assert mem3.importance == 0.5


# ============================================================================
# Confidence Evolution Tests
# ============================================================================

class TestConfidenceEvolution:
    """Test confidence evolution functionality."""

    @pytest.mark.asyncio
    async def test_evolve_confidence_old_memories(self, memory_api):
        """Test that old memories have their confidence decayed."""
        # Add a memory
        memory_id = await memory_api.add("Old memory", user_id="test-user")

        # Manually set last_accessed_at and created_at to long ago
        memory = await memory_api.get(memory_id)
        memory.last_accessed_at = datetime.utcnow() - timedelta(days=50)
        memory.created_at = datetime.utcnow() - timedelta(days=60)
        await memory_api.storage.update(memory_id, memory)

        # Run evolution with low threshold
        result = await memory_api.evolve_confidence(days_threshold=10)

        # Check results
        assert result["total_count"] >= 1
        assert result["updated_count"] >= 1
        assert result["avg_decay"] > 0

        # Verify memory was actually updated
        updated_memory = await memory_api.get(memory_id)
        assert updated_memory.confidence < 1.0

    @pytest.mark.asyncio
    async def test_evolve_confidence_recent_memories_unchanged(self, memory_api):
        """Test that recently accessed memories are not decayed."""
        # Add a memory
        memory_id = await memory_api.add("Recent memory", user_id="test-user")

        # Set last_accessed_at to recent
        memory = await memory_api.get(memory_id)
        memory.last_accessed_at = datetime.utcnow() - timedelta(days=5)
        await memory_api.storage.update(memory_id, memory)

        # Run evolution
        result = await memory_api.evolve_confidence(days_threshold=30)

        # Check that this memory was not updated
        assert result["updated_count"] == 0

        # Verify confidence unchanged
        updated_memory = await memory_api.get(memory_id)
        assert updated_memory.confidence == 1.0

    @pytest.mark.asyncio
    async def test_reinforce_confidence_manual(self, memory_api):
        """Test manual confidence reinforcement."""
        # Add a memory with low confidence
        memory_id = await memory_api.add("Low confidence memory", user_id="test-user")
        memory = await memory_api.get(memory_id)
        memory.confidence = 0.5
        await memory_api.storage.update(memory_id, memory)

        # Reinforce
        new_confidence = await memory_api.reinforce_confidence(memory_id, reason="manual")

        # Check confidence increased
        assert new_confidence > 0.5
        assert new_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_reinforce_confidence_relation(self, memory_api):
        """Test confidence reinforcement when creating relations."""
        # Add two memories
        mem1_id = await memory_api.add("Memory 1", user_id="test-user")
        mem2_id = await memory_api.add("Memory 2", user_id="test-user")

        # Set low confidence
        for mem_id in [mem1_id, mem2_id]:
            memory = await memory_api.get(mem_id)
            memory.confidence = 0.5
            await memory_api.storage.update(mem_id, memory)

        # Create relation (should reinforce both)
        await memory_api.add_relation(mem1_id, mem2_id, RelationType.RELATED)

        # Check both memories were reinforced
        mem1 = await memory_api.get(mem1_id)
        mem2 = await memory_api.get(mem2_id)

        assert mem1.confidence > 0.5
        assert mem2.confidence > 0.5

    @pytest.mark.asyncio
    async def test_reinforce_nonexistent_memory(self, memory_api):
        """Test that reinforcing nonexistent memory raises error."""
        with pytest.raises(StorageError):
            await memory_api.reinforce_confidence("nonexistent-id")


# ============================================================================
# Forgetting Mechanism Tests
# ============================================================================

class TestForgettingMechanism:
    """Test forgetting/cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_forgotten_dry_run(self, memory_api):
        """Test dry run mode for cleanup."""
        # Add memories with varying characteristics
        mem1_id = await memory_api.add("Important memory", user_id="test-user")
        mem2_id = await memory_api.add("Unimportant memory", user_id="test-user")

        # Set importance manually
        mem1 = await memory_api.get(mem1_id)
        mem1.importance = 1.0
        await memory_api.storage.update(mem1_id, mem1)

        mem2 = await memory_api.get(mem2_id)
        mem2.importance = 0.1
        await memory_api.storage.update(mem2_id, mem2)

        # Run dry run
        result = await memory_api.cleanup_forgotten(threshold=0.3, dry_run=True)

        # Check that no memories were actually deleted
        assert result["deleted_count"] >= 0
        all_memories = await memory_api.get_all(user_id="test-user")
        # Memories should still exist
        assert len(all_memories) >= 2

    @pytest.mark.asyncio
    async def test_cleanup_forgotten_actual_deletion(self, memory_api):
        """Test actual deletion of forgotten memories."""
        # Create a memory with low firmness potential
        memory_id = await memory_api.add("Unimportant old memory", user_id="test-user")
        memory = await memory_api.get(memory_id)
        memory.importance = 0.0
        memory.confidence = 0.1
        memory.last_accessed_at = None
        memory.access_count = 0
        await memory_api.storage.update(memory_id, memory)

        # Run cleanup with low threshold
        result = await memory_api.cleanup_forgotten(threshold=0.5, dry_run=False)

        # Check the memory was deleted
        assert result["deleted_count"] >= 1
        deleted_memory = await memory_api.get(memory_id)
        assert deleted_memory is None

    @pytest.mark.asyncio
    async def test_cleanup_with_user_filter(self, memory_api):
        """Test cleanup with user filter."""
        # Add memories for different users
        await memory_api.add("User 1 memory", user_id="user-1")
        await memory_api.add("User 2 memory", user_id="user-2")

        # Cleanup only user-1
        result = await memory_api.cleanup_forgotten(
            threshold=0.1,
            user_id="user-1",
            dry_run=True
        )

        # Should only process user-1 memories
        user2_memories = await memory_api.get_all(user_id="user-2")
        assert len(user2_memories) > 0


# ============================================================================
# Subconscious Exploration Tests
# ============================================================================

class TestSubconsciousExploration:
    """Test subconscious relation exploration."""

    @pytest.mark.asyncio
    async def test_explore_relations_low_confidence(self, memory_api):
        """Test exploration finds relations for low-confidence memories."""
        # Add memories with similar content but low confidence
        mem1_id = await memory_api.add("Python programming", user_id="test-user")
        mem2_id = await memory_api.add("Python code", user_id="test-user")

        # Set low confidence
        for mem_id in [mem1_id, mem2_id]:
            memory = await memory_api.get(mem_id)
            memory.confidence = 0.3
            await memory_api.storage.update(mem_id, memory)

        # Run exploration
        result = await memory_api.explore_relations(
            confidence_threshold=0.5,
            similarity_threshold=0.7,
            max_new_relations=5
        )

        # Should have processed some memories
        assert result["processed_count"] >= 2

    @pytest.mark.asyncio
    async def test_explore_relations_no_candidates(self, memory_api):
        """Test exploration with no low-confidence memories."""
        # Add memories with high confidence
        await memory_api.add("High confidence 1", user_id="test-user")
        await memory_api.add("High confidence 2", user_id="test-user")

        # Run exploration with high threshold
        result = await memory_api.explore_relations(confidence_threshold=0.9)

        # Should process no memories
        assert result["processed_count"] == 0
        assert result["new_relations"] == 0

    @pytest.mark.asyncio
    async def test_explore_relations_max_limit(self, memory_api):
        """Test exploration respects max_new_relations limit."""
        # Add multiple memories
        for i in range(10):
            await memory_api.add(f"Memory {i}", user_id="test-user")

        # Set all to low confidence
        memories = await memory_api.get_all(user_id="test-user")
        for memory in memories:
            memory.confidence = 0.3
            await memory_api.storage.update(memory.memory_id, memory)

        # Run exploration with low limit
        result = await memory_api.explore_relations(
            confidence_threshold=0.5,
            max_new_relations=2
        )

        # Should not create more than max
        assert result["new_relations"] <= 2


# ============================================================================
# Time Aggregation Tests
# ============================================================================

class TestTimeAggregation:
    """Test time aggregation and reporting."""

    @pytest.mark.asyncio
    async def test_compress_memories(self, memory_api):
        """Test compressing memories in a time range."""
        # Add memories within a time range
        await memory_api.add("Memory 1", user_id="test-user")
        await memory_api.add("Memory 2", user_id="test-user")

        # Get all memories to determine time range
        memories = await memory_api.get_all(user_id="test-user")
        start_date = memories[0].created_at - timedelta(seconds=1)
        end_date = memories[-1].created_at + timedelta(seconds=1)

        # Compress
        compressed = await memory_api.compress_memories(
            start_date=start_date,
            end_date=end_date,
            summary="Compressed summary of memories",
            user_id="test-user"
        )

        # Check compressed memory was created
        assert compressed is not None
        assert compressed.content == "Compressed summary of memories"
        assert compressed.metadata["compressed"] == True

        # Check original memories marked as compressed
        original_memories = await memory_api.get_all(user_id="test-user")
        compressed_originals = [m for m in original_memories if m.metadata.get("compressed", False)]
        assert len(compressed_originals) >= 2

    @pytest.mark.asyncio
    async def test_compress_memories_empty_range(self, memory_api):
        """Test compressing with no memories in range."""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 12, 31)

        with pytest.raises(StorageError):
            await memory_api.compress_memories(
                start_date=start_date,
                end_date=end_date,
                summary="No memories in this range",
                user_id="test-user"
            )

    @pytest.mark.asyncio
    async def test_generate_summary_report(self, memory_api):
        """Test generating a summary report."""
        # Add some memories
        await memory_api.add("Memory 1", user_id="test-user")
        await memory_api.add("Memory 2", user_id="test-user")

        # Get memories to determine time range
        memories = await memory_api.get_all(user_id="test-user")
        start_date = memories[0].created_at - timedelta(seconds=1)
        end_date = memories[-1].created_at + timedelta(seconds=1)

        # Generate report
        report = await memory_api.generate_report(
            start_date=start_date,
            end_date=end_date,
            user_id="test-user",
            report_type="summary"
        )

        # Check report content
        assert "Memory Summary Report" in report
        assert "Total Memories" in report
        assert "Average Confidence" in report

    @pytest.mark.asyncio
    async def test_generate_detailed_report(self, memory_api):
        """Test generating a detailed report."""
        # Add a memory
        await memory_api.add("Test memory", user_id="test-user")

        # Get memory to determine time range
        memories = await memory_api.get_all(user_id="test-user")
        start_date = memories[0].created_at - timedelta(seconds=1)
        end_date = memories[0].created_at + timedelta(seconds=1)

        # Generate detailed report
        report = await memory_api.generate_report(
            start_date=start_date,
            end_date=end_date,
            user_id="test-user",
            report_type="detailed"
        )

        # Check report content
        assert "Detailed Memory Report" in report
        assert "Memory 1" in report
        assert "Test memory" in report

    @pytest.mark.asyncio
    async def test_generate_timeline_report(self, memory_api):
        """Test generating a timeline report."""
        # Add a memory
        await memory_api.add("Test memory", user_id="test-user")

        # Get memory to determine time range
        memories = await memory_api.get_all(user_id="test-user")
        start_date = memories[0].created_at - timedelta(seconds=1)
        end_date = memories[0].created_at + timedelta(seconds=1)

        # Generate timeline report
        report = await memory_api.generate_report(
            start_date=start_date,
            end_date=end_date,
            user_id="test-user",
            report_type="timeline"
        )

        # Check report content
        assert "Memory Timeline" in report
        assert "Test memory" in report

    @pytest.mark.asyncio
    async def test_generate_report_no_memories(self, memory_api):
        """Test report with no memories in range."""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 12, 31)

        report = await memory_api.generate_report(
            start_date=start_date,
            end_date=end_date,
            user_id="test-user",
            report_type="summary"
        )

        assert "No memories found" in report

    @pytest.mark.asyncio
    async def test_generate_report_invalid_type(self, memory_api):
        """Test report with invalid type."""
        await memory_api.add("Test", user_id="test-user")

        memories = await memory_api.get_all(user_id="test-user")
        start_date = memories[0].created_at - timedelta(seconds=1)
        end_date = memories[0].created_at + timedelta(seconds=1)

        with pytest.raises(ValueError, match="Unknown report type"):
            await memory_api.generate_report(
                start_date=start_date,
                end_date=end_date,
                report_type="invalid"
            )


# ============================================================================
# Integration Tests
# ============================================================================

class TestEvolutionIntegration:
    """Test full evolution workflows."""

    @pytest.mark.asyncio
    async def test_full_evolution_cycle(self, memory_api):
        """Test a complete evolution cycle."""
        # 1. Add memories
        mem_ids = []
        for i in range(5):
            mem_id = await memory_api.add(f"Memory {i}", user_id="test-user")
            mem_ids.append(mem_id)

        # 2. Create some relations
        await memory_api.add_relation(mem_ids[0], mem_ids[1], RelationType.RELATED)
        await memory_api.add_relation(mem_ids[1], mem_ids[2], RelationType.RELATED)

        # 3. Explore for more relations
        exploration_result = await memory_api.explore_relations(
            confidence_threshold=1.0,  # Include all memories
            max_new_relations=3
        )

        # 4. Evolve confidence (should be minimal for new memories)
        evolution_result = await memory_api.evolve_confidence(days_threshold=30)

        # 5. Generate report
        memories = await memory_api.get_all(user_id="test-user")
        start_date = memories[0].created_at - timedelta(seconds=1)
        end_date = memories[-1].created_at + timedelta(seconds=1)

        report = await memory_api.generate_report(
            start_date=start_date,
            end_date=end_date,
            user_id="test-user",
            report_type="summary"
        )

        # Verify results
        assert exploration_result["processed_count"] >= 0
        assert evolution_result["total_count"] >= 5
        assert "Total Memories" in report
