"""
Unit tests for memory evolution strategies.

Tests all strategy interfaces and their default implementations:
- ConfidenceEvolutionStrategy
- FirmnessCalculator
- ForgettingThresholdStrategy
- RelationExplorationStrategy
"""

from datetime import datetime, timedelta
import pytest

from mempy.core.memory import Memory
from mempy.core.interfaces import RelationType
from mempy.strategies.confidence import (
    ConfidenceEvolutionStrategy,
    SimpleConfidenceStrategy
)
from mempy.strategies.firmness import (
    FirmnessCalculator,
    WeightedFirmnessCalculator
)
from mempy.strategies.forgetting import (
    ForgettingThresholdStrategy,
    FixedThresholdStrategy,
    ImportanceAwareThreshold,
    TimeAwareThreshold
)
from mempy.strategies.exploration import (
    RelationExplorationStrategy,
    CosineSimilarityExplorer,
    AdaptiveSimilarityExplorer
)


# ============================================================================
# Confidence Evolution Strategy Tests
# ============================================================================

class TestConfidenceEvolutionStrategy:
    """Test ConfidenceEvolutionStrategy interface and implementations."""

    def test_cannot_instantiate_abstract_strategy(self):
        """Abstract strategy should not be instantiable."""
        with pytest.raises(TypeError):
            ConfidenceEvolutionStrategy()

    @pytest.mark.asyncio
    async def test_simple_strategy_reference_reinforcement(self):
        """Test simple strategy reference reinforcement."""
        strategy = SimpleConfidenceStrategy()
        memory = Memory(
            memory_id="test-1",
            content="Test",
            embedding=[0.1] * 128,
            confidence=0.5
        )

        increment = await strategy.reinforce_on_reference(memory)

        assert increment == SimpleConfidenceStrategy.DEFAULT_REFERENCE_INCREMENT
        assert increment == 0.1

    @pytest.mark.asyncio
    async def test_simple_strategy_relation_reinforcement(self):
        """Test simple strategy relation reinforcement."""
        strategy = SimpleConfidenceStrategy()
        memory = Memory(
            memory_id="test-1",
            content="Test",
            embedding=[0.1] * 128,
            confidence=0.5
        )

        increment = await strategy.reinforce_on_relation(memory)

        assert increment == SimpleConfidenceStrategy.DEFAULT_RELATION_INCREMENT
        assert increment == 0.05

    @pytest.mark.asyncio
    async def test_simple_strategy_time_decay(self):
        """Test simple strategy time decay calculation."""
        strategy = SimpleConfidenceStrategy()
        memory = Memory(
            memory_id="test-1",
            content="Test",
            embedding=[0.1] * 128,
            confidence=0.5
        )

        # Test 10 days decay
        decay = await strategy.decay_over_time(memory, 10)
        assert decay == 0.1  # 10 * 0.01

        # Test 30 days decay
        decay = await strategy.decay_over_time(memory, 30)
        assert decay == 0.3  # 30 * 0.01

        # Test negative days (edge case)
        decay = await strategy.decay_over_time(memory, -5)
        assert decay == 0.0

    @pytest.mark.asyncio
    async def test_simple_strategy_custom_parameters(self):
        """Test simple strategy with custom parameters."""
        strategy = SimpleConfidenceStrategy(
            reference_increment=0.2,
            relation_increment=0.1,
            daily_decay_rate=0.02
        )
        memory = Memory(
            memory_id="test-1",
            content="Test",
            embedding=[0.1] * 128,
        )

        ref_increment = await strategy.reinforce_on_reference(memory)
        rel_increment = await strategy.reinforce_on_relation(memory)
        decay = await strategy.decay_over_time(memory, 10)

        assert ref_increment == 0.2
        assert rel_increment == 0.1
        assert decay == 0.2  # 10 * 0.02


# ============================================================================
# Firmness Calculator Tests
# ============================================================================

class TestFirmnessCalculator:
    """Test FirmnessCalculator interface and implementations."""

    def test_cannot_instantiate_abstract_calculator(self):
        """Abstract calculator should not be instantiable."""
        with pytest.raises(TypeError):
            FirmnessCalculator()

    def test_weighted_calculator_basic_calculation(self):
        """Test basic firmness calculation."""
        calculator = WeightedFirmnessCalculator()

        # Create memory with all fields
        memory = Memory(
            memory_id="test-1",
            content="Test",
            embedding=[0.1] * 128,
            confidence=0.8,
            last_accessed_at=datetime.now(),
            access_count=5,
            importance=0.7
        )

        firmness = calculator.calculate(
            memory=memory,
            relation_count=3,
            avg_relation_confidence=0.7
        )

        # Should be a valid score
        assert 0.0 <= firmness <= 1.0
        # Should be reasonably high given good stats
        assert firmness > 0.3

    def test_weighted_calculator_no_access(self):
        """Test firmness for never-accessed memory."""
        calculator = WeightedFirmnessCalculator()

        memory = Memory(
            memory_id="test-1",
            content="Test",
            embedding=[0.1] * 128,
            confidence=0.5,
            last_accessed_at=None,  # Never accessed
            access_count=0,
        )

        firmness = calculator.calculate(
            memory=memory,
            relation_count=0,
            avg_relation_confidence=0.0
        )

        # Should be low due to no access and no relations
        assert firmness < 0.3

    def test_weighted_calculator_high_firmness(self):
        """Test firmness for highly accessed memory with many relations."""
        calculator = WeightedFirmnessCalculator()

        memory = Memory(
            memory_id="test-1",
            content="Test",
            embedding=[0.1] * 128,
            confidence=1.0,
            last_accessed_at=datetime.now(),  # Just accessed
            access_count=20,  # Many accesses
            importance=1.0
        )

        firmness = calculator.calculate(
            memory=memory,
            relation_count=10,  # Many relations
            avg_relation_confidence=1.0  # High quality relations
        )

        # Should be very high
        assert firmness > 0.7

    def test_weighted_calculator_recency_decay(self):
        """Test recency score decay over time."""
        calculator = WeightedFirmnessCalculator(recency_window_days=7)

        # Recent access
        memory_recent = Memory(
            memory_id="test-1",
            content="Test",
            embedding=[0.1] * 128,
            last_accessed_at=datetime.now() - timedelta(days=1),
            access_count=5,
            confidence=0.8
        )

        # Old access
        memory_old = Memory(
            memory_id="test-2",
            content="Test",
            embedding=[0.1] * 128,
            last_accessed_at=datetime.now() - timedelta(days=30),
            access_count=5,
            confidence=0.8
        )

        firmness_recent = calculator.calculate(memory_recent, 3, 0.7)
        firmness_old = calculator.calculate(memory_old, 3, 0.7)

        # Recent should have higher firmness
        assert firmness_recent > firmness_old

    def test_weighted_calculator_custom_weights(self):
        """Test calculator with custom weights."""
        # Emphasize confidence over other factors
        custom_weights = {
            "recency": 0.1,
            "count": 0.1,
            "relation": 0.1,
            "quality": 0.1,
            "confidence": 0.6  # 60% weight on confidence
        }

        calculator = WeightedFirmnessCalculator(weights=custom_weights)

        memory = Memory(
            memory_id="test-1",
            content="Test",
            embedding=[0.1] * 128,
            confidence=1.0,
            last_accessed_at=None,
            access_count=0
        )

        firmness = calculator.calculate(memory, 0, 0.0)

        # Should be high due to 60% weight on confidence=1.0
        assert firmness > 0.5

    def test_weighted_calculator_invalid_weights(self):
        """Test that invalid weights raise error."""
        invalid_weights = {
            "recency": 0.5,
            "count": 0.5,
            "relation": 0.1,
            "quality": 0.1,
            "confidence": 0.1
        }  # Sum = 1.3, not 1.0

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            WeightedFirmnessCalculator(weights=invalid_weights)


# ============================================================================
# Forgetting Threshold Strategy Tests
# ============================================================================

class TestForgettingThresholdStrategy:
    """Test ForgettingThresholdStrategy interface and implementations."""

    def test_cannot_instantiate_abstract_strategy(self):
        """Abstract strategy should not be instantiable."""
        with pytest.raises(TypeError):
            ForgettingThresholdStrategy()

    def test_fixed_threshold_basic(self):
        """Test basic fixed threshold strategy."""
        strategy = FixedThresholdStrategy(threshold=0.3)

        memory = Memory(
            memory_id="test-1",
            content="Test",
            embedding=[0.1] * 128
        )

        # Below threshold - should forget
        assert strategy.should_forget(memory, firmness=0.2) is True

        # At threshold - should not forget
        assert strategy.should_forget(memory, firmness=0.3) is False

        # Above threshold - should not forget
        assert strategy.should_forget(memory, firmness=0.5) is False

    def test_fixed_threshold_custom_threshold(self):
        """Test fixed threshold with custom value."""
        strategy = FixedThresholdStrategy(threshold=0.7)

        memory = Memory(
            memory_id="test-1",
            content="Test",
            embedding=[0.1] * 128
        )

        # Below 0.7
        assert strategy.should_forget(memory, firmness=0.6) is True

        # At or above 0.7
        assert strategy.should_forget(memory, firmness=0.7) is False
        assert strategy.should_forget(memory, firmness=0.8) is False

    def test_fixed_threshold_invalid_threshold(self):
        """Test that invalid threshold raises error."""
        with pytest.raises(ValueError):
            FixedThresholdStrategy(threshold=1.5)  # Too high

        with pytest.raises(ValueError):
            FixedThresholdStrategy(threshold=-0.1)  # Too low

    def test_importance_aware_threshold(self):
        """Test importance-aware threshold strategy."""
        strategy = ImportanceAwareThreshold(
            base_threshold=0.3,
            adjustment_factor=0.3
        )

        # Important memory (importance=1.0)
        memory_important = Memory(
            memory_id="test-1",
            content="Important",
            embedding=[0.1] * 128,
            importance=1.0
        )

        # Unimportant memory (importance=0.0)
        memory_trivial = Memory(
            memory_id="test-2",
            content="Trivial",
            embedding=[0.1] * 128,
            importance=0.0
        )

        # Same firmness for both
        firmness = 0.35

        # Important memory should NOT be forgotten (threshold ≈ 0.3)
        assert strategy.should_forget(memory_important, firmness) is False

        # Trivial memory SHOULD be forgotten (threshold ≈ 0.6)
        assert strategy.should_forget(memory_trivial, firmness) is True

    def test_time_aware_threshold(self):
        """Test time-aware threshold strategy."""
        strategy = TimeAwareThreshold(
            base_threshold=0.3,
            max_adjustment=0.2,
            adjustment_per_year=0.1
        )

        # Old memory (2 years)
        memory_old = Memory(
            memory_id="test-1",
            content="Old",
            embedding=[0.1] * 128,
            created_at=datetime.now() - timedelta(days=730)
        )

        # New memory (1 day)
        memory_new = Memory(
            memory_id="test-2",
            content="New",
            embedding=[0.1] * 128,
            created_at=datetime.now() - timedelta(days=1)
        )

        firmness = 0.25

        # Old memory should NOT be forgotten (threshold ≈ 0.1)
        assert strategy.should_forget(memory_old, firmness) is False

        # New memory SHOULD be forgotten (threshold ≈ 0.3)
        assert strategy.should_forget(memory_new, firmness) is True


# ============================================================================
# Relation Exploration Strategy Tests
# ============================================================================

class TestRelationExplorationStrategy:
    """Test RelationExplorationStrategy interface and implementations."""

    def test_cannot_instantiate_abstract_strategy(self):
        """Abstract strategy should not be instantiable."""
        with pytest.raises(TypeError):
            RelationExplorationStrategy()

    @pytest.mark.asyncio
    async def test_cosine_similarity_explorer(self):
        """Test cosine similarity exploration."""
        explorer = CosineSimilarityExplorer(max_relations=10)

        # Create memories with different embeddings
        memory1 = Memory(
            memory_id="mem-1",
            content="Similar content",
            embedding=[0.5] * 128,  # Similar to mem-2
        )

        memory2 = Memory(
            memory_id="mem-2",
            content="Similar content",
            embedding=[0.5] * 128,
        )

        memory3 = Memory(
            memory_id="mem-3",
            content="Different content",
            embedding=[0.1] * 128,  # Different from mem-1 and mem-2
        )

        memories = [memory1, memory2, memory3]

        # With low threshold, should find relations
        relations = await explorer.explore(memories, similarity_threshold=0.9)

        # Should find relation between mem-1 and mem-2 (identical)
        assert len(relations) >= 1

        # Check relation structure
        mem1, mem2, rel_type = relations[0]
        assert rel_type == RelationType.SIMILAR

    @pytest.mark.asyncio
    async def test_cosine_similarity_explorer_empty(self):
        """Test explorer with empty memory list."""
        explorer = CosineSimilarityExplorer()

        relations = await explorer.explore([], similarity_threshold=0.8)

        assert len(relations) == 0

    @pytest.mark.asyncio
    async def test_cosine_similarity_explorer_high_threshold(self):
        """Test explorer with high similarity threshold."""
        explorer = CosineSimilarityExplorer()

        # Create memories with orthogonal embeddings (dissimilar)
        import numpy as np
        memories = []
        for i in range(5):
            # Create orthogonal-like vectors
            embedding = np.zeros(128)
            embedding[i] = 1.0  # Each vector has a 1.0 at a different position
            memories.append(
                Memory(
                    memory_id=f"mem-{i}",
                    content=f"Content {i}",
                    embedding=embedding.tolist()
                )
            )

        # Very high threshold
        relations = await explorer.explore(memories, similarity_threshold=0.99)

        # Should find very few or no relations with orthogonal vectors
        assert len(relations) <= 1

    @pytest.mark.asyncio
    async def test_adaptive_similarity_explorer(self):
        """Test adaptive similarity explorer."""
        explorer = AdaptiveSimilarityExplorer(
            max_relations=10,
            max_relations_per_memory=3,
            confidence_threshold=0.7
        )

        # Create memories with varying confidence
        memories = [
            Memory(
                memory_id=f"mem-{i}",
                content=f"Content {i}",
                embedding=[0.5] * 128,
                confidence=0.5  # Below threshold
            )
            for i in range(5)
        ]

        # Add one high-confidence memory (should be filtered out)
        memories.append(
            Memory(
                memory_id="mem-high",
                content="High confidence",
                embedding=[0.5] * 128,
                confidence=0.9  # Above threshold
            )
        )

        relations = await explorer.explore(memories, similarity_threshold=0.8)

        # Should find relations only among low-confidence memories
        for mem1, mem2, _ in relations:
            assert mem1.confidence < 0.7
            assert mem2.confidence < 0.7

    @pytest.mark.asyncio
    async def test_explorer_relation_types(self):
        """Test that explorer uses appropriate relation types."""
        explorer = CosineSimilarityExplorer()

        # Identical embeddings
        memory1 = Memory(
            memory_id="mem-1",
            content="Test",
            embedding=[0.5] * 128,
        )

        memory2 = Memory(
            memory_id="mem-2",
            content="Test",
            embedding=[0.5] * 128,
        )

        relations = await explorer.explore([memory1, memory2], similarity_threshold=0.9)

        assert len(relations) > 0
        # CosineSimilarityExplorer always uses SIMILAR
        assert relations[0][2] == RelationType.SIMILAR

    @pytest.mark.asyncio
    async def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation is correct."""
        explorer = CosineSimilarityExplorer()

        # Orthogonal vectors (similarity = 0)
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = explorer._cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.01  # Should be ~0

        # Parallel vectors (similarity = 1)
        vec3 = [1.0, 1.0, 1.0]
        vec4 = [1.0, 1.0, 1.0]
        sim = explorer._cosine_similarity(vec3, vec4)
        assert abs(sim - 1.0) < 0.01  # Should be ~1

        # Opposite vectors (similarity = -1)
        vec5 = [1.0, 1.0, 1.0]
        vec6 = [-1.0, -1.0, -1.0]
        sim = explorer._cosine_similarity(vec5, vec6)
        assert abs(sim - (-1.0)) < 0.01  # Should be ~-1


# ============================================================================
# Integration Tests
# ============================================================================

class TestStrategyIntegration:
    """Test strategies working together."""

    @pytest.mark.asyncio
    async def test_full_evolution_pipeline(self):
        """Test a complete evolution pipeline with all strategies."""
        # Setup strategies
        confidence_strategy = SimpleConfidenceStrategy()
        firmness_calculator = WeightedFirmnessCalculator()
        forgetting_strategy = FixedThresholdStrategy(threshold=0.3)

        # Create a memory
        memory = Memory(
            memory_id="test-1",
            content="Test memory",
            embedding=[0.1] * 128,
            confidence=0.6,
            last_accessed_at=datetime.now(),
            access_count=3,
            importance=0.8
        )

        # Simulate evolution
        # 1. Reinforce confidence on reference
        increment = await confidence_strategy.reinforce_on_reference(memory)
        memory.confidence = min(1.0, memory.confidence + increment)
        assert memory.confidence == 0.7

        # 2. Calculate firmness
        firmness = firmness_calculator.calculate(
            memory=memory,
            relation_count=2,
            avg_relation_confidence=0.7
        )
        assert 0.0 <= firmness <= 1.0

        # 3. Check if should be forgotten
        should_forget = forgetting_strategy.should_forget(memory, firmness)

        # With good stats, should not be forgotten
        assert should_forget is False

    @pytest.mark.asyncio
    async def test_low_firmness_memory_forgetting(self):
        """Test that low firmness memories are marked for forgetting."""
        confidence_strategy = SimpleConfidenceStrategy()
        firmness_calculator = WeightedFirmnessCalculator()
        forgetting_strategy = FixedThresholdStrategy(threshold=0.3)

        # Create neglected memory
        memory = Memory(
            memory_id="neglected-1",
            content="Old neglected memory",
            embedding=[0.1] * 128,
            confidence=0.2,  # Low confidence
            last_accessed_at=None,  # Never accessed
            access_count=0,  # Never accessed
            importance=0.3  # Low importance
        )

        # Apply time decay
        decay = await confidence_strategy.decay_over_time(memory, days=100)
        memory.confidence = max(0.0, memory.confidence - decay)

        # Calculate firmness (should be very low)
        firmness = firmness_calculator.calculate(
            memory=memory,
            relation_count=0,
            avg_relation_confidence=0.0
        )

        # Should have very low firmness
        assert firmness < 0.2

        # Should be marked for forgetting
        assert forgetting_strategy.should_forget(memory, firmness) is True
