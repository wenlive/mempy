"""LOCOMO dataset loader and data structures.

This module provides classes for loading and working with the LOCOMO dataset,
which evaluates long-term conversational memory in dialogue systems.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import json


@dataclass
class Turn:
    """A single turn in a conversation session.

    Attributes:
        speaker: Name of the speaker (speaker_a or speaker_b)
        dia_id: Unique dialogue ID
        text: Content of the turn
        img_url: Optional image URL
        blip_caption: Optional BLIP-generated caption
    """

    speaker: str
    dia_id: str
    text: str
    img_url: Optional[str] = None
    blip_caption: Optional[str] = None


@dataclass
class Session:
    """A conversation session with multiple turns.

    Attributes:
        session_id: Session identifier (e.g., "session_0")
        turns: List of conversation turns
        date_time: Session timestamp
        observation: Optional generated observation
        summary: Optional session summary
        events: Optional event summaries for each speaker
    """

    session_id: str
    turns: List[Turn]
    date_time: Optional[datetime] = None
    observation: Optional[str] = None
    summary: Optional[str] = None
    events: Optional[Dict[str, List[str]]] = None

    @classmethod
    def from_dict(cls, session_id: str, data: List[Dict], metadata: Optional[Dict] = None) -> "Session":
        """Create a Session from raw data.

        Args:
            session_id: Session identifier
            data: List of turn dictionaries
            metadata: Optional metadata containing date_time, observation, summary, events
        """
        turns = [
            Turn(
                speaker=turn["speaker"],
                dia_id=turn["dia_id"],
                text=turn["text"],
                img_url=turn.get("img_url"),
                blip_caption=turn.get("blip_caption"),
            )
            for turn in data
        ]

        # Parse datetime if available
        date_time = None
        if metadata and "date_time" in metadata:
            try:
                date_time = datetime.fromisoformat(metadata["date_time"])
            except (ValueError, TypeError):
                pass

        return cls(
            session_id=session_id,
            turns=turns,
            date_time=date_time,
            observation=metadata.get("observation") if metadata else None,
            summary=metadata.get("summary") if metadata else None,
            events=metadata.get("events") if metadata else None,
        )


@dataclass
class QA:
    """A question-answer pair for evaluation.

    Attributes:
        question: The question text
        answer: The expected answer
        category: Question category (e.g., "factual", "temporal", "causal")
        evidence: List of dialogue IDs containing evidence
    """

    question: str
    answer: str
    category: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class Conversation:
    """A complete LOCOMO conversation.

    Attributes:
        sample_id: Unique conversation identifier
        speaker_a: Name of first speaker
        speaker_b: Name of second speaker
        sessions: List of conversation sessions in chronological order
        qa_pairs: List of question-answer pairs for evaluation
    """

    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: List[Session]
    qa_pairs: List[QA] = field(default_factory=list)

    @property
    def num_sessions(self) -> int:
        """Return the number of sessions in this conversation."""
        return len(self.sessions)

    @property
    def num_turns(self) -> int:
        """Return the total number of turns across all sessions."""
        return sum(len(s.turns) for s in self.sessions)

    @property
    def num_qa_pairs(self) -> int:
        """Return the number of QA pairs."""
        return len(self.qa_pairs)

    def get_all_texts(self) -> List[str]:
        """Get all turn texts as a flat list."""
        texts = []
        for session in self.sessions:
            for turn in session.turns:
                texts.append(f"{turn.speaker}: {turn.text}")
        return texts

    def get_session_texts(self, session_index: int) -> List[str]:
        """Get texts for a specific session."""
        if session_index >= len(self.sessions):
            raise IndexError(f"Session index {session_index} out of range")
        session = self.sessions[session_index]
        return [f"{turn.speaker}: {turn.text}" for turn in session.turns]


class LOCOMODataset:
    """LOCOMO dataset loader.

    This class loads and provides access to the LOCOMO benchmark dataset
    for evaluating long-term conversational memory.

    Usage:
        >>> dataset = LOCOMODataset("path/to/locomo10.json")
        >>> for conv in dataset.get_conversations():
        ...     print(f"Conversation {conv.sample_id}: {conv.num_sessions} sessions")
    """

    def __init__(self, data_path: str):
        """Initialize the LOCOMO dataset.

        Args:
            data_path: Path to the LOCOMO JSON file
        """
        self.data_path = Path(data_path)
        self._data: Optional[List[Dict]] = None
        self._conversations: Optional[List[Conversation]] = None

    def load(self) -> None:
        """Load the dataset from disk."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"LOCOMO data file not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

    @property
    def data(self) -> List[Dict]:
        """Get the raw data, loading if necessary."""
        if self._data is None:
            self.load()
        return self._data

    def get_conversations(self) -> List[Conversation]:
        """Get all conversations as Conversation objects.

        Returns:
            List of Conversation objects
        """
        if self._conversations is None:
            self._conversations = []
            for sample in self.data:
                conv = self._parse_conversation(sample)
                if conv:
                    self._conversations.append(conv)
        return self._conversations

    def get_conversation(self, sample_id: str) -> Optional[Conversation]:
        """Get a specific conversation by ID.

        Args:
            sample_id: The conversation sample ID

        Returns:
            Conversation object or None if not found
        """
        for conv in self.get_conversations():
            if conv.sample_id == sample_id:
                return conv
        return None

    def get_all_qa_pairs(self) -> List[tuple[Conversation, QA]]:
        """Get all QA pairs with their associated conversations.

        Returns:
            List of (Conversation, QA) tuples
        """
        pairs = []
        for conv in self.get_conversations():
            for qa in conv.qa_pairs:
                pairs.append((conv, qa))
        return pairs

    def _parse_conversation(self, sample: Dict) -> Optional[Conversation]:
        """Parse a conversation from raw data.

        Args:
            sample: Raw conversation data from JSON

        Returns:
            Conversation object or None if parsing fails
        """
        try:
            sample_id = sample.get("sample_id", "")
            conv_data = sample.get("conversation", {})

            speaker_a = conv_data.get("speaker_a", "")
            speaker_b = conv_data.get("speaker_b", "")

            # Parse sessions
            sessions = []
            session_num = 1  # LOCOMO data uses 1-based indexing (session_1, session_2, etc.)
            while True:
                session_key = f"session_{session_num}"
                date_key = f"session_{session_num}_date_time"
                obs_key = f"session_{session_num}_observation"

                if session_key not in conv_data:
                    break

                session_data = conv_data[session_key]
                metadata = {
                    "date_time": conv_data.get(date_key),
                    "observation": conv_data.get(obs_key),
                    "summary": conv_data.get(f"{session_key}_summary"),
                    "events": conv_data.get(f"events_{session_key}"),
                }

                session = Session.from_dict(session_key, session_data, metadata)
                sessions.append(session)
                session_num += 1

            # Parse QA pairs
            qa_pairs = []
            for qa_data in sample.get("qa", []):
                qa = QA(
                    question=qa_data.get("question", ""),
                    answer=qa_data.get("answer", ""),
                    category=qa_data.get("category", ""),
                    evidence=qa_data.get("evidence", []),
                )
                qa_pairs.append(qa)

            return Conversation(
                sample_id=sample_id,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                sessions=sessions,
                qa_pairs=qa_pairs,
            )

        except Exception as e:
            # Log error but continue
            print(f"Warning: Failed to parse conversation: {e}")
            return None

    @property
    def num_conversations(self) -> int:
        """Return the total number of conversations."""
        return len(self.get_conversations())

    @property
    def num_qa_pairs(self) -> int:
        """Return the total number of QA pairs."""
        return sum(conv.num_qa_pairs for conv in self.get_conversations())

    def __len__(self) -> int:
        """Return the number of conversations."""
        return self.num_conversations

    def __repr__(self) -> str:
        """Return string representation."""
        return f"LOCOMODataset(conversations={self.num_conversations}, qa_pairs={self.num_qa_pairs})"


def load_locomo(data_path: str) -> LOCOMODataset:
    """Convenience function to load the LOCOMO dataset.

    Args:
        data_path: Path to the LOCOMO JSON file

    Returns:
        Loaded LOCOMODataset

    Example:
        >>> dataset = load_locomo("tests/benchmarks/data/locomo10.json")
        >>> print(dataset)
    """
    return LOCOMODataset(data_path)
