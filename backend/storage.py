"""JSON-based storage for conversations with file locking for concurrency safety."""

import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import DATA_DIR


def ensure_data_dir():
    """Ensure the data directory exists."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_conversation_path(conversation_id: str) -> str:
    """Get the file path for a conversation."""
    return os.path.join(DATA_DIR, f"{conversation_id}.json")


@contextmanager
def _locked_file(path: str, mode: str = 'r'):
    """Context manager that acquires an exclusive file lock for safe read-modify-write."""
    ensure_data_dir()
    # For write modes, open with r+ if file exists, else w
    if 'w' in mode or '+' in mode:
        f = open(path, mode)
    else:
        f = open(path, mode)
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield f
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def _read_and_modify(conversation_id: str, modifier):
    """
    Atomically read, modify, and write a conversation file with file locking.
    `modifier` is a callable that takes the conversation dict and modifies it in place.
    """
    path = get_conversation_path(conversation_id)
    if not os.path.exists(path):
        raise ValueError(f"Conversation {conversation_id} not found")

    with _locked_file(path, 'r+') as f:
        conversation = json.load(f)
        modifier(conversation)
        f.seek(0)
        f.truncate()
        json.dump(conversation, f, indent=2, ensure_ascii=False)

    return conversation


def create_conversation(conversation_id: str) -> Dict[str, Any]:
    """Create a new conversation."""
    ensure_data_dir()

    conversation = {
        "id": conversation_id,
        "created_at": datetime.utcnow().isoformat(),
        "title": "New Conversation",
        "messages": []
    }

    path = get_conversation_path(conversation_id)
    with _locked_file(path, 'w') as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)

    return conversation


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Load a conversation from storage."""
    path = get_conversation_path(conversation_id)

    if not os.path.exists(path):
        return None

    with _locked_file(path, 'r') as f:
        return json.load(f)


def save_conversation(conversation: Dict[str, Any]):
    """Save a conversation to storage."""
    ensure_data_dir()

    path = get_conversation_path(conversation['id'])
    with _locked_file(path, 'w') as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)


def list_conversations() -> List[Dict[str, Any]]:
    """List all conversations (metadata only)."""
    ensure_data_dir()

    conversations = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            path = os.path.join(DATA_DIR, filename)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    conversations.append({
                        "id": data["id"],
                        "created_at": data["created_at"],
                        "title": data.get("title", "New Conversation"),
                        "message_count": len(data["messages"])
                    })
            except (json.JSONDecodeError, KeyError):
                continue  # Skip corrupted files

    conversations.sort(key=lambda x: x["created_at"], reverse=True)
    return conversations


def add_user_message(conversation_id: str, content: str):
    """Add a user message to a conversation (with file locking)."""
    def modifier(conv):
        conv["messages"].append({
            "role": "user",
            "content": content
        })
    _read_and_modify(conversation_id, modifier)


def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage3: Dict[str, Any],
    metadata: Dict[str, Any] = None
):
    """
    Add an assistant message with all 3 stages to a conversation.
    Metadata (label_to_model, aggregate_rankings) is now persisted alongside stages.
    """
    def modifier(conv):
        msg = {
            "role": "assistant",
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
        }
        if metadata:
            msg["metadata"] = metadata
        conv["messages"].append(msg)
    _read_and_modify(conversation_id, modifier)


def update_conversation_title(conversation_id: str, title: str):
    """Update the title of a conversation (with file locking)."""
    def modifier(conv):
        conv["title"] = title
    _read_and_modify(conversation_id, modifier)
