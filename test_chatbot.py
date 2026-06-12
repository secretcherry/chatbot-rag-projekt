import pytest
import rag_engine

def test_is_clearly_offtopic():
    assert rag_engine.is_clearly_offtopic("What is the weather today?") == True
    assert rag_engine.is_clearly_offtopic("Give me a recipe for pancakes") == True
    
    assert rag_engine.is_clearly_offtopic("Where is the asyncio module documented?") == False
    assert rag_engine.is_clearly_offtopic("When was f-string syntax introduced?") == False
    assert rag_engine.is_clearly_offtopic("How do I use decorators?") == False

def test_system_prompt_contains_rules():
    prompt = rag_engine.get_system_prompt()
    assert "RULES:" in prompt
    assert "```python" in prompt