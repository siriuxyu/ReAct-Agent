from agent.prompts import PREFERENCE_EXTRACTION_SYSTEM_PROMPT


def test_prompt_covers_identity_fields():
    """Prompt must explicitly mention key identity fields so the LLM extracts them."""
    required_terms = ["name", "occupation", "location", "age", "nationality"]
    prompt_lower = PREFERENCE_EXTRACTION_SYSTEM_PROMPT.lower()
    missing = [t for t in required_terms if t not in prompt_lower]
    assert missing == [], f"Prompt missing identity terms: {missing}"


def test_prompt_has_personal_examples():
    """Prompt must include concrete identity examples."""
    prompt_lower = PREFERENCE_EXTRACTION_SYSTEM_PROMPT.lower()
    has_examples = (
        "e.g." in prompt_lower
        or "example" in prompt_lower
        or "such as" in prompt_lower
    )
    assert has_examples, "Prompt should contain examples to guide the LLM"


def test_prompt_mentions_personal_type():
    """Prompt must link identity to the PERSONAL type."""
    assert "personal" in PREFERENCE_EXTRACTION_SYSTEM_PROMPT.lower(), \
        "Prompt must mention PERSONAL preference type"
