from __future__ import annotations

from ner import extract_entities


def test_extract_entities_catches_phone_email_and_sku():
    text = "Contact me at +7 999 123-45-67 or mail test@example.com about SKU-42."
    entities = extract_entities(text)
    types = {item["type"] for item in entities}
    assert {"phone", "email", "sku"} <= types
