import pytest

from procurement import parser
from procurement.schemas import OfferCore


class DummyMessage:
    def __init__(self, parsed=None, content=None):
        self.parsed = parsed
        self.content = content


class DummyChoice:
    def __init__(self, message):
        self.message = message


class DummyCompletion:
    def __init__(self, message):
        self.choices = [DummyChoice(message)]


@pytest.mark.asyncio
async def test_parse_vendor_reply_with_mocked_llm(monkeypatch):
    payloads = iter(
        [
            {
                "price_per_unit": 100.5,
                "min_batch": 5,
                "lead_time_days": 14,
                "terms_text": "Net 15",
            },
            {
                # Нет цены/партии, проверяем, что вернется None и terms_text будет из письма.
                "lead_time_days": 7,
            },
        ]
    )

    def fake_chat(*args, **kwargs):
        data = next(payloads)
        return DummyCompletion(DummyMessage(parsed=data))

    monkeypatch.setattr(parser, "chat", fake_chat)

    first = await parser.parse_vendor_reply("Offer 1")
    second = await parser.parse_vendor_reply("Offer 2 raw text")

    assert isinstance(first, OfferCore)
    assert first.price_per_unit == 100.5
    assert first.min_batch == 5
    assert first.lead_time_days == 14
    assert first.terms_text == "Net 15"

    assert second.price_per_unit is None
    assert second.min_batch is None
    assert second.lead_time_days == 7
    assert second.terms_text == "Offer 2 raw text"
