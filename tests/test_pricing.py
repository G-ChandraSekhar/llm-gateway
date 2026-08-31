from app.core.pricing import compute_cost_micros, get_pricing


def test_known_model_returns_pricing():
    pricing = get_pricing("gpt-4o-mini")
    assert pricing is not None
    assert pricing.input_usd_per_million == 0.15
    assert pricing.output_usd_per_million == 0.60


def test_unknown_model_returns_none():
    assert get_pricing("some-model-that-does-not-exist") is None


def test_compute_cost_micros_matches_hand_calculation():
    # 15 prompt tokens + 8 completion tokens on gpt-4o-mini ($0.15/$0.60 per million)
    cost = compute_cost_micros("gpt-4o-mini", prompt_tokens=15, completion_tokens=8)
    expected_usd = (15 * 0.15 + 8 * 0.60) / 1_000_000
    assert cost == round(expected_usd * 1_000_000)
    assert cost == 7  # hand-verified: 2.25 + 4.8 = 7.05 millionths of a dollar -> rounds to 7


def test_compute_cost_micros_returns_none_for_unpriced_model():
    assert compute_cost_micros("some-unpriced-model", prompt_tokens=100, completion_tokens=100) is None


def test_gpt4o_is_pricier_than_mini_for_same_usage():
    cost_4o = compute_cost_micros("gpt-4o", prompt_tokens=1000, completion_tokens=1000)
    cost_mini = compute_cost_micros("gpt-4o-mini", prompt_tokens=1000, completion_tokens=1000)
    assert cost_4o > cost_mini
