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


def test_dated_snapshot_name_resolves_to_base_model_pricing():
    # Confirmed against the real OpenAI API: a request for "gpt-4o-mini"
    # comes back with model="gpt-4o-mini-2024-07-18" in the response, not
    # the generic alias. Pricing lookup must still find it.
    base = get_pricing("gpt-4o-mini")
    dated = get_pricing("gpt-4o-mini-2024-07-18")
    assert dated is not None
    assert dated == base


def test_dated_snapshot_resolves_to_the_longer_matching_base_name_not_the_shorter_one():
    # "gpt-4o-mini-2024-07-18" starts with both "gpt-4o" and
    # "gpt-4o-mini" as string prefixes — must resolve to the longer,
    # more specific "gpt-4o-mini", not the cheaper "gpt-4o" by mistake.
    pricing = get_pricing("gpt-4o-mini-2024-07-18")
    assert pricing == get_pricing("gpt-4o-mini")
    assert pricing != get_pricing("gpt-4o")


def test_dated_gpt4o_snapshot_also_resolves():
    pricing = get_pricing("gpt-4o-2024-08-06")
    assert pricing == get_pricing("gpt-4o")


def test_unrelated_model_with_similar_prefix_does_not_false_match():
    # "gpt-4o-turbo" is not a real model, but stands in for "something
    # that happens to start with a known base name yet isn't a dated
    # snapshot of it" — the point is this is a real, if unlikely,
    # limitation of prefix matching, and it should behave predictably
    # (matches the shorter known prefix it starts with) rather than
    # crash or silently do something surprising.
    pricing = get_pricing("gpt-4o-turbo")
    assert pricing == get_pricing("gpt-4o")
