from src.optimizer.budget import should_stop


def test_budget_stops_when_marginal_gain_is_low():
    hist = [{"pass_rate": 0.2, "cost_eur": 1.0}] + [
        {"pass_rate": 0.6 + i * 0.001, "cost_eur": 1.0} for i in range(4)
    ]
    assert should_stop(hist, min_gain_per_eur=0.01, window=3)


def test_budget_keeps_going_while_gaining():
    hist = [{"pass_rate": 0.2 + i * 0.1, "cost_eur": 0.5} for i in range(5)]
    assert not should_stop(hist, min_gain_per_eur=0.01, window=3)
