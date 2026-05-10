from tnfx_full_model.forward_pricing import cip_forward, executable_terms


def test_cip_and_sides():
    f = cip_forward(3.39, 0.08, 0.03, 180)
    assert abs(f - 3.39 * (1 + 0.08 * 180 / 360) / (1 + 0.03 * 180 / 360)) < 1e-12
    row = dict(direction="outflow", spot_ask=3.39, spot_bid=3.37, F_CIP_ask=f, F_CIP_bid=3.4, forward_ask=float("nan"), forward_bid=float("nan"))
    assert executable_terms(row)["pricing_side"] == "ask"
    row["direction"] = "inflow"
    assert executable_terms(row)["pricing_side"] == "bid"

