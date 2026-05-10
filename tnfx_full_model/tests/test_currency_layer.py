from tnfx_full_model.currency_layer import decompose_currency


def test_currency_decomposition_identities():
    p = dict(h_R=0.7, h_C=0.6, r=0.1, beta=0.2, fx_debt_service_share=0.04,
             alpha_R_EUR=0.6, alpha_C_EUR=0.5, alpha_D_EUR=0.75, g_EXT=1)
    d = decompose_currency(p)
    assert abs(d["alpha_R_USD"] + p["alpha_R_EUR"] - 1) < 1e-12
    assert abs(d["delta_CF_EUR"] + d["delta_CF_USD"] - (p["h_R"] - p["h_C"] * (1 - p["r"]))) < 1e-10
    assert abs(d["delta_net_EUR"] + d["delta_net_USD"] - d["delta_net_total"]) < 1e-10
    p["g_EXT"] = 0
    d2 = decompose_currency(p)
    assert d2["fx_debt_service_share_EUR"] == 0
    assert d2["fx_debt_service_share_USD"] == 0

