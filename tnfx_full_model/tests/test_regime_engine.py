from tnfx_full_model.regime_engine import is_feasible_regime


def state(**kw):
    base = dict(g_NR=0, g_PAE=0, g_TE=0, g_AEO=0, g_ACC=0, g_EXT=0, g_CIRC=0)
    base.update(kw)
    return base


def test_regime_rules():
    assert not is_feasible_regime("exporter", state(g_CIRC=1))
    assert is_feasible_regime("importer", state(g_CIRC=1))
    assert is_feasible_regime("processor", state(g_CIRC=1))
    assert is_feasible_regime("trader", state(g_CIRC=1))
    assert not is_feasible_regime("trader", state(g_TE=1, g_PAE=0))
    assert not is_feasible_regime("processor", state(g_NR=1, g_PAE=0))
    assert not is_feasible_regime("exporter", state(g_PAE=1, g_ACC=0))
    assert not is_feasible_regime("importer", state(g_PAE=1, g_ACC=1))


def test_g_CIRC_zero_for_exporter():
    bad = {"g_NR": 0, "g_PAE": 1, "g_TE": 0, "g_AEO": 0, "g_ACC": 1, "g_EXT": 0, "g_CIRC": 1}
    assert not is_feasible_regime("exporter", bad)


def test_g_CIRC_allowed_for_processor_trader_importer():
    for fam in ["importer", "processor", "trader"]:
        good = {"g_NR": 0, "g_PAE": 0, "g_TE": 0, "g_AEO": 0, "g_ACC": 0, "g_EXT": 0, "g_CIRC": 1}
        assert is_feasible_regime(fam, good)
