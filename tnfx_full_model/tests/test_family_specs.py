from tnfx_full_model.family_specs import load_families


def test_families_load_and_forbidden_keys(config):
    families = load_families(config.model)
    assert set(families) == {"importer", "exporter", "processor", "trader"}
    assert families["importer"]["beta"] == [0.75, 0.90]
    for spec in families.values():
        assert "pi_F" not in spec
        assert "kappa" not in spec

