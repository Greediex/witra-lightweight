"""Each control changes only its declared factor relative to its reference."""
BASE = dict(conditioning="attached", loss_weighting="uw", readout="three",
            coordinate_head="expectation", normalization="mean_center", online_augmentation=True)
VARIANTS = {name: {**BASE, **changes} for name, changes in {
    "MAIN": {}, "DETACH": {"conditioning": "detached"},
    "UNIFORM": {"conditioning": "uniform"}, "EQUAL": {"loss_weighting": "equal"},
    "SINGLE_TOKEN": {"readout": "single"}, "NO_FLOOR": {"conditioning": "none"},
    "GAUSSIAN": {"conditioning": "gaussian"}, "DEPTH4": {"depth_override": 4},
    "DIRECT": {"coordinate_head": "direct"},
    "DIRECT_CLAMP": {"coordinate_head": "direct_clamp"},
    "NO_ONLINE_AUG": {"online_augmentation": False},
    "POWER_NO_ONLINE_AUG": {"normalization": "power", "online_augmentation": False},
}.items()}
CORE = ["MAIN", "DETACH", "UNIFORM", "EQUAL"]
REFERENCES = {name: "MAIN" for name in VARIANTS if name != "MAIN"}
REFERENCES["POWER_NO_ONLINE_AUG"] = "NO_ONLINE_AUG"
