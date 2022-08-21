
def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    from .default import _C as cfg
    return cfg.clone()


cfg = get_cfg_defaults()
## Merging dynamic_filter configuration
cfg.merge_from_file("config/dynamic_filter/{}.yaml".format(cfg.DYNAMIC_FILTER.TAIL_MODEL.lower()))
cfg.merge_from_file("config/dynamic_filter/{}.yaml".format(cfg.DYNAMIC_FILTER.HEAD_MODEL.lower()))
cfg.merge_from_file("config/dynamic_filter/{}.yaml".format(cfg.HISA_QUERY.TAIL_MODEL.lower()))
cfg.merge_from_file("config/dynamic_filter/{}.yaml".format(cfg.HISA_VIDEO.TAIL_MODEL.lower()))
## Merging solver configuration
print(cfg.SOLVER.TYPE.lower())
cfg.merge_from_file("config/solver/{}.yaml".format(cfg.SOLVER.TYPE.lower()))
