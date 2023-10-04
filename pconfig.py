import yaml

def load_config(paths):
    paths = paths if isinstance(paths, list) else [paths]
    cs = []
    for p in paths:
        with open(p, 'r') as f:
            d = yaml.safe_load(f)
            cs.append(d)
    return dict_to_namespace(merge_dicts(cs))

def merge_dicts(dicts):
    d = {}
    for od in dicts:
        d = merge_dict(d, od)
    return d

def merge_dict(d1, d2):
    if isinstance(d1, dict) and isinstance(d2, dict):
        d = {}
        for k in d1.keys():
            d[k] = merge_dict(d1[k], d2.get(k, None))
        for k in d2.keys():
            d[k] = merge_dict(d1.get(k, None), d2[k])
        return d
    elif d1 is None:
        return d2
    elif d2 is None:
        return d1
    else:
        return d2 # second argument does overriding

def dict_to_namespace(d):
    from types import SimpleNamespace
    if isinstance(d, dict):
        d = {k: dict_to_namespace(d[k]) for k in d.keys()}
        return SimpleNamespace(**d)
    else:
        return d

def namespace_to_dict(ns):
    from types import SimpleNamespace
    if isinstance(ns, SimpleNamespace):
        d = ns.__dict__
        return {k: namespace_to_dict(d[k]) for k in d.keys()}
    else:
        return ns

def model_path(path_from_config, path_from_args):
    if not path_from_args:
        return 'models/' + path_from_config
    else:
        return path_from_args

#load_config(['config/small_train.yaml', 'config/cloud_cpu.yaml'])