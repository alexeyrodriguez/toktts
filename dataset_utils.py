import datasets
from datasets.utils.py_utils import Pickler, pklregister

# Needed for deterministic hashing of sets across python runs
# Not needed anymore when datasets is updated, see https://github.com/huggingface/datasets/pull/6318
@pklregister(set)
def _save_set(pickler, obj):
    from datasets.fingerprint import Hasher

    args = (sorted(obj, key=Hasher.hash),)
    pickler.save_reduce(set, args, obj=obj)


def cut_rows(ds, n_rows):
    n = min(len(ds), n_rows)
    return ds.select(range(n))

def map_list(ds, f, col, num_proc=None):
    """
    Adds a column `col` to a dataset where rows may be duplicated if `f` returns multiple values
    Easier to use than the low level batching interface
    Possibly this introduces a lot of overhead, so only use when the function is expensive. Todo, optimize?
    """
    def map_fn(batch):
        vals = f({ k: v[0] for k, v in batch.items() })
        d = { k: [v[0] for _ in vals] for k, v in batch.items() }
        d.update({ col: vals})
        return d
    return ds.map(map_fn, batched=True, batch_size=1, num_proc=num_proc)

