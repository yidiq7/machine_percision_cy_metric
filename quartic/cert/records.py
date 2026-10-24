"""The record files of scripts/cert_run.py: the columns, and one run from one or more parts.

A record file holds `records` (one row for each final box, with the columns COLUMNS),
`target`, `columns`, `commit` (the git commit of the code of the run) and, for each file
of local forms that the run read, its reference center in `local_refs` and its SHA-256 in
`local_sha`. A run on several GPUs writes one file for each part, and `load` joins them.

The manifest of scripts/cert_local_values.py ties the files of local forms to the files
that define the potential: its header holds the SHA-256 of --data and --forms, and each
line the reference center, the SHA-256 of center_<r>.npz and its largest gap over its
bound, with all digits, so that every reader applies the same test gap <= 1.
"""
import hashlib

import numpy as np

COLUMNS = ('ref', 'depth', 'lo0', 'lo1', 'lo2', 'lo3', 'side', 'delta', 'bound', 'center',
           'polynomial', 'remainder', 'hmax', 'lam_big', 'order', 'a', 'grad_tau', 'grad_g',
           'lam_low', 'mu_up', 'mu_low', 'ric', 'k_tau', 'k_t', 'k_atlas', 'lam_atlas',
           'Lam_atlas')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def load(paths):
    """The records, the target, the commit and {ref: SHA-256} of the parts of one run.

    The parts must have the columns COLUMNS, one target and one commit, and no reference
    center may appear in two parts. A file of a run before the hashes has no commit and no
    hashes; it gives commit None and an empty dict.
    """
    recs, targets, commits, shas, seen = [], set(), set(), {}, set()
    for path in paths:
        z = np.load(path)
        if 'columns' not in z.files or tuple(str(c) for c in z['columns']) != COLUMNS:
            raise ValueError(f'{path}: the columns differ from quartic.cert.records.COLUMNS')
        rec = np.asarray(z['records'])
        if rec.ndim != 2 or rec.shape[1] != len(COLUMNS):
            raise ValueError(f'{path}: the records are not a table of {len(COLUMNS)} columns')
        targets.add(float(z['target']))
        commits.add(str(z['commit']) if 'commit' in z.files else None)
        part = {}
        if 'local_sha' in z.files:
            part = {int(r): str(s) for r, s in zip(z['local_refs'], z['local_sha'])}
        refs = set(np.unique(rec[:, 0]).astype(int).tolist()) | set(part)
        if refs & seen:
            raise ValueError(f'{path}: a reference center of this part is also in another part')
        seen |= refs
        shas.update(part)
        recs.append(rec)
    if len(targets) != 1 or len(commits) != 1:
        raise ValueError(f'the parts have targets {targets} and commits {commits}')
    return np.concatenate(recs), targets.pop(), commits.pop(), shas


def write_manifest(path, data, forms, rows):
    """The manifest for data = (path, SHA-256), forms = (path, SHA-256) and rows of
    (reference center, SHA-256 of its file, largest gap over its bound)."""
    with open(path, 'w') as fh:
        fh.write(f'# data {data[0]} {data[1]}\n# forms {forms[0]} {forms[1]}\n')
        fh.write('# reference, SHA-256 of center_<r>.npz, largest gap over its bound\n')
        for r, digest, gap in rows:
            fh.write(f'{r} {digest} {float(gap)!r}\n')


def read_manifest(path):
    """({'data': SHA-256, 'forms': SHA-256}, {ref: (SHA-256, gap)}) of a manifest."""
    head, rows = {}, {}
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == '#':
            if len(parts) == 4 and parts[1] in ('data', 'forms'):
                head[parts[1]] = parts[3]
            continue
        r, digest, gap = parts
        rows[int(r)] = (digest, float(gap))
    return head, rows


def manifest_errors(shas, refs, manifest, data_sha, forms_sha):
    """The defects of a run against a manifest (empty if none): the manifest must be that of
    the given --data and --forms, and every reference center with records, and every file
    that the run read, needs a hash equal to that of a file whose gap is at most 1."""
    head, rows = manifest
    errors = []
    if head.get('data') != data_sha or head.get('forms') != forms_sha:
        errors.append('the manifest is that of other files --data or --forms')
    bad = sorted(r for r in set(refs) | set(shas) if r not in shas or r not in rows
                 or rows[r][0] != shas[r] or not rows[r][1] <= 1)
    if bad:
        errors.append(f'{len(bad)} reference centers without a hash, with another hash, or '
                      f'with a failed check {bad[:5]}')
    return errors
