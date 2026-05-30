# Improved 7-Model Singular Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `thesis_test_singular_improved.ipynb` — the 7 dual-branch variants retrained through one upgraded shared harness (AdamW + warmup→cosine + label smoothing + SWA + best-by-val-F1 + AMP), with ONE shared Optuna search on the Full Model whose HPs are locked and reused by all 7.

**Architecture:** A Python generator script (`_build_singular_improved.py`) emits the notebook JSON. Model classes + data pipeline are copied verbatim from `thesis_test_singular_derm7pt.ipynb`; the new training harness, the shared Optuna search, and the comparison cells are authored fresh. Validation runs by exec-ing the emitted code cells in a subprocess (compile + functional micro-tests on MPS with `pretrained=False`, then a SMOKE end-to-end run).

**Tech Stack:** PyTorch (MPS/CUDA), torchvision ResNet50, Optuna (TPE + MedianPruner, SQLite storage), sklearn metrics, Jupyter nbformat 4.

**Spec:** `docs/superpowers/specs/2026-05-30-singular-7model-improved-design.md`

---

## Source-of-truth cell map (in `thesis_test_singular_derm7pt.ipynb`)

| Cell | Reuse |
|------|-------|
| 2 | imports + seed + device |
| 4 | DATASET SELECTOR toggle |
| 5 | data assembly + `SkinLesionDualDataset` + transforms + `build_dataset` + `make_loaders` |
| 7 | `ResNet50Backbone`, `EarlyStopping`, `train_one_epoch`, `validate`, `run_training` (OLD — discard), `evaluate`; fixed HPs |
| 8 | `evaluate_checkpoint_quiet`, `render_comparison`, `mini_compare` |
| 10 | `SingleBranchRGBClassifier` |
| 13 | `DualBranchBaseline` |
| 18 | `SEBlock`, `SEBottleneck`, `SEResNet50Backbone`, `DualBranchSEResNet` |
| 23 | `DualBranchElementwiseFusion` |
| 28 | `CrossAttentionFusion`, `BidirectionalCrossAttnFusion`, `DualBranchBiCrossAttn` |
| 33 | `DualBranchSECrossCombined` |

Model-definition cells (10,13,18,23,28,33) may have trailing scaffold (param-count prints, `_m = ...`). The generator strips everything from the first column-0 line that is **not** `class`/`def`/`@`/`#`/blank after the first class begins.

---

## File Structure

- **Create:** `_build_singular_improved.py` — generator (deleted after final notebook is produced).
- **Create:** `thesis_test_singular_improved.ipynb` — the deliverable.
- **Create (transient):** `_val_singular.py` — validation harness used during development (deleted at end).
- **Read-only:** `thesis_test_singular_derm7pt.ipynb` (source cells).
- **Runtime artifacts (produced when the notebook runs, not by this plan):** `optuna_singular_{ds}.db`, `singular_bestparams_{ds}.json`, `thesis_*_improved_{ds}_best.pth`, `comparison_improved_{ds}.csv`, `comparison_improved_{ds}.png`.

---

## Task 1: Generator skeleton + setup/toggle/loaders cells

**Files:**
- Create: `_build_singular_improved.py`

- [ ] **Step 1: Write the generator skeleton with a cell-extraction helper**

```python
"""Generator for thesis_test_singular_improved.ipynb. Run, validate, then delete."""
import json, re

SRC = 'thesis_test_singular_derm7pt.ipynb'
src_nb = json.load(open(SRC))
def cell_src(i): return ''.join(src_nb['cells'][i]['source'])

def classes_only(src):
    """Keep imports/classes/defs; drop trailing scaffold (param prints, _m = ...)."""
    lines = src.split('\n')
    out, seen_class = [], False
    for ln in lines:
        if re.match(r'^(class|def)\s', ln): seen_class = True
        # after classes started, a column-0 line that's not class/def/@/#/blank = scaffold -> stop
        if seen_class and ln and not ln[0].isspace() and not re.match(r'^(class|def|@|#|import|from)\s', ln) and not re.match(r'^[A-Z_]+\s*=', ln) is None:
            pass
        if seen_class and re.match(r'^(_m\s*=|print\()', ln):
            break
        out.append(ln)
    return '\n'.join(out).rstrip()

cells = []
def md(s): cells.append(('markdown', s))
def code(s): cells.append(('code', s))

# ---- Cell: title ----
md(r'''# Thesis — Improved 7-Model Singular Comparison

All 7 dual-branch variants trained through one upgraded shared harness
(AdamW + warmup->cosine + label smoothing + SWA + best-by-val-F1 + AMP), with a single
shared Optuna search on the Full Model whose hyperparameters are locked and reused by all 7
(fair architecture comparison). Image-only (clinical + dermoscopic), custom 70/15/15 split.
Toggle DATASET_DIR for Derm7pt or Milk10k; artifacts suffixed `_improved_{dataset}`.''')

# ---- Cell: imports (verbatim) + optuna/amp/swa ----
imports = cell_src(2).rstrip() + '''

import json, math
from torch.optim.swa_utils import AveragedModel, SWALR
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
optuna.logging.set_verbosity(optuna.logging.WARNING)
print('optuna', optuna.__version__)'''
md('## 1 — Setup')
code(imports)

# ---- Cell: dataset toggle (verbatim) ----
md('### Dataset selector')
code(cell_src(4))

# ---- Cell: data assembly + loaders (verbatim, add drop_last + enable Windows workers) ----
loaders = cell_src(5)
loaders = loaders.replace(
    "_NUM_WORKERS = 0 if device.type in ('mps', 'cpu') or os.name == 'nt' else min(8, os.cpu_count() or 4)",
    "_NUM_WORKERS = 0 if device.type in ('mps', 'cpu') else min(8, os.cpu_count() or 4)  # Windows/CUDA gets workers")
loaders = loaders.replace(
    "train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=samp, **kw)",
    "train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=samp, drop_last=True, **kw)")
code(loaders)

# (Tasks 2-7 append more cells here)

# ---- assemble ----
def emit():
    nbc = []
    for t, s in cells:
        ls = s.split('\n'); srclist = [l+'\n' for l in ls[:-1]] + [ls[-1]]
        if t == 'markdown':
            nbc.append({'cell_type':'markdown','metadata':{},'source':srclist})
        else:
            nbc.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':srclist})
    nb = {'cells':nbc,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
          'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
    json.dump(nb, open('thesis_test_singular_improved.ipynb','w'), indent=1)
    print('wrote thesis_test_singular_improved.ipynb', len(nbc), 'cells')
emit()
```

- [ ] **Step 2: Run the generator**

Run: `python3 _build_singular_improved.py`
Expected: `wrote thesis_test_singular_improved.ipynb 7 cells`

- [ ] **Step 3: Verify the notebook parses and code cells compile**

Run:
```bash
python3 -c "import json; nb=json.load(open('thesis_test_singular_improved.ipynb')); [compile(''.join(c['source']),'x','exec') for c in nb['cells'] if c['cell_type']=='code']; print('compile OK', len(nb['cells']),'cells')"
```
Expected: `compile OK 7 cells`

- [ ] **Step 4: Commit**

```bash
git add _build_singular_improved.py thesis_test_singular_improved.ipynb
git commit -m "feat(notebook): scaffold improved 7-model notebook (setup/toggle/loaders)"
```

---

## Task 2: Backbone + all 7 model classes (copied verbatim)

**Files:**
- Modify: `_build_singular_improved.py` (append before `# ---- assemble ----`)

- [ ] **Step 1: Append the backbone + model-class cells**

Insert this block in the generator at the `(Tasks 2-7 append more cells here)` marker:

```python
# ---- Cell: backbone + fixed constants ----
md('## 2 — Backbone, constants, model classes')
# Cell 7 holds ResNet50Backbone + helpers + fixed HPs. Keep ONLY the backbone class
# and the fixed-HP constants; the old run_training/evaluate are replaced in Task 3.
c7 = cell_src(7)
backbone = c7[c7.index('class ResNet50Backbone'): c7.index('class EarlyStopping')].rstrip()
# fixed HP block lives after the backbone class; grab the LR..GRAD_CLIP lines
hp_block = '\n'.join(l for l in c7.split('\n') if re.match(r'^(LR|DROPOUT|WEIGHT_DECAY|BATCH_SIZE|FINAL_EPOCHS|PATIENCE|UNFREEZE_EPOCH|GRAD_CLIP)\s*=', l))
extra = '''WARMUP_EPOCHS  = 3
SWA_START_FRAC = 0.75
USE_AMP        = (device.type == 'cuda')   # mixed precision, CUDA only (no-op on MPS/CPU)
NC = len(class_names)'''
code(backbone + '\n\n# ── Fixed structural constants ──\n' + hp_block + '\n' + extra)

# ---- Cells: the 7 model classes (verbatim, scaffold stripped) ----
md('### Model classes (verbatim from the baseline notebook)')
for idx in (10, 13, 18, 23, 28, 33):
    code(classes_only(cell_src(idx)))
```

- [ ] **Step 2: Regenerate**

Run: `python3 _build_singular_improved.py`
Expected: `wrote thesis_test_singular_improved.ipynb 15 cells`

- [ ] **Step 3: Write the validation harness (functional test of all 7 models)**

Create `_val_singular.py`:
```python
import json, torch, torch.nn as nn
nb = json.load(open('thesis_test_singular_improved.ipynb'))
cc = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
ns = {}
# exec everything up to and including the model classes (skip the dataset cells that hit disk)
for c in cc:
    if 'DATASET SELECTOR' in c or 'Dataset-specific dataframe assembly' in c:
        continue
    if any(k in c for k in ['import os','class ResNet50Backbone','class SingleBranchRGBClassifier',
            'class DualBranchBaseline','class SEBlock','class DualBranchElementwiseFusion',
            'class CrossAttentionFusion','class DualBranchSECrossCombined']):
        exec(c, ns)
ns['class_names'] = ['BCC','MEL','MISC','NV','SK']
device = ns['device']
NC = 5
builders = [
    ('Single-RGB',    lambda: ns['SingleBranchRGBClassifier'](num_classes=NC, dropout=0.32)),
    ('Dual-Concat',   lambda: ns['DualBranchBaseline'](num_classes=NC, dropout=0.32)),
    ('SE-Concat',     lambda: ns['DualBranchSEResNet'](num_classes=NC, dropout=0.32)),
    ('Add-Fusion',    lambda: ns['DualBranchElementwiseFusion'](num_classes=NC, dropout=0.32, fusion='add')),
    ('Mul-Fusion',    lambda: ns['DualBranchElementwiseFusion'](num_classes=NC, dropout=0.32, fusion='mul')),
    ('BiCrossAttn',   lambda: ns['DualBranchBiCrossAttn'](num_classes=NC, dropout=0.32)),
    ('Full',          lambda: ns['DualBranchSECrossCombined'](num_classes=NC, dropout=0.32)),
]
c = torch.randn(2,3,224,224).to(device); d = torch.randn(2,3,224,224).to(device)
for name, build in builders:
    m = build().to(device)
    # all models built with pretrained backbones download once; force eval forward+backward
    out = m(c, d)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (2, NC), (name, logits.shape)
    logits.sum().backward()
    print(f'  {name:12s} OK logits={tuple(logits.shape)}')
print('ALL 7 MODELS forward/backward OK')
```

NOTE: the model constructors default `pretrained=True` → first run downloads ResNet50 weights (~100MB). To skip the download in validation, the harness may pass `pretrained=False` if the constructor accepts it — check each signature; all 7 accept `pretrained`. Update each builder lambda to add `pretrained=False`.

- [ ] **Step 4: Run validation**

Run: `python3 _val_singular.py`
Expected: 7 lines `... OK logits=(2, 5)` then `ALL 7 MODELS forward/backward OK`

- [ ] **Step 5: Commit**

```bash
git add _build_singular_improved.py thesis_test_singular_improved.ipynb _val_singular.py
git commit -m "feat(notebook): add backbone + 7 model classes (verbatim)"
```

---

## Task 3: Improved training harness

**Files:**
- Modify: `_build_singular_improved.py`

- [ ] **Step 1: Append the helpers + `run_training_improved` cell**

Insert in the generator (after the model-class loop):

```python
md('## 3 — Improved shared trainer')

# keep the verbatim `validate` (val loss/acc) from cell 7
c7 = cell_src(7)
validate_fn = c7[c7.index('def validate('): c7.index('def run_training(')].rstrip()

trainer = validate_fn + '''


def warmup_cosine_lambda(warmup_epochs, phase_epochs):
    def lr_lambda(ep):
        if warmup_epochs and ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        prog = (ep - warmup_epochs) / max(1, phase_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return lr_lambda


@torch.no_grad()
def val_f1_macro(model, loader):
    model.eval()
    preds, ys = [], []
    for clinic, derm, y in loader:
        out = model(clinic.to(device), derm.to(device))
        logits = out[0] if isinstance(out, tuple) else out
        preds.extend(logits.argmax(1).cpu().numpy()); ys.extend(y.numpy())
    return f1_score(np.array(ys), np.array(preds), average='macro', zero_division=0)


def update_bn_dual(loader, model):
    """Recompute BatchNorm running stats for a two-input model (torch update_bn is single-input)."""
    momenta = {}
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats(); momenta[mod] = mod.momentum; mod.momentum = None
    if not momenta:
        return
    was = model.training; model.train()
    with torch.no_grad():
        for clinic, derm, _ in loader:
            model(clinic.to(device), derm.to(device))
    for bn, m in momenta.items(): bn.momentum = m
    model.train(was)


def run_training_improved(model, ckpt_path, tag, train_loader, val_loader, loss_weights, hp,
                          total_epochs=FINAL_EPOCHS, patience=PATIENCE, use_swa=True,
                          trial=None, verbose=True):
    """Shared improved trainer for all 7 models. hp=dict(lr, weight_decay, dropout, label_smoothing).
    AdamW + warmup->cosine + label-smoothed weighted CE + SWA + best-by-val-F1 + AMP.
    If `trial` is given, reports val-F1 each epoch for Optuna pruning."""
    model = model.to(device); model._freeze_backbones()
    lr, wd, ls = hp['lr'], hp['weight_decay'], hp['label_smoothing']
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=ls)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    def build_os(params, base_lr, phase_ep, warmup):
        opt = optim.AdamW(list(params), lr=base_lr, weight_decay=wd)
        if warmup and warmup > 0:
            sched = optim.lr_scheduler.LambdaLR(opt, warmup_cosine_lambda(warmup, phase_ep))
        else:
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, phase_ep))
        return opt, sched
    opt, sched = build_os(filter(lambda p: p.requires_grad, model.parameters()),
                          lr, UNFREEZE_EPOCH, WARMUP_EPOCHS)

    swa_model = AveragedModel(model) if use_swa else None
    swa_start = int(SWA_START_FRAC * total_epochs); swa_sched = None; swa_n = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_f1, best_state, early_counter, early_stop_epoch = -1.0, None, 0, None
    start = time.time(); last_epoch = 0

    for epoch in range(total_epochs):
        last_epoch = epoch
        if epoch == UNFREEZE_EPOCH:
            model.unfreeze_resnets()
            opt, sched = build_os(model.parameters(), lr / 4, total_epochs - UNFREEZE_EPOCH, 0)
            if use_swa: swa_sched = SWALR(opt, swa_lr=lr / 8)

        model.train(); tl = tc = tn = 0.0
        for clinic, derm, y in train_loader:
            clinic, derm, y = clinic.to(device), derm.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type='cuda', enabled=USE_AMP):
                out = model(clinic, derm)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, y)
            if USE_AMP:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
            tl += loss.item() * clinic.size(0); tc += logits.argmax(1).eq(y).sum().item(); tn += clinic.size(0)
        tr_loss, tr_acc = tl / tn, tc / tn
        va_loss, va_acc = validate(model, val_loader, criterion)
        va_f1 = val_f1_macro(model, val_loader)

        in_swa = use_swa and epoch >= swa_start
        if in_swa:
            swa_model.update_parameters(model); swa_n += 1
            if swa_sched is not None: swa_sched.step()
        else:
            sched.step()

        history['train_loss'].append(tr_loss); history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc); history['val_acc'].append(va_acc)
        history['val_f1'].append(va_f1)

        if va_f1 > best_f1:
            best_f1, best_state, early_counter = va_f1, copy.deepcopy(model.state_dict()), 0
        else:
            early_counter += 1
        if verbose:
            tag2 = 'SWA' if in_swa else f'es {early_counter}/{patience}'
            print(f'  Ep {epoch+1:2d}/{total_epochs} | train {tr_loss:.4f}/{tr_acc:.4f} | '
                  f'val {va_loss:.4f}/{va_acc:.4f} | f1 {va_f1:.4f} | {tag2}')
        if trial is not None:
            trial.report(va_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if (not in_swa) and early_counter >= patience:
            early_stop_epoch = epoch + 1
            if verbose: print(f'  -> early stop {early_stop_epoch}')
            break

    # finalize: best-F1 snapshot vs SWA-averaged (only if SWA actually ran), keep higher val-F1
    candidates = []
    if best_state is not None:
        model.load_state_dict(best_state); candidates.append(('best_f1', model, best_f1))
    if use_swa and swa_n > 0:
        update_bn_dual(train_loader, swa_model)
        candidates.append(('swa', swa_model.module, val_f1_macro(swa_model.module, val_loader)))
    name, chosen, chosen_f1 = max(candidates, key=lambda c: c[2])
    save_state = copy.deepcopy(chosen.state_dict())
    torch.save({'model_state_dict': save_state, 'class_names': class_names, 'label_map': label_map,
                'history': history, 'best_val_f1': chosen_f1, 'selected': name,
                'early_stop_epoch': early_stop_epoch, 'hp': hp}, ckpt_path)
    if verbose:
        print(f'{tag}: {(time.time()-start)/60:.1f} min | selected {name} | val f1 {chosen_f1:.4f} | saved {ckpt_path}')
    return history
'''
code(trainer)
```

- [ ] **Step 2: Regenerate**

Run: `python3 _build_singular_improved.py`
Expected: `wrote thesis_test_singular_improved.ipynb 17 cells`

- [ ] **Step 3: Micro-run the trainer on one model (MPS, 2 fake epochs)**

Append to `_val_singular.py` (new block at end) and re-run:
```python
# --- trainer micro-run ---
import torch, copy, time, math, numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR
import torch.optim as optim
from sklearn.metrics import f1_score
# exec the trainer cell + constants
for c in cc:
    if 'def run_training_improved' in c or 'WARMUP_EPOCHS' in c or 'def validate(' in c:
        exec(c, ns)
ns.update({'class_names':['BCC','MEL','MISC','NV','SK'],'label_map':{n:i for i,n in enumerate(['BCC','MEL','MISC','NV','SK'])},
           'FINAL_EPOCHS':3,'PATIENCE':9,'UNFREEZE_EPOCH':1,'WARMUP_EPOCHS':1,'GRAD_CLIP':1.0,
           'SWA_START_FRAC':0.66,'USE_AMP':False,'BATCH_SIZE':4})
class FakeLoader:
    def __init__(self,n=2): self.n=n
    def __iter__(self):
        for _ in range(self.n):
            yield (torch.randn(4,3,224,224),torch.randn(4,3,224,224),torch.randint(0,5,(4,)))
m = ns['DualBranchSECrossCombined'](num_classes=5, dropout=0.32, pretrained=False)
lw = torch.ones(5).to(ns['device'])
hp = dict(lr=7.5e-4, weight_decay=1e-4, dropout=0.32, label_smoothing=0.1)
hist = ns['run_training_improved'](m, '/tmp/_t.pth', 'micro', FakeLoader(), FakeLoader(), lw, hp,
                                   total_epochs=3, patience=9, use_swa=True, verbose=True)
import os; assert os.path.exists('/tmp/_t.pth'); os.remove('/tmp/_t.pth')
assert len(hist['val_f1'])>=1
print('TRAINER micro-run OK (SWA + best-F1 + checkpoint save)')
```

Run: `python3 _val_singular.py`
Expected: per-epoch lines incl an `SWA` epoch, then `TRAINER micro-run OK ...`

- [ ] **Step 4: Commit**

```bash
git add _build_singular_improved.py thesis_test_singular_improved.ipynb _val_singular.py
git commit -m "feat(notebook): add run_training_improved (AdamW+warmup+LS+SWA+best-F1+AMP)"
```

---

## Task 4: Shared Optuna search on the Full Model

**Files:**
- Modify: `_build_singular_improved.py`

- [ ] **Step 1: Append the search cell**

```python
md('''## 4 — Shared hyperparameter search (Full Model)

ONE Optuna search tunes lr / weight_decay / dropout / label_smoothing on the Full Model
(the thesis contribution). Best HPs are locked to JSON and reused by ALL 7 models so the
architecture comparison stays fair. Resumable SQLite study; warm-started with the
already-tuned baseline. Set SEARCH=False to skip and reuse a previously locked JSON.''')

code('''SEARCH        = True          # False -> load existing bestparams JSON, skip search
N_TRIALS      = 15
SEARCH_EPOCHS = 25
SEARCH_PAT    = 10
STUDY_DB      = f'sqlite:///optuna_singular_{DATASET_NAME}.db'
STUDY_NAME    = f'singular_improved_{DATASET_NAME}'
BESTPARAMS    = f'singular_bestparams_{DATASET_NAME}.json'

WARM_START = {'lr': 0.00075, 'weight_decay': 0.000957, 'dropout': 0.32, 'label_smoothing': 0.1}

def hp_objective(trial):
    hp = {
        'lr':              trial.suggest_float('lr', 1e-4, 3e-3, log=True),
        'weight_decay':    trial.suggest_float('weight_decay', 1e-5, 3e-3, log=True),
        'dropout':         trial.suggest_float('dropout', 0.2, 0.5),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.15),
    }
    tr, va, _, _ = make_loaders(BATCH_SIZE, ACTIVE_DS)
    model = DualBranchSECrossCombined(num_classes=NC, dropout=hp['dropout'], pretrained=True)
    hist = run_training_improved(model, f'/tmp/_search_{DATASET_NAME}.pth', f'trial{trial.number}',
                                 tr, va, ACTIVE_DS['loss_weights'], hp,
                                 total_epochs=SEARCH_EPOCHS, patience=SEARCH_PAT,
                                 use_swa=True, trial=trial, verbose=False)
    return max(hist['val_f1'])

if SEARCH:
    study = optuna.create_study(study_name=STUDY_NAME, storage=STUDY_DB, direction='maximize',
                                sampler=TPESampler(seed=SEED, multivariate=True),
                                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=8),
                                load_if_exists=True)
    if len(study.trials) == 0:
        study.enqueue_trial(WARM_START)
    remaining = max(0, N_TRIALS - len(study.trials))
    print(f'{STUDY_NAME}: {len(study.trials)} trials done, running {remaining} more...')
    if remaining > 0:
        study.optimize(hp_objective, n_trials=remaining, gc_after_trial=True)
    BEST_HP = study.best_params
    json.dump({'dataset': DATASET_NAME, 'best_val_f1': study.best_value, 'best_hp': BEST_HP},
              open(BESTPARAMS, 'w'), indent=2)
    print(f'Best val f1 {study.best_value:.4f} | {BEST_HP} | saved {BESTPARAMS}')
else:
    BEST_HP = json.load(open(BESTPARAMS))['best_hp']
    print(f'Loaded locked HPs: {BEST_HP}')''')
```

- [ ] **Step 2: Regenerate**

Run: `python3 _build_singular_improved.py`
Expected: `wrote thesis_test_singular_improved.ipynb 19 cells`

- [ ] **Step 3: Validate the search wiring with a stubbed trainer (fast, no GPU train)**

Append to `_val_singular.py` and re-run:
```python
# --- search wiring (stubbed trainer) ---
import optuna, json as _json, os
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
search_cell = next(c for c in cc if 'def hp_objective' in c)
ns2 = dict(ns)
ns2.update({'SEED':42,'BATCH_SIZE':4,'NC':5,'DATASET_NAME':'derm7pt',
            'ACTIVE_DS':{'loss_weights':None},'make_loaders':lambda b,ds:(None,None,None,None),
            'DualBranchSECrossCombined':lambda **k: None})
def fake_train(model, ckpt, tag, tr, va, lw, hp, total_epochs=25, patience=10, use_swa=True, trial=None, verbose=False):
    base = 0.6 + 0.1*(hp['label_smoothing']>0)
    for ep in range(total_epochs):
        if trial is not None:
            trial.report(base*(0.7+0.3*ep/total_epochs), ep)
            if trial.should_prune(): raise optuna.TrialPruned()
    return {'val_f1':[base]}
ns2['run_training_improved'] = fake_train
ns2['optuna']=optuna; ns2['TPESampler']=TPESampler; ns2['MedianPruner']=MedianPruner; ns2['json']=_json
if os.path.exists('/tmp/optuna_derm7pt.db'): os.remove('/tmp/optuna_derm7pt.db')
patched = search_cell.replace('N_TRIALS      = 15','N_TRIALS      = 4') \
    .replace("STUDY_DB      = f'sqlite:///optuna_singular_{DATASET_NAME}.db'","STUDY_DB='sqlite:////tmp/optuna_derm7pt.db'") \
    .replace("BESTPARAMS    = f'singular_bestparams_{DATASET_NAME}.json'","BESTPARAMS='/tmp/_bp.json'")
exec(patched, ns2)
assert os.path.exists('/tmp/_bp.json')
bp = _json.load(open('/tmp/_bp.json')); assert 'best_hp' in bp and 'lr' in bp['best_hp']
os.remove('/tmp/_bp.json'); os.remove('/tmp/optuna_derm7pt.db')
print('SEARCH wiring OK | best_hp keys:', list(bp['best_hp'].keys()))
```

Run: `python3 _val_singular.py`
Expected: `SEARCH wiring OK | best_hp keys: ['lr', 'weight_decay', 'dropout', 'label_smoothing']`

- [ ] **Step 4: Commit**

```bash
git add _build_singular_improved.py thesis_test_singular_improved.ipynb _val_singular.py
git commit -m "feat(notebook): shared Optuna search on Full Model, lock HPs to JSON"
```

---

## Task 5: Per-section training cells (all 7 use locked HPs)

**Files:**
- Modify: `_build_singular_improved.py`

- [ ] **Step 1: Append one train cell per model**

```python
md('## 5 — Train all 7 with locked HPs')

# (label, ckpt stem, builder-expression-as-string)
TRAIN_SPECS = [
    ('Single-RGB',        'thesis_singlebranch_rgb',   'SingleBranchRGBClassifier(num_classes=NC, dropout=BEST_HP["dropout"])'),
    ('Dual-Branch Concat','thesis_baseline_concat',    'DualBranchBaseline(num_classes=NC, dropout=BEST_HP["dropout"])'),
    ('SE-ResNet Concat',  'thesis_se_concat',          'DualBranchSEResNet(num_classes=NC, dropout=BEST_HP["dropout"])'),
    ('Add-Fusion',        'thesis_addfusion',          'DualBranchElementwiseFusion(num_classes=NC, dropout=BEST_HP["dropout"], fusion="add")'),
    ('Mul-Fusion',        'thesis_mulfusion',          'DualBranchElementwiseFusion(num_classes=NC, dropout=BEST_HP["dropout"], fusion="mul")'),
    ('BiCrossAttn',       'thesis_crossattn',          'DualBranchBiCrossAttn(num_classes=NC, dropout=BEST_HP["dropout"])'),
    ('Full (SE+Cross)',   'thesis_full_se_crossattn',  'DualBranchSECrossCombined(num_classes=NC, dropout=BEST_HP["dropout"])'),
]
for label, stem, builder in TRAIN_SPECS:
    code(f'''# {label}
ckpt = f'{stem}_improved_{{DATASET_NAME}}_best.pth'
if Path(ckpt).exists():
    print(f'[skip] {label}: {{ckpt}} exists')
else:
    tr_loader, va_loader, _, _ = make_loaders(BATCH_SIZE, ACTIVE_DS)
    _model = {builder}
    print('\\n=== {label} ===')
    run_training_improved(_model, ckpt, '{label}', tr_loader, va_loader,
                          ACTIVE_DS['loss_weights'], BEST_HP,
                          total_epochs=FINAL_EPOCHS, patience=PATIENCE, use_swa=True)
    del _model
    if torch.cuda.is_available(): torch.cuda.empty_cache()''')
```

- [ ] **Step 2: Regenerate**

Run: `python3 _build_singular_improved.py`
Expected: `wrote thesis_test_singular_improved.ipynb 27 cells`

- [ ] **Step 3: Verify compile + that each train cell references a valid builder**

Run:
```bash
python3 -c "
import json
nb=json.load(open('thesis_test_singular_improved.ipynb'))
codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
[compile(c,'x','exec') for c in codes]
trains=[c for c in codes if '_improved_{DATASET_NAME}_best.pth' in c]
assert len(trains)==7, len(trains)
print('compile OK | 7 train cells present')
"
```
Expected: `compile OK | 7 train cells present`

- [ ] **Step 4: Commit**

```bash
git add _build_singular_improved.py thesis_test_singular_improved.ipynb
git commit -m "feat(notebook): per-model training cells using locked shared HPs"
```

---

## Task 6: Grand comparison + old-vs-improved delta

**Files:**
- Modify: `_build_singular_improved.py`

- [ ] **Step 1: Append the comparison helpers (verbatim) + grand-comparison cell**

```python
md('## 6 — Grand comparison (7 models, held-out test) + old-vs-improved delta')

# reuse evaluate_checkpoint_quiet + render_comparison verbatim from baseline cell 8
code(cell_src(8))

code('''# Evaluate all 7 improved checkpoints on the held-out test set.
EVAL_SPECS = [
    ('Single-RGB',        SingleBranchRGBClassifier,   dict(num_classes=NC, dropout=BEST_HP['dropout']),                       'thesis_singlebranch_rgb'),
    ('Dual-Branch Concat',DualBranchBaseline,          dict(num_classes=NC, dropout=BEST_HP['dropout']),                       'thesis_baseline_concat'),
    ('SE-ResNet Concat',  DualBranchSEResNet,          dict(num_classes=NC, dropout=BEST_HP['dropout']),                       'thesis_se_concat'),
    ('Add-Fusion',        DualBranchElementwiseFusion, dict(num_classes=NC, dropout=BEST_HP['dropout'], fusion='add'),         'thesis_addfusion'),
    ('Mul-Fusion',        DualBranchElementwiseFusion, dict(num_classes=NC, dropout=BEST_HP['dropout'], fusion='mul'),         'thesis_mulfusion'),
    ('BiCrossAttn',       DualBranchBiCrossAttn,       dict(num_classes=NC, dropout=BEST_HP['dropout']),                       'thesis_crossattn'),
    ('Full (SE+Cross)',   DualBranchSECrossCombined,   dict(num_classes=NC, dropout=BEST_HP['dropout']),                       'thesis_full_se_crossattn'),
]
_, _, test_loader, _ = make_loaders(BATCH_SIZE, ACTIVE_DS)
results = {}
for label, cls, kwargs, stem in EVAL_SPECS:
    kwargs = dict(kwargs); kwargs['pretrained'] = False
    ckpt = f'{stem}_improved_{DATASET_NAME}_best.pth'
    m = evaluate_checkpoint_quiet(cls, kwargs, ckpt, test_loader)
    if m is None:
        print(f'  {label}: {ckpt} missing — skipped'); continue
    results[label] = m
    print(f'  {label:<20s} acc={m["accuracy"]:.4f} f1_macro={m["f1_macro"]:.4f} kappa={m["cohen_kappa"]:.4f}')

df_cmp = render_comparison(results, DATASET_NAME, 'Improved 7-model comparison', 'comparison_improved')
if df_cmp is not None:
    df_cmp.to_csv(f'comparison_improved_{DATASET_NAME}.csv')
    print(f'Saved comparison_improved_{DATASET_NAME}.csv')''')

code('''# Old-vs-improved delta (reads the baseline notebook's comparison CSV if present).
old_csv = f'comparison_metrics_{DATASET_NAME}.csv'
if Path(old_csv).exists() and df_cmp is not None:
    old = pd.read_csv(old_csv, index_col=0)
    common = [i for i in df_cmp.index if i in old.index]
    delta = (df_cmp.loc[common, ['accuracy','f1_macro','cohen_kappa']]
             - old.loc[common, ['accuracy','f1_macro','cohen_kappa']])
    delta.columns = ['Δacc','Δf1_macro','Δkappa']
    print('Improved minus baseline:')
    print(delta.round(4).to_string())
    delta.to_csv(f'delta_improved_{DATASET_NAME}.csv')
else:
    print(f'(no baseline {old_csv} found — skipping delta)')''')
```

- [ ] **Step 2: Regenerate + compile check**

Run: `python3 _build_singular_improved.py && python3 -c "import json; nb=json.load(open('thesis_test_singular_improved.ipynb')); [compile(''.join(c['source']),'x','exec') for c in nb['cells'] if c['cell_type']=='code']; print('compile OK', len(nb['cells']),'cells')"`
Expected: `compile OK 31 cells`

- [ ] **Step 3: Commit**

```bash
git add _build_singular_improved.py thesis_test_singular_improved.ipynb
git commit -m "feat(notebook): grand comparison + old-vs-improved delta table"
```

---

## Task 7: Optional aux-contrastive toggle (OFF by default)

**Files:**
- Modify: `_build_singular_improved.py`

- [ ] **Step 1: Append the contrastive add-on cell (and wire an optional term into the trainer)**

First, extend `run_training_improved` to support an optional contrastive term. In the trainer cell string (Task 3), the loss line currently is:
```python
                loss = criterion(logits, y)
```
Replace it (in the generator, via `.replace`) so it adds the term when the model exposes `.project` and `hp` carries a positive `contrastive_lambda`:
```python
trainer = trainer.replace(
"""                out = model(clinic, derm)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, y)""",
"""                out = model(clinic, derm)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, y)
                clam = hp.get('contrastive_lambda', 0.0)
                if clam > 0 and hasattr(model, 'project'):
                    zc, zd = model.project(clinic, derm)
                    loss = loss + clam * nt_xent_loss(zc, zd, hp.get('contrastive_temp', 0.5))""")
```
Place this `.replace` immediately after the `trainer = validate_fn + '''...'''` block in Task 3's generator code, and add `nt_xent_loss` to that same trainer cell:
```python
trainer = trainer + '''


def nt_xent_loss(z_clinic, z_derm, temperature=0.5):
    B = z_clinic.size(0)
    z = torch.cat([z_clinic, z_derm], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim.masked_fill_(torch.eye(2*B, dtype=torch.bool, device=z.device), float('-inf'))
    targets = (torch.arange(2*B, device=z.device) + B) % (2*B)
    return F.cross_entropy(sim, targets)
'''
```

Then append a markdown + a Full-Model-with-projection subclass cell:
```python
md('''## 7 — (Optional) Aux-contrastive Full Model — OFF by default

Joint loss `CE + λ·NT-Xent(z_clinic, z_derm)` keeps ImageNet features (unlike the
pretrain-replace variant that hurt). Adds two small projection heads to the Full Model only.
Set RUN_CONTRASTIVE=True to train this side-experiment; it does not touch the fair 7-model run.''')

code('''RUN_CONTRASTIVE = False   # side-experiment toggle

class FullModelProjected(DualBranchSECrossCombined):
    """Full Model + cross-modal projection heads for the aux-contrastive term."""
    def __init__(self, num_classes=5, dropout=0.32, pretrained=True, proj_dim=128):
        super().__init__(num_classes=num_classes, dropout=dropout, pretrained=pretrained)
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.proj_clinic = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(inplace=True), nn.Linear(2048, proj_dim))
        self.proj_derm   = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(inplace=True), nn.Linear(2048, proj_dim))

    def project(self, clinic_img, derm_img):
        fc = self.pool2d(self.resnet_clinic(clinic_img)).flatten(1)
        fd = self.pool2d(self.resnet_derm(derm_img)).flatten(1)
        return F.normalize(self.proj_clinic(fc), dim=1), F.normalize(self.proj_derm(fd), dim=1)

if RUN_CONTRASTIVE:
    hp_c = dict(BEST_HP); hp_c['contrastive_lambda'] = 0.2; hp_c['contrastive_temp'] = 0.5
    ckpt = f'thesis_full_contrastive_improved_{DATASET_NAME}_best.pth'
    tr_loader, va_loader, test_loader, _ = make_loaders(BATCH_SIZE, ACTIVE_DS)
    _m = FullModelProjected(num_classes=NC, dropout=BEST_HP['dropout'])
    run_training_improved(_m, ckpt, 'Full+Contrastive', tr_loader, va_loader,
                          ACTIVE_DS['loss_weights'], hp_c,
                          total_epochs=FINAL_EPOCHS, patience=PATIENCE, use_swa=True)
    m = evaluate_checkpoint_quiet(FullModelProjected, dict(num_classes=NC, dropout=BEST_HP['dropout'], pretrained=False),
                                  ckpt, test_loader)
    print('Full+Contrastive test:', {k: round(v,4) for k,v in m.items()})
else:
    print('RUN_CONTRASTIVE=False — skipped (flip to True for the side-experiment).')''')
```

- [ ] **Step 2: Regenerate + compile check**

Run: `python3 _build_singular_improved.py && python3 -c "import json; nb=json.load(open('thesis_test_singular_improved.ipynb')); [compile(''.join(c['source']),'x','exec') for c in nb['cells'] if c['cell_type']=='code']; print('compile OK', len(nb['cells']),'cells')"`
Expected: `compile OK 33 cells`

- [ ] **Step 3: Validate the contrastive path (projection + NT-Xent forward/backward)**

Append to `_val_singular.py` and re-run:
```python
# --- contrastive path ---
for c in cc:
    if 'class FullModelProjected' in c or 'def nt_xent_loss' in c:
        exec(c, ns)
fm = ns['FullModelProjected'](num_classes=5, dropout=0.32, pretrained=False).to(ns['device'])
ci = torch.randn(3,3,224,224).to(ns['device']); di = torch.randn(3,3,224,224).to(ns['device'])
zc, zd = fm.project(ci, di)
assert zc.shape == (3,128) and zd.shape == (3,128)
l = ns['nt_xent_loss'](zc, zd, 0.5); l.backward()
print('CONTRASTIVE path OK | nt_xent loss', round(l.item(),4))
```
Expected: `CONTRASTIVE path OK | nt_xent loss <number>`

- [ ] **Step 4: Commit**

```bash
git add _build_singular_improved.py thesis_test_singular_improved.ipynb _val_singular.py
git commit -m "feat(notebook): optional aux-contrastive Full Model (off by default)"
```

---

## Task 8: SMOKE end-to-end + cleanup + push

**Files:**
- Modify: `_build_singular_improved.py` (add SMOKE flag wiring to the search/constants cells)
- Delete: `_build_singular_improved.py`, `_val_singular.py` (after final notebook produced)

- [ ] **Step 1: Add a SMOKE flag to the constants cell**

In the Task 2 constants `code(...)` string, append a SMOKE block so a fast end-to-end run is possible. Add to the `extra` string:
```python
extra = extra + '''

# ── SMOKE: fast end-to-end validation (subset + few epochs). Set False for the real run. ──
SMOKE = False
if SMOKE:
    FINAL_EPOCHS = 3
    PATIENCE = 9'''
```
And in the Task 4 search cell, make trial/epoch counts honor SMOKE by prefixing:
```python
code('''if 'SMOKE' in globals() and SMOKE:
    SEARCH, N_TRIALS, SEARCH_EPOCHS = True, 2, 2
''' + <the existing search-cell body string>)
```
(Concatenate the SMOKE guard ahead of the existing search-cell source; the search-cell `SEARCH=True/N_TRIALS=15/SEARCH_EPOCHS=25` lines run after and are overwritten only when SMOKE.)

NOTE: simplest concrete implementation — define `SMOKE` in constants; in the search cell replace the first three assignments with SMOKE-aware values:
```python
search_body = search_body.replace('SEARCH        = True', 'SEARCH        = True')  # leave
search_body = "SMOKE = globals().get('SMOKE', False)\n" + search_body
search_body = search_body.replace('N_TRIALS      = 15', 'N_TRIALS      = 2 if SMOKE else 15')
search_body = search_body.replace('SEARCH_EPOCHS = 25', 'SEARCH_EPOCHS = 2 if SMOKE else 25')
```
Apply these replacements to the search cell string before `code(search_body)`.

- [ ] **Step 2: Regenerate**

Run: `python3 _build_singular_improved.py`
Expected: `wrote thesis_test_singular_improved.ipynb 33 cells`

- [ ] **Step 3: Full SMOKE run via nbconvert (Derm7pt subset)**

Set the toggle to Derm7pt and SMOKE=True for this check only, then execute the notebook end-to-end:
```bash
python3 - <<'PY'
import json
nb=json.load(open('thesis_test_singular_improved.ipynb'))
for c in nb['cells']:
    s=''.join(c['source'])
    if "DATASET_DIR = Path('dataset') / 'Milk10k'" in s:
        c['source']=[l.replace("'Milk10k'","'Derm7pt'") + ('\n' if not l.endswith('\n') else '') for l in c['source']]
    if 'SMOKE = False' in s:
        c['source']=[l.replace('SMOKE = False','SMOKE = True') for l in c['source']]
json.dump(nb,open('/tmp/_smoke.ipynb','w'),indent=1)
print('smoke notebook ready')
PY
jupyter nbconvert --to notebook --execute /tmp/_smoke.ipynb --output /tmp/_smoke_out.ipynb --ExecutePreprocessor.timeout=1800 2>&1 | tail -5
echo "EXIT: $?"
```
Expected: nbconvert completes with EXIT 0; `/tmp/_smoke_out.ipynb` exists. (Downloads ResNet50 weights on first run; allow time.)

- [ ] **Step 4: Confirm SMOKE produced the artifacts**

Run:
```bash
ls -la singular_bestparams_derm7pt.json comparison_improved_derm7pt.csv thesis_full_se_crossattn_improved_derm7pt_best.pth 2>&1
```
Expected: all three exist (created during the SMOKE run).

- [ ] **Step 5: Clean up generator + validation + smoke artifacts**

```bash
rm -f _build_singular_improved.py _val_singular.py /tmp/_smoke.ipynb /tmp/_smoke_out.ipynb
rm -f optuna_singular_derm7pt.db singular_bestparams_derm7pt.json comparison_improved_derm7pt.* delta_improved_derm7pt.csv thesis_*_improved_derm7pt_best.pth
```
(The SMOKE run's artifacts are throwaway — the real run regenerates them. The notebook itself is the deliverable.)

- [ ] **Step 6: Final commit + push**

```bash
git add thesis_test_singular_improved.ipynb docs/superpowers/plans/2026-05-30-singular-7model-improved.md
git rm --cached _build_singular_improved.py _val_singular.py 2>/dev/null || true
git commit -m "feat(notebook): improved 7-model singular notebook complete (SMOKE-validated)"
git push origin artifacts
```

---

## Self-Review

**Spec coverage:**
- 7 models copied verbatim → Task 2 ✓
- Shared trainer (AdamW, warmup→cosine, label smoothing, SWA, best-by-val-F1, early-stop paused in SWA) → Task 3 ✓
- AMP + Windows workers + drop_last → Task 1 (loaders) + Task 3 (AMP) ✓
- ONE shared Optuna search on Full Model, lock HPs to JSON, SQLite resumable, warm-start → Task 4 ✓
- All 7 trained with locked HPs, `_improved_{ds}` suffix → Task 5 ✓
- 7-metric eval + grand comparison + old-vs-improved delta → Task 6 ✓
- Optional aux-contrastive toggle (off by default, Full Model only, projection heads) → Task 7 ✓
- Dataset toggle, image-only, custom 70/15/15 → Task 1 (verbatim cells 4,5) ✓
- SMOKE validation → Task 8 ✓

**Placeholder scan:** No TBD/TODO; all code blocks concrete. Task 8 Step 1 gives both a prose and a concrete `.replace` implementation — use the concrete one.

**Type consistency:** `run_training_improved(model, ckpt_path, tag, train_loader, val_loader, loss_weights, hp, ...)` signature identical across Tasks 3, 4, 5, 7. `hp` keys (`lr, weight_decay, dropout, label_smoothing`, optional `contrastive_lambda, contrastive_temp`) consistent. `BEST_HP` produced in Task 4, consumed in Tasks 5–7. `evaluate_checkpoint_quiet(cls, kwargs, ckpt, loader)` matches baseline cell 8 signature. Model constructors all accept `num_classes, dropout, pretrained` (+ `fusion` for elementwise). Checkpoint dict key `model_state_dict` matches what `evaluate_checkpoint_quiet` loads.

**Risk note:** all model constructors default `pretrained=True` (weight download). Validation harnesses pass `pretrained=False`; the real notebook keeps `pretrained=True` (intended).
