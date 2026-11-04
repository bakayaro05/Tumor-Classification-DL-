# Model weights, checkpoint archive, integrity checks and restoration

> **University college mini-project · brain tumour MRI classification · research and educational use only.** The saved pipelines are not clinically validated and must not be used for diagnosis, patient triage, or treatment decisions.
>
> **Status verified 5 November 2026:** The GitHub **`main` tree intentionally does not track project `.pth` weights**, but two separate **public GitHub Releases now provide the original model artifacts**. Publication of the files does not by itself demonstrate that a clean clone has passed a new end-to-end replay test.

**Repository:** https://github.com/bakayaro05/Tumor-Classification-DL-

## 1. Choose the correct release

| Release | Contents | Who needs it? |
|---|---|---|
| [**`models-v1.0` — frozen inference pipelines**](https://github.com/bakayaro05/Tumor-Classification-DL-/releases/tag/models-v1.0) | **15 selected CNN–LSTM–StandardScaler–PCA–SVM pipelines**, five backbones × Phase 1, Phase 2A, Phase 2B. Three TAR files: `models-v1.0-phase1.tar`, `models-v1.0-phase2a.tar`, `models-v1.0-phase2b.tar`. Each selected pipeline includes its neural checkpoint, fitted scaler, fitted PCA, SVM, and internal result artifact. | Readers who want to load the frozen models, use the seven-tab Streamlit dashboard, or replay selected-pipeline inference. **Start here.** |
| [**`research-checkpoints-v1.0` — additional historical checkpoints**](https://github.com/bakayaro05/Tumor-Classification-DL-/releases/tag/research-checkpoints-v1.0) | **71 additional neural checkpoints** in `research-checkpoints-v1.0-part01.tar` through `part12.tar`, plus `REMAINING_CHECKPOINTS_MANIFEST.json` and `SHA256SUMS`. These comprise five non-selected Phase 2A epoch-30 states, 65 non-selected Phase 2B states, and one legacy ResNet152 checkpoint. | Researchers recovering the full training history, comparing alternate states, or auditing experiment provenance. **Not required for ordinary dashboard inference.** |

Together the releases preserve **86 neural checkpoint files**: 15 selected + 71 additional. The complete frozen *inference* bundles are the **15 in `models-v1.0`**, not the 71 historical checkpoint files.

The project-level representative is **ResNet152, Phase 2A, `best_val_within_30`, epoch 10**. This was a *retrospective research representative*, partly informed by external results, and should not be described as a prospectively validated clinical winner. **Formal rated Grad-CAM/occlusion studies used Phase 1 ResNet152 instead.**

## 2. Prepare a fresh clone and download the 15-pipeline release

Use **Python 3.12** and a compatible PyTorch/torchvision pair as explained in [REPRODUCE.md](REPRODUCE.md). Run model restoration in a **fresh clone**, not over the original training directory. The following example uses a checkout path compatible with the existing scripts that assume `~/brain_tumor_project`.

```bash
# Requires git and the GitHub CLI (`gh`) for release download.
# Do not overwrite an existing ~/brain_tumor_project directory.
git clone https://github.com/bakayaro05/Tumor-Classification-DL-.git "$HOME/brain_tumor_project"
cd "$HOME/brain_tumor_project"

git rev-parse HEAD
mkdir -p _release_downloads/models-v1.0

gh release download models-v1.0 \
  --repo bakayaro05/Tumor-Classification-DL- \
  --pattern 'models-v1.0-*.tar' \
  --dir _release_downloads/models-v1.0

ls -lh _release_downloads/models-v1.0/
```

Alternatively, download the **three exact TAR assets** manually from the public release page into `_release_downloads/models-v1.0/`. These are **three separate archives**, **not one split TAR stream**: extract each independently. For only one experimental phase, restore that phase's TAR, then verify the corresponding five pipelines; a full 15-pipeline replay needs all three.

### Archive integrity before extraction

As of 5 November 2026, **`models-v1.0` has no attached `SHA256SUMS` release asset yet**. However, the public GitHub release asset API exposes the following **server-recorded `sha256:` digests** for its three TAR files; these are not guesses. A `SHA256SUMS` file containing these three lines has been prepared for upload to the same release. You can also create it directly and verify downloaded files before extraction:

```bash
cd "$HOME/brain_tumor_project/_release_downloads/models-v1.0"
cat > SHA256SUMS <<'CHECKSUMS'
628eebcd5ffdd80a26a28c86e3b4f10ef5e89ade5caf406cb480f0fe375cd05e  models-v1.0-phase1.tar
3d8392c8bb93fde511b883e5a1f85b6a7383b27643d716aab4014891fc99bd88  models-v1.0-phase2a.tar
5c7aa329edcf851a0139ccb24969a1d224111d17987024d8aa114ad9b9c7fae1  models-v1.0-phase2b.tar
CHECKSUMS
sha256sum --check SHA256SUMS
cd "$HOME/brain_tumor_project"
```

**Stop if any TAR-level digest fails.** Server-recorded digests are evidence for the expected archive bytes, but this documentation review did **not** download and rehash the TAR files itself. After extraction, also verify **75 constituent file hashes** (15 selected pipelines × five files) against the tracked [frozen manifest](../phase2/external_validation/frozen_15_manifest.json). Both levels of integrity checking matter.

Before extracting, inspect the archive members and reject archives with absolute paths, `..` traversal, or link/special-file entries. The following command both checks the members and extracts into the repository root using Python's `data` tar extraction filter:

```bash
python - <<'PY'
from pathlib import Path, PurePosixPath
import tarfile

root = Path.cwd().resolve()
archives = sorted((root / '_release_downloads/models-v1.0').glob('models-v1.0-*.tar'))
assert len(archives) == 3, f'Expected 3 TAR archives; found {len(archives)}'

# Validate *all* members before writing any file.
for archive in archives:
    with tarfile.open(archive, 'r:') as tar:
        members = tar.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise RuntimeError(f'Unsafe archive member {member.name!r} in {archive.name}')
        print(f'Checked {archive.name}: {len(members)} members')

for archive in archives:
    with tarfile.open(archive, 'r:') as tar:
        tar.extractall(root, filter='data')  # Python 3.12+
    print('Extracted:', archive.name)
PY
```

**Important:** Extraction writes to the archive's original project-relative paths. Perform it only in the disposable clone and inspect the archive member list (`tar -tf file.tar`) first if you have local modifications you wish to preserve. Downloaded TAR archives can be kept outside the Git worktree after validation; never commit checkpoint files to normal Git history.

## 3. Verify all 15 frozen pipelines against the research manifest

**This is the decisive post-extraction check**: verify every selected model's checkpoint, StandardScaler, PCA, SVM, and saved internal result against the *original frozen* SHA-256 manifest. It verifies artifact **bytes**, not scientific independence or clinical validity.

```bash
cd "$HOME/brain_tumor_project"
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
manifest_path = root / 'phase2/external_validation/frozen_15_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
candidates = manifest['candidates']
assert len(candidates) == 15, f'Unexpected frozen candidate count: {len(candidates)}'

checked = 0
errors = []
for candidate in candidates:
    label = f"{candidate['backbone']} / {candidate['stage']} / {candidate['checkpoint_state']}"
    for kind, relative_path in candidate['paths'].items():
        expected = candidate['sha256'][kind]
        path = root / relative_path
        if not path.is_file():
            errors.append(f'MISSING: {label}: {relative_path}')
            continue
        hasher = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                hasher.update(chunk)
        actual = hasher.hexdigest()
        if actual != expected:
            errors.append(f'MISMATCH: {label}: {relative_path}')
        else:
            checked += 1

print(f'Frozen artifact checks passed: {checked}/75')
if errors:
    for message in errors:
        print(message)
    raise SystemExit(1)
assert checked == 75, f'Expected 75 matching hashes, got {checked}'
print('All 15 selected five-part bundles match the original frozen manifest.')
PY
```

If verification fails, **do not bypass the hash check**, rename a different checkpoint to satisfy a path, combine files from different phases, or refit the scaler/PCA/SVM on external images. Re-download and investigate first. Exact checksum equality means the file bytes match the historical record; it does not establish source-image or patient-level independence.

## 4. Confirm that the seven-tab dashboard can load a selected pipeline

First install an appropriate **PyTorch/torchvision** build following https://pytorch.org/get-started/locally/, and the project dependencies from [REPRODUCE.md](REPRODUCE.md). `requirements-gui.txt` does **not** install PyTorch. The recorded machine used Python **3.12.14**, PyTorch **2.11.0+cu128**, torchvision **0.26.0+cu128**, and an RTX 5070 Laptop GPU; a different device may require different wheels and may not reproduce bit-for-bit timing or outputs.

This lightweight check reports the existence of *all four runtime model components* for each of the 15 pipeline IDs. It is not a weight-deserialization test:

```bash
python - <<'PY'
from pathlib import Path
from gui.model_registry import all_pipeline_artifacts

for item in all_pipeline_artifacts(Path.cwd()):
    missing = [str(path.relative_to(Path.cwd())) for path in item.missing_prediction_files()]
    status = 'READY' if not missing else f'MISSING {len(missing)}'
    print(f'{item.pipeline_id:30s} {status}')
    for path in missing:
        print('   ', path)
PY
```

For a **non-diagnostic pipeline smoke test** (requires compatible dependencies and enough RAM/VRAM), run one synthetic RGB input through the final **ResNet152 Phase 2A** pipeline. This confirms model loading and inference plumbing only; its predicted class has **no medical or accuracy interpretation**:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
from gui.inference_engine import ModelManager

manager = ModelManager(Path.cwd(), device_preference='auto')
synthetic_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
result = manager.predict(synthetic_image, 'resnet152_phase2a')
print('Pipeline:', result.pipeline_id if hasattr(result, 'pipeline_id') else 'resnet152_phase2a')
print('Smoke-test inference completed; synthetic classification is not meaningful.')
PY
```

Then start the actual application:

```bash
streamlit run app.py
```

It presents seven tabs with 15 selectable frozen pipelines. The GUI consumes **single 2-D raster MRI images**, not DICOM volumes or patient studies. The final output is the **linear SVM**, not the neural classification head. SVM score bars are **relative scores, not calibrated probabilities**. The historical `verify_gui_setup.py` and `README_GUI.md` target the older Phase 1 interface and do not validate the full 15-pipeline application.

A complete **clean-clone download → restore → 75-hash verification → real inference replay** is **still to be independently performed and recorded**. Do not claim it has passed solely because the release assets exist. A synthetic smoke test additionally cannot verify dataset performance.

## 5. Restore the 71 additional historical checkpoints (optional)

Only do this if you need the **full archive of historical states**. This is separate from and not a substitute for the 15 selected inference pipelines.

```bash
cd "$HOME/brain_tumor_project"
mkdir -p _release_downloads/research-checkpoints-v1.0

gh release download research-checkpoints-v1.0 \
  --repo bakayaro05/Tumor-Classification-DL- \
  --pattern 'research-checkpoints-v1.0-part*.tar' \
  --pattern 'REMAINING_CHECKPOINTS_MANIFEST.json' \
  --pattern 'SHA256SUMS' \
  --dir _release_downloads/research-checkpoints-v1.0

cd _release_downloads/research-checkpoints-v1.0
sha256sum --check SHA256SUMS
```

**Stop if the archive-level hash check fails.** The release's own manifest lists original relative checkpoint paths, archive membership and per-checkpoint SHA-256 values. Examine the manifest and verify extracted files against it. Do **not** assume its JSON structure matches the different `frozen_15_manifest.json` schema. All **12 TAR files** are independent archives preserving project-relative paths, not compressed fragments to concatenate. Apply the same pre-extraction path, link and special-file safety checks described in §2, then extract in a **fresh working copy**.

These 71 checkpoint states can contain alternate experiment results that were not selected for the final frozen 15. **Do not use them to reselect models after seeing external test outcomes**. Reproducing the full external evaluation still requires the originally used datasets, exact image order and complete matched SVM pipelines.

## 6. Research interpretation and provenance

- The 7,023-image development dataset's historical original partitions had **562 material cross-split duplicate pairs** involving 250 test images. Replaying those scores does not fix contamination or establish patient-level independence.
- The project representative's BDNeuro V8 exact-overlap-clean evaluation was **88.31% on 2,644 images**, while the exploratory **Web-40** score was **37.5%**. The representative was chosen retrospectively partly using BDNeuro evidence. These are descriptive research outcomes, not clinical guarantees.
- The 1,648-image BDNeuro balanced set is **nested** within the same exact-overlap-clean cohort, not an additional independent cohort.
- The 20-case rated Grad-CAM and 11-case rated SVM occlusion evaluations used **Phase 1 ResNet152**. Grad-CAM explains the neural head; occlusion probes the final SVM. Neither is tumour segmentation.
- Saved checkpoints, pickle/joblib files and model archives should be loaded **only from trusted, verified sources**. Hash-checking confirms identity, but does not make arbitrary downloaded serialized objects safe.
- Dataset redistribution, particularly sourced Web-40 panels and potentially copied clinical MRI images, requires rights review independently of these model releases.

See [REPRODUCE.md](REPRODUCE.md), [repository documentation](REPOSITORY_DOCUMENTATION.md), [dataset sources](DATASETS_AND_REFERENCES.md), and [the GitHub research-report edition](RESEARCH_REPORT_GITHUB_EDITION.pdf) for the full methodology and evidence chain.
