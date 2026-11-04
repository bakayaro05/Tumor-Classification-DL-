# Public GitHub Release Audit and Reproducibility Notes

**Audit date:** 4 November 2026.  
**Repository:** https://github.com/bakayaro05/Tumor-Classification-DL-  
**Observed `main` tree:** `b881e99829afac69de34ccc21350010c697ab7c1`.  
**Authorization status:** The GitHub connector in this session is authenticated as **a different account without write permission** on `bakayaro05/Tumor-Classification-DL-`. None of the new README or documentation files have been pushed by this assistant; the files are prepared for placement into the repository.

## Observed facts from the updated repository

- **12,020 tracked files** (about **388 MB** combined blob sizes); the tree was not truncated.
- A **final seven-tab** 15-frozen-pipeline `app.py` is now committed.
- Phase 1 experiment results, five Phase 2A experiment bundles, **35 Phase 2B run directories**, external comparison results for **BDNeuro full / balanced / Web-40**, frozen selection and hash records, XAI CSVs, earlier deliverables and source-history packages are present.
- **0 `.pth` neural checkpoints** are tracked; `.gitignore` intentionally ignores these, while **240 `.joblib`** artifacts, 405 JSON, 100 CSV, 342 NPY and **10,713 JPG** items are tracked. Results may be independently inspected, but a fresh checkout **cannot perform MRI inference** without the model-weight checkpoints.
- `data/brain_tumor_mri/` (development Nickparvar image folder) is absent in tracked Git paths; fixed `data/splits/train_indices.npy` and `val_indices.npy` are present.
- `README_GUI.md` is an **older four-tab Phase 1 GUI guide** and `verify_gui_setup.py` checks only Phase 1 artifacts; neither fully describes the new seven-tab dashboard.
- Earlier `final_project_docs/FINAL_QC_BLOCKER.md` is a **historical immediate post-duplicate-audit status**, superseded by later completed Phase 2 and external validation work; the duplicate finding remains valid.
- `final_deliverables/FINAL_PROJECT_REPORT.pdf` is an **earlier report**. The **revised 60-page report** used to create the new GitHub edition was supplied separately and is **not** currently an identically named tracked GitHub file.

## Files prepared to publish

Place the files exactly as follows, relative to the repository root:

```text
README.md
GITHUB_DESCRIPTION.txt             # helper for repository About description; optional to commit

docs/REPOSITORY_DOCUMENTATION.md

docs/REPOSITORY_DOCUMENTATION.pdf

docs/RESEARCH_REPORT_GITHUB_EDITION.pdf

docs/DATASETS_AND_REFERENCES.md

docs/PUBLICATION_NOTES.md
```

The GitHub edition of the full report uses the scientific body (original chapters 1–15, references and appendices, including all original chart and Streamlit GUI screenshot pages) from the revised report, **not** the older 68-page artifact. The university certificate, acknowledgement and named submission pages are excluded and a neutral university college mini-project cover and publication note are inserted. No project-member or project-guide names are deliberately included in the new public-facing documents.

## Recommended publication workflow

From a local clone of the user-owned repository, with current `main` first updated:

```bash
cd ~/brain_tumor_project  # or the path to your current repo clone

git remote -v
git status --short
git branch --show-current

git pull --ff-only origin main
# Copy the prepared README.md and docs/* files to their listed locations.

git add README.md docs/REPOSITORY_DOCUMENTATION.md \
  docs/REPOSITORY_DOCUMENTATION.pdf docs/RESEARCH_REPORT_GITHUB_EDITION.pdf \
  docs/DATASETS_AND_REFERENCES.md docs/PUBLICATION_NOTES.md

git diff --cached --stat
git diff --cached --check
# inspect and verify the staged set; if it is correct:
git commit -m "docs: publish complete research README and university mini-project report"
git push origin main
```

**No force-push is recommended.** If other users modified `main`, reconcile changes first. If the commit includes data with unclear permissions, resolve those permissions before publication. Do not automatically `git add -A` over a research directory containing private MRI files, credentials, checkpoints, unreviewed web panels or generated data.

## Matters to address separately (not hidden by documentation)

1. **Neural weights:** publish a documented, verified checkpoint-release mechanism **only if sharing is authorized and desired**, or state explicitly that live inference requires the original trained artifacts. Git LFS or a GitHub Release can distribute large artifacts; **none has been verified as available** and no download URL is invented here. Each model's checkpoint must match its frozen scaler/PCA/SVM state.
2. **Dataset-image redistribution:** the public tree currently contains the original BDNeuro V8 images, exact-clean and balanced copies, and Web-40 panels. The Mendeley Version 8 landing page states CC BY 4.0; however, the copy's internal README contains license/attribution placeholders. Check Mendeley/upstream provenance and individual publication-image rights. The Nickparvar dataset is externally linked rather than mirrored under the expected development directory. Do not equate an accessible GitHub path with blanket permission to redistribute medical-image pixels.
3. **Historical claims:** the original local QC said **174/174 automated checks** and **75/75 frozen artifact hashes** passed. Because tracked Git is missing `.pth` weights, this session could not independently redo the complete hash audit; cite the original QC as a historical record, not a newly executed validation.
4. **Readiness:** the current seven-tab app is present and coded to show all 15; functional inference of all 15 from the public checkout has **not** been verified without the corresponding `.pth` files. The old `verify_gui_setup.py` is Phase 1-only.
5. **Scholarly attribution:** cite [Nickparvar/Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), [BDNeuro Mendeley Version 8](https://data.mendeley.com/datasets/zwr4ntf94j/8), [base paper](https://doi.org/10.1007/s13721-025-00718-8) and each applicable Web-40 source page; see `DATASETS_AND_REFERENCES.md`.

**Mini-project scope:** A completed research and educational system, not a medical device, not patient-diagnostic software, and not clinically validated.
