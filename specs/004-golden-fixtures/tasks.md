# Tasks: Golden Fixtures (laya:004)

- [x] T001 Spec with clarifications (spec.md)
- [x] T002 `experiments/golden/common.py` (determinism, fixture agent, writers, meta)
- [x] T003 `weights.py` + `golden weights-inventory`
- [x] T004 `forward.py` four cases + staged forward + fast-path equality check + `golden export`
- [x] T005 `kernels.py` + `golden kernels`
- [x] T006 CLI command `golden` in `experiments/cli.py`; `artifacts/` in `.gitignore`
- [x] T007 `tests/experiments/test_golden.py`: presence, shapes, meta, padding, fast-path failure, kernel failure, byte identity, size
- [x] T008 Two full runs of every action compared by sha256: identical
