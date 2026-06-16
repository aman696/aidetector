## What this changes

Brief description of the change and why.

## Type

- [ ] Bug fix
- [ ] New analyzer / feature (has tests, added before wiring into the classifier)
- [ ] Docs
- [ ] Refactor (value-identical — proven by a test)

## Impact on the model

- [ ] No effect on feature values
- [ ] Changes feature values -> bumped `_CLASSICAL_FORMULA_EPOCH` and noted that
      re-extraction + retraining is required
- [ ] Changes training / evaluation

## Checklist

- [ ] `python -m pytest -q` passes
- [ ] Docstrings state purpose / args / returns; score-direction claims are
      backed by a measured probe and a test
- [ ] Hyperparameter / threshold choices explained in this description
- [ ] Docs updated in the same PR if behaviour / commands / layout / accuracy
      changed (README headline, RESEARCH detail, WORKFLOW pipeline)
- [ ] No emojis in documentation
