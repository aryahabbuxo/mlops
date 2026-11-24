## What and why

<!-- One or two sentences. For a model change, say why the new model should replace the old one. -->

## Checklist

- [ ] Tests pass locally (`python -m pytest -q`) and ruff is clean (`ruff check . && ruff format --check .`)
- [ ] Promotion gates pass (`python -m solarwind.devices.gates check --serve serve`) if the bundle, `dvc.lock` or model code changed
- [ ] No data, model files, `mlflow.db`, logs, tokens or secrets committed (data goes through DVC)
- [ ] Model change has a reason above and a registry entry (`python -m solarwind.devices.registry register`)
- [ ] Existing API responses unchanged; new fields or endpoints only
- [ ] Docs updated if behaviour, endpoints or setup changed
