# Governance: GreenOps day-ahead forecasts

Team 18, AI252TA (MLOps). Phase 3. Scope: the three champion models, the serving bundle, the API, the
registry, the pipeline and the monitoring around them.

Governance here means: who may change what users see, under which rules, how those rules are enforced
by tools rather than by trust, and how anyone can check afterwards what happened and why. Every policy
below has an enforcement and evidence you can open.

## 1. Roles and accountability

| Role | Responsible for | Cannot do alone |
|---|---|---|
| Model owner | Data, training, gates, registry, drift and performance monitoring, model card | Merge to `main` without the required checks; promote without a recorded reason |
| Platform owner | API, Docker image, CI/CD, deployment, metrics, monitoring stack, access settings | Merge to `main` without the required checks |
| Reviewer | Approves pull requests into `main` | Approve their own pull request (GitHub forbids it) |

**Why:** two people are involved in anything users will see, so a single mistake or a single bad
decision cannot reach the live site.

## 2. Change control on `main`

| Policy | Enforced by | Evidence |
|---|---|---|
| Every change reaches `main` through a pull request | GitHub branch protection on `main` | Settings > Branches screenshot |
| A pull request merges only when `lint-and-test`, `model-gates` and `docker-smoke` pass | Required status checks | Pull request page, Actions run |
| Approval by a second collaborator, rules apply to admins too | Branch protection | Settings > Branches |
| No force pushes, `main` cannot be deleted | Branch protection | Settings > Branches |
| Model and pipeline files have named owners | `.github/CODEOWNERS` | The file |
| Every pull request answers a checklist (tests, gates, no data or secrets, reason for a model change) | `.github/pull_request_template.md` | Any pull request |

**Why:** `main` deploys automatically to the public site. Without these rules, one push could ship an
untested model or a broken image.

## 3. Model promotion policy

A model version may serve users only if all of these hold:

1. It passes the seven quality gates (G1 to G7, decision D11), measured from the model itself.
2. On the same recent months, it is not worse than the current champion (same-window comparison).
3. Its promotion records a reason and an approver as registry tags, with a timestamp.
4. Promotion is a human command. The retraining loop can register a candidate; it never promotes.

The previous champion is kept with the alias `previous`, so `registry rollback` restores it in one
command.

**Why:** accuracy alone is not enough (decision D4). A model must be safe, fair across sites, honest
about its scores, and better on the data users see now, and someone must take responsibility for it.

## 4. Versioning policy

| Rule | How |
|---|---|
| A version never changes after registration; only aliases move | MLflow aliases `champion`, `challenger`, `previous` (decision D9) |
| Every version records what made it | Tags: data window, git commit, file SHA-256, metrics, gate result, reason, approver |
| The data window is versioned with the code | `params.yaml` (train and validation dates), tracked by DVC |
| What is deployed is pinned | `dvc.lock` holds the bundle hash; CI and deploy never retrain (D8) |
| The running service says what it runs | `/version`: API version, git commit, model version per task |

**Why we version:** to reproduce any forecast from its exact model and data, to compare models fairly,
to roll back in one command when something goes wrong, and to answer "who changed what, when and why".

## 5. Data policy

| Policy | Enforced by |
|---|---|
| Only public NASA POWER weather; no personal data | Data card; inputs are weather only |
| Raw data and model files live in DVC (DagsHub), never in git | `.gitignore`, DVC stages |
| People who only run the project get read access to the data | DagsHub collaborator roles |
| Every dataset and bundle has a checksum | `dvc.lock`, manifest SHA-256 |

**Why:** no privacy risk, reproducible inputs, and nobody can overwrite the bundle that `main` depends
on by accident.

## 6. Security

| Policy | Enforced by |
|---|---|
| Model files load only trusted types and only if their SHA-256 matches | skops, `bundle.py` (risk R15) |
| Secrets live only in GitHub secrets and git-ignored local config | `.gitignore`, CI secrets |
| The deployed image needs no secrets and runs as a normal user | `Dockerfile` |
| A leaked token is revoked and replaced, and recorded | Risk register R17 |

## 7. Monitoring and incident response

| Signal | Threshold | Action |
|---|---|---|
| Data drift (PSI per input) | WARN 0.10, ALERT 0.25 (D16) | Investigate; on ALERT, check performance |
| Performance drift (monthly error vs persistence) | Model worse than persistence | Retrain candidate on newer data |
| Calm-day alert quality (quarterly PR-AUC) | Below yesterday's flag | Retrain candidate |
| Service health | `/health`, `/ready`, error rate, latency | Roll back the image (revert on `main`) |

Response path: alert, investigate, retrain a candidate, gates, same-window comparison, approval,
promote or refuse; if a live model misbehaves, `registry rollback`, rebuild the bundle, redeploy.

**Why:** a model can fail silently after deployment while every test still passes. Monitoring closes
that gap (risk R8).

## 8. Audit trail

| Question | Answered by |
|---|---|
| Which code is live? | `/version` git commit, GitHub Actions run, image tag |
| Which model is live and why? | Registry tags (reason, approver, gates), `results/registry/history.json` |
| What did the model say for a request? | Prediction log and request id |
| Who changed the pipeline? | Git history, pull requests |
| What did the gates and monitors find? | `results/gates/report.json`, `results/monitoring/` |

## 9. Responsible use

GreenOps is an awareness tool: one day ahead, eight sites, simulated devices. The page shows the margin
over "tomorrow = today", the data mode and the model version. It is not for grid, safety or money
decisions (D15, R18).

## 10. Mapping to the NIST AI Risk Management Framework

| Function | What it asks | Where we do it |
|---|---|---|
| Govern | Roles, policies, accountability | Sections 1 to 4, CODEOWNERS, branch protection |
| Map | Context, intended use, risks | Model card, data card, risk register, responsible use (section 9) |
| Measure | Test and track the risks | 7 gates, test suite, drift and performance monitoring |
| Manage | Act on what is measured | Promotion policy, rollback, incident response (section 7) |
