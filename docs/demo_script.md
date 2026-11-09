# Demo script: GreenOps, tomorrow's energy weather

About 7 minutes, no slides. Two people: **A** drives the page, **B** drives the terminal and answers technical questions. One rule: never say "prediction" or "accurate". Say "outlook" and give the margin.

## Before you start (10 minutes ahead)

| Check | How | Done |
|---|---|---|
| Live page opens | https://greenops-api-latest.onrender.com: open it 3 minutes early (free host, cold start about a minute), header says "online" | |
| Local copy ready (backup 1) | Terminal 1: `$env:PREDICTOR="bundle"; $env:SERVE_DIR="serve"; $env:SHADOW_DIR="serve_shadow"; uvicorn solarwind.serving.app:app --port 8001`, wait about 45 s for startup, then open `http://127.0.0.1:8001/#site=pavagada&date=2026-09-20` | |
| Recording ready (backup 2) | `demo_fallback.mp4` on the desktop, plays | |
| Terminal 2 in the repo | `cd C:\dev\mlops; .venv\Scripts\activate` | |
| TensorBoard | `tensorboard --logdir runs/tb`, open <http://localhost:6006> (TEXT tab) | |
| GitHub Actions tab | the latest green run on `main` | |
| Browser | zoom 100%, no other tabs visible, notifications off, battery plugged in | |

Open these views as bookmarks so nothing is typed live:
`#site=pavagada&date=2026-09-20` , `#site=koppal&date=2026-03-04` , `#site=pavagada&date=2019-10-30`

## The run

| Time | Who | Show | Say |
|---|---|---|---|
| 0:00 | A | Page, Pavagada, 20 Sep | "A household in Karnataka wants to know what tomorrow's sun or wind means for a lamp, a water heater, a small turbine. We treat today as 20 September and ask for tomorrow." Read the verdict aloud. |
| 0:40 | A | Callouts, then the sky | "The scene is the forecast: sky clarity follows sunlight, the lamp glows by hours lit, the heater shows its temperature against the 45 °C line. The devices are simulated from spec sheets and the page says so." |
| 1:10 | A | **Reveal what happened** | "Because this is a replay we can check. The outlook missed by about one unit here. We show the miss." |
| 1:30 | A | What-if slider to 60%, then Reset | "Change today's sunlight and the outlook moves, the heater drops below 45 °C and the backup switches on." |
| 2:00 | A | Koppal, 4 Mar | "A wind site. Turbines nearly stop, the calm-day alert fires. Same model code, different task." (Check the numbers in rehearsal.) |
| 2:30 | A | **Play 14 days** (pause after 5 s) | "Each step is a fresh call to the live API." |
| 2:50 | A | Pavagada, 30 Oct 2019 | "The first day of data. Not enough history, so the service does not crash or invent a forecast: it copies today and says 'Fallback' in amber." |
| 3:15 | A | **How this is checked** | "Model versions come from the running service. The margins over 'tomorrow = today' are small: 6.3% for sunlight, 2.5% for wind. We show that on purpose." Close the dialog. |
| 3:45 | B | Terminal: `python -m solarwind.devices.gates check --serve serve` | "Seven release gates. Every model must beat the simple baseline, stay in physical limits, treat every site fairly, survive a missing day and 5% noise, and tell the truth about its own metrics." Point at the PASS lines. |
| 4:30 | B | `python -m solarwind.devices.gates make-bad --serve serve --out serve_bad --mode scaled` then `python -m solarwind.devices.gates check --serve serve_bad --champion serve` | "Now a deliberately broken model: a kWh to Wh unit bug. It fails G1, G2 and G3 and the command exits with an error. In CI that stops the build." |
| 5:15 | B | `python -m solarwind.devices.registry status` | "The registry keeps champion, challenger and previous. The champion was not touched; a failed candidate cannot be promoted, and `rollback` restores the previous one." |
| 5:40 | B | TensorBoard, TEXT tab leaderboard, then the LSTM runs | "We were asked to consider an LSTM in TensorFlow. We fixed the rule first: promote only if the 5-seed mean validation error is at least 2% lower. It was, by 3.1% and 2.9%. We still serve it only as a shadow, because it needs TensorFlow in the image and has no device-level comparison yet. The page can show it beside the main model." |
| 6:15 | B | GitHub Actions run, then the live Render URL | "Every push runs lint and tests, the same gate command, a Docker smoke test; the image is published only from `main`. The container carries the pinned bundle, so deploy never retrains and needs no secrets." |
| 6:45 | A | Page | "An outlook, not a promise: eight sites, one day ahead, honest about its margin." |

## If something breaks

| Problem | Do |
|---|---|
| Live page slow or banner shows | Say "cold start", wait 20 s, or switch to the local copy |
| Live page down | Local copy (`localhost:8001`), same views |
| Local server will not start | Play `demo_fallback.mp4` and narrate over it |
| LSTM switch says "not loaded" | "The shadow model is wired in the API and tested; it is not mounted in this build." Show the TensorBoard runs instead |
| Terminal command fails | Say what it should print, show `results/gates/report.json` |

## Questions the evaluator is likely to ask

| Question | Answer |
|---|---|
| Why is the LSTM not the champion? | It passes the accuracy rule, but TensorFlow is a large dependency for the serving image and slows startup, it has no device-level comparison yet, and a promoted model must also pass the gates. It runs as a shadow; the decision and the revisit condition are in D4. |
| The margin is tiny. Why ship it? | One day ahead is hard to beat persistence. We measure the margin, publish it, and gate on it. The value is the controlled process; the gain over seasonal and naive baselines is large. |
| How do you know there is no leakage? | Date split, scalers and cut-offs fitted on training days only, and gate G7 changes all data after the as-of day and requires the forecast not to change. A test builds a leaky path on purpose and checks the gate catches it. |
| What about drift? | Not automated yet; it is an open risk (R8). Replay shows forecast against actual for any window, and the next step is a monitor with a retraining trigger. |
| Why a replay and not live weather? | NASA POWER publishes days late, so a live "tomorrow" is not clean, and the demo must work offline and be reproducible. |
| What if the weather is missing? | A flagged persistence fallback, never an error (gate G6). |
| Is the model file safe to load? | skops loads only trusted types, every file is SHA-256 checked, the library version must match, and the service refuses to start otherwise. |
| How do you roll back? | `registry rollback` restores the previous champion; the image is rebuilt from the pinned bundle in `dvc.lock`. |
| Why are the devices simulated? | No measured field data. Every spec is tagged published, worked out or our choice. The page and the model card say so. |
| Who is it for? | Awareness: a household or farm deciding what to expect tomorrow. Not grid dispatch or safety. |

## Rehearsals

Run it twice end to end with a timer. Write down what slipped.

| | Date and time | Total | Slowest step | Fix |
|---|---|---|---|---|
| Rehearsal 1 | | | | |
| Rehearsal 2 | | | | |

Record the fallback once: Windows key + Alt + R (Game Bar), run the whole page section (0:00 to 3:45) and the gate commands, save as `demo_fallback.mp4`.
