# Billing Agent + Billing Sandbox

Your part of the multi-agent customer-support market described in the master
project context. This runs standalone right now — no API keys, no
dependencies, nothing to install beyond Python 3.

## Files

| File              | What it does                                                          |
|-------------------|------------------------------------------------------------------------|
| `sandbox.py`      | Fake billing system. Holds hidden ground-truth state (payment/refund/invoice status) for 3 example orders, and exposes `check_*` (read) and action tools (write) — the agent never touches state directly. |
| `capability.py`   | The capability card Billing broadcasts to the other agents, plus keyword weights used for a quick rule-based confidence estimate. |
| `bidding.py`      | `estimate_raw_confidence()` + the exact bid formula from the master doc (`load_penalty`, `urgency_weight`, `final_bid`). |
| `negotiation.py`  | PROPOSE / COUNTER / ACCEPT / REJECT message builders, bounded to 3 rounds, plus a `MockTechnicalAgent` stand-in for your teammate's not-yet-built agent. |
| `llm_client.py`   | Empty hook — wire this up to Ollama/vLLM/an API later. Nothing depends on it yet. |
| `agent.py`        | `BillingAgent` class — ties everything together: `submit_bid → investigate → diagnose → execute → negotiate (if needed) → final_statement`. |
| `demo.py`         | Runs 3 test tickets end-to-end and prints the whole trace, including the exact worked example ticket from the master doc (section 20). |

## Run it

```bash
cd billing_agent
python3 demo.py
```

You should see, for each ticket: the bid math, the diagnosis, the sandbox
state before/after, any negotiation transcript, and a final statement — this
final statement + state changes are what your vetting layer (cosine
similarity + LLM judge) will later score.

## The 3 test scenarios (mirrors master doc section 13)

Same *kind* of complaint ("where's my refund / order"), three different root
causes, so the agent has to actually investigate instead of pattern-matching
the ticket text:

- **O-1001** — refund genuinely stuck at the gateway → agent releases it.
- **O-1002** — refund auto-rejected (bad bank details) → agent can't
  self-fix, flags for manual review.
- **O-1003** — refund already processed correctly → this isn't a billing
  problem at all, so the agent proposes handing the technical half
  (dashboard visibility) to the Technical Agent via PROPOSE/ACCEPT.

## What to build next (in this order)

1. **More order scenarios** in `sandbox.py` — right now there are 3. Add
   more edge cases (missing invoice, duplicate charge, subscription
   double-billed, etc.) so the demo isn't trivially memorized.
2. **Swap keyword confidence for a real LLM call.** In `bidding.py`,
   replace `estimate_raw_confidence()`'s body with a prompt like:
   *"On a scale of 0-1, how confident are you (a billing specialist agent)
   that you can resolve this ticket: <ticket text>? Only output a number."*
   sent through `llm_client.call_llm()`.
3. **Swap the rule-based `diagnose()` for an LLM call**, once you're
   comfortable the rule-based version proves the pipeline works. Give the
   model the investigation results (`payment`, `refund`, `invoice` dicts)
   and ask it to pick from your defined tool names — don't let it invent
   new tools.
4. **Replace `MockTechnicalAgent`** with a real call to your teammate's
   Technical Agent once it exists (likely an HTTP/WebSocket call once you
   standardize the interface — section 26 of the master doc).
5. **Expose this as a service** (e.g. a small FastAPI app with a
   `POST /bid` and `POST /handle_ticket` endpoint) so the hackathon's
   central auction/dispatch layer can call into it — this is the
   "modular enough to plug in later" requirement from the master doc.

## What this deliberately does NOT do (per the master doc's non-goals)

- No real payments, no real database, no message queue.
- No central orchestrator deciding "this ticket is billing's" — the bid
  formula is the only thing that decides that, and in the full system it
  competes against Technical's and Retention's bids.
- No finalized universal JSON schema for negotiation/tickets — this is
  intentionally left easy to reshape once the team agrees on one.

## Baby-steps setup (if you're doing this from a completely empty machine)

1. Install Python 3.10+ if you don't have it (`python3 --version` to check).
2. Unzip/copy this `billing_agent/` folder wherever you keep your code.
3. Open a terminal, `cd` into `billing_agent/`.
4. Run `python3 demo.py`. If you see the 3 ticket traces printed, you're
   done — the pipeline works.
5. Open the files in this order to understand the flow: `sandbox.py` →
   `capability.py` → `bidding.py` → `agent.py` → `negotiation.py` →
   `demo.py`.
6. Start editing `sandbox.py` first (add your own order scenarios), then
   re-run `demo.py` after every change to make sure nothing breaks.
