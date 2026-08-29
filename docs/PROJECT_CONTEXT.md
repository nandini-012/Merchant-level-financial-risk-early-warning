# PROJECT CONTEXT

## Project

**Working Name:** Merchant Risk Early-Warning & Loss-Exposure Detector

**Project:** Razorpay AI Builder Internship 2026 Buildathon

**Development Constraint:** Approximately 9 days / 20–30 focused engineering hours

**Current Phase:** Step 1 — Project Definition & Feasibility

**Status:** Candidate problem defined; technical feasibility not yet validated.

---

## 1. Objective

Build a genuinely working, defensible AI/ML system that addresses one clearly
defined financial-loss/risk problem relevant to a payment ecosystem.

The goal is to maximize the probability of producing a shortlist-worthy
submission rather than maximizing feature count or technical complexity.

The system must:

- solve one clearly defined problem
- use meaningful ML
- have rigorous held-out evaluation
- report precision, recall and F1
- quantify false-positive cost
- be explainable
- remain strictly defensive
- be understandable and defensible by a 3rd-year CSE/AI-ML student
- avoid unnecessary features and buzzwords

---

## 2. Candidate Problem

Detect unusual merchant-level behavioural patterns that may indicate an
emerging fraud/risk episode and provide an explainable early warning.

The system should attempt to identify meaningful changes in merchant behaviour,
rather than treating every unusual transaction as fraud.

---

## 3. Conceptual Pipeline

Transactions
    ↓
Transaction-level risk signals
    ↓
Merchant behavioural baseline
    ↓
Temporal behavioural analysis
    ↓
Merchant-level risk signal
    ↓
Potential transaction-value exposure
    ↓
Explainable alert
    ↓
Recommended investigation/risk-control action

This architecture is conceptual and NOT YET FROZEN.

---

## 4. Primary User

TBD.

Candidate:
- Payment-risk analyst / risk operations analyst

Must be validated before finalization.

---

## 5. Decision Supported

TBD.

Potential decisions:
- investigate
- prioritize investigation
- review
- apply an appropriate risk-control process

The system should not automatically perform consequential financial actions
unless explicitly justified, authorized, and supported by the project scope.

---

## 6. Input Data

TBD.

Required properties will likely include some combination of:

- transaction identifier
- merchant identifier
- timestamp
- transaction amount
- fraud/risk label or another defensible target
- relevant transaction attributes

These are requirements to investigate, NOT assumptions that a particular
dataset contains them.

---

## 7. ML Role

ML must have a measurable and necessary role.

The project must not use ML merely because it is an AI/ML project.

The final formulation must establish:

1. what the model predicts
2. why ML is appropriate
3. what the target/label represents
4. how predictions are evaluated
5. what happens when predictions are wrong

---

## 8. Loss / Exposure

The system may quantify transaction-value exposure associated with a detected
risk event.

"Exposure" must not automatically be represented as guaranteed or expected
financial loss.

Any monetary estimate must have an explicit mathematical definition and
clearly state its assumptions.

---

## 9. Evaluation Requirements

Minimum required metrics:

- Precision
- Recall
- F1
- Confusion matrix
- False-positive cost

Evaluation must use a genuinely held-out test set.

Temporal leakage must be investigated carefully because the problem concerns
merchant behaviour over time.

Accuracy alone is not sufficient evidence of model quality.

---

## 10. Responsible AI

The system is strictly defensive.

Potential issues to evaluate:

- false positives
- false negatives
- merchant impact
- analyst workload
- explainability
- data leakage
- bias in available data
- inappropriate automated decisions

The system should support human investigation rather than blindly replacing
human judgement where the consequences are significant.

---

## 11. Razorpay Claims

Do NOT claim that Razorpay uses a particular internal system, dataset,
algorithm, settlement process, fraud-control mechanism, or data source unless
supported by an authoritative public source.

Do NOT assume access to:

- Razorpay internal transaction data
- cross-gateway transaction visibility
- internal merchant risk scores
- internal fraud labels
- internal settlement systems
- private customer/merchant information
- newly-created UPI-handle information
- any other proprietary/internal infrastructure

Publicly documented Razorpay capabilities and our project's proposed
architecture must be clearly distinguished.

---

## 12. Scope

### MUST HAVE

TBD after data feasibility analysis.

### SHOULD HAVE

TBD.

### COULD HAVE

TBD.

### EXPLICITLY OUT OF SCOPE

- unnecessary microservices
- unnecessary distributed infrastructure
- LLM-based fraud scoring without a demonstrated need
- complex models without measurable benefit
- automatic financial actions without proper justification
- unsupported claims about Razorpay internal systems
- features added solely for presentation value

---

## 13. Architecture

Status: NOT FROZEN

---

## 14. Dataset

Status: NOT SELECTED

---

## 15. Model

Status: NOT SELECTED

---

## 16. Backend

Status: NOT STARTED

---

## 17. Frontend

Status: NOT STARTED

---

## 18. Experiments

Experiments will use unique IDs:

EXP-001
EXP-002
EXP-003
...

Each experiment should record:

- hypothesis
- dataset/version
- features
- model
- parameters
- validation strategy
- metrics
- result
- interpretation
- decision

---

## 19. Engineering Methodology

The project will use rapid iterative development with:

- small implementation loops
- explicit acceptance criteria
- continuous verification
- ML experimentation
- Git-based version control

Development methodology should optimize for the limited project timeline
without sacrificing evaluation quality.

---

## 20. AI Agent Rules

Any external AI coding/research agent must:

1. Inspect existing project state before acting.
2. Never invent unavailable facts.
3. Clearly distinguish FACT, OBSERVATION, ASSUMPTION and PROPOSAL.
4. Never silently change architecture.
5. Never silently expand scope.
6. Never claim a test passed unless it was actually executed.
7. Report failures honestly.
8. Avoid modifying unrelated files.
9. Stop and request review when a task requires an architectural change.
10. Provide evidence for important technical claims.

---

## 21. Decision Authority

CHAT 00 — Command Center

External AI agents are execution/research assistants.

They do not independently change:

- project objective
- scope
- architecture
- ML formulation
- evaluation methodology
- technology choices with major consequences

without Command Center approval.

---

## 22. Current Risks

1. Merchant-level ground-truth definition is unresolved.
2. Dataset suitability is unresolved.
3. Temporal leakage risk is unresolved.
4. "Anomaly" must not automatically be equated with "fraud".
5. Monetary exposure must not be presented as guaranteed loss.
6. Actual Razorpay relevance must be supported without inventing internal details.

---

## 23. Current Next Action

Determine whether an available dataset can support a defensible merchant-level
risk/fraud formulation and held-out evaluation.

---

## 24. Status

## Target Definition — LOOP 002D

Primary prediction target:

future_7_fraud >= 2

Definition:
At a merchant-day observation T, the target is 1 when the merchant
experiences at least 2 fraudulent transactions during the subsequent
7-day period; otherwise 0.

Prediction horizon:
7 days.

Rationale:
- 6,270 positive observations in the training analysis window
- 1.9148% positive rate
- Represents repeated future fraud activity rather than a single event
- Provides substantially more positive examples than >=3 fraud
- Better aligned with merchant-level early-warning than >=1 fraud

Alternative targets evaluated:
- future_7_fraud >= 1: 12.9605%
- future_7_fraud >= 3: 0.2703%
- future fraud amount thresholds
- future rate relative to historical baseline

These alternatives remain candidates for sensitivity analysis but are
not the primary target.

Important:
Future fraud variables are labels/outcomes only.
They must never be used as model features.

## LOOP 003 — Feature Specification

### Prediction point

Each training observation represents a merchant at day T.

The model may use only information available at or before T.

The model predicts whether the merchant will experience at least
2 fraudulent transactions during the subsequent 7-day period.

### Primary target

future_7_fraud >= 2

### Feature design principles

1. Features must be computable using data available at prediction time.
2. Future observations from T+1 through T+7 must never be used as features.
3. Future fraud labels must never be used as features.
4. Features should describe merchant behaviour rather than individual
   consumer identity.
5. Features should be explainable to a technical evaluator.
6. Feature count should remain small enough for the team to understand
   and defend.
7. The feature pipeline must be reproducible from the raw training data.
8. The same feature-generation logic must later be applied to the
   chronological test period.

### Tier A — Mandatory features

Transaction velocity:
- transaction count in previous 1 day
- transaction count in previous 3 days
- transaction count in previous 7 days
- transaction count in previous 14 days

Fraud behaviour:
- fraud count in previous 7 days
- fraud count in previous 14 days
- fraud rate in previous 7 days
- fraud rate in previous 14 days

Behaviour change:
- recent 7-day transaction count compared with previous 7-day period
- recent 7-day fraud count compared with previous 7-day period

Transaction amount:
- total transaction amount in previous 7 days
- average transaction amount in previous 7 days
- maximum transaction amount in previous 7 days

### Tier B — Candidate features

- transaction amount standard deviation
- fraud amount in previous 7 days
- fraud amount in previous 14 days
- active days in previous 7/14 days
- concentration of transactions across active days
- hour-of-day activity statistics

Tier B features must be evaluated empirically before being retained.

### Tier C — Excluded unless later justified

- customer first/last name
- credit card number
- transaction number
- date of birth
- gender
- street address
- raw ZIP code
- raw customer identity fields

Reason:
The project is intended to detect merchant-level behavioural risk.
These fields add complexity and may shift the system toward customer-level
classification rather than the intended merchant-level problem.

### Leakage controls

The feature pipeline must guarantee:

feature_time <= prediction_time

and must never access:

prediction_time + 1 day through prediction_time + 7 days.

The target may use the future window, but features may not.

### Test-set policy

fraudTest.csv must remain completely untouched until the final
chronological evaluation stage.

No threshold tuning, feature selection, model selection, or hyperparameter
selection may use the test set.

### Current decision

Do not train the final model yet.

Next implementation task:
construct a reproducible merchant-day feature dataset from fraudTrain.csv
using only historical information.


## Operational Decision — LOOP 006E

Operational alert-capacity assumption:

20 merchant alerts per day.

Rationale:
The 20-alert/day operating point provides a defensible balance between
early-warning coverage and analyst alert burden. On the validation period,
this corresponds to 1,220 alerts over 61 days, with 113 true positives,
1,107 false positives, 476 false negatives, and 37,303 true negatives.

The resulting validation threshold is:

0.814822766216

The threshold was derived exclusively from the validation period and was
frozen before final holdout evaluation.

Final holdout performance at this locked threshold:

- Alerts: 912
- Alerts/day: 20.2667
- True positives: 79
- False positives: 833
- False negatives: 530
- True negatives: 27,761
- Precision: 0.086623
- Recall: 0.129721
- F1: 0.103879
- ROC-AUC: 0.820637
- PR-AUC: 0.071587
- False-positive rate: 0.029132
- False alerts/day: 18.5111

The holdout was evaluated only after the threshold was locked and was not
used for threshold selection.

This is an operational modelling assumption, not a claim about actual
Razorpay analyst capacity or financial operating costs.