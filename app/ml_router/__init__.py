"""Milestone 22: an independent, local ML intent classifier (TF-IDF +
Logistic Regression) that predicts one of DIRECT / WEB / TIME / DATE for a
user message.

--------------------------------------------------------------------------
NOT WIRED INTO PRODUCTION — read this before importing this package
elsewhere
--------------------------------------------------------------------------
Nothing in `app/agent/`, `app/services/`, or `app/main.py` imports
anything from this package, and nothing in this package imports from
`app/agent/tool_execution.py`, `app/agent/permissions.py`,
`app/agent/loop.py`, or `app/agent/orchestrator.py`. That is deliberate
and load-bearing, not an oversight:

- Milestone 22's charter is to BUILD and BENCHMARK a candidate classifier,
  not to integrate one. Production routing remains exactly what Milestone
  20/21 left it: `Router.classify_hint()` (regex-based, advisory) plus
  `LLMDecisionMaker._deterministic_temporal_decision` (the Milestone-20
  deterministic time/date override) plus the LLM's own decision.
- This classifier's output is a plain string label with a confidence
  score — nothing more. It has no method that touches a `Tool`, a
  `ToolRegistry`, a `PermissionPolicy`, or an `ExecutionContext`, and
  cannot execute, authorize, or confirm anything, structurally: there is
  no code path from this package to any of those.
- Integrating this classifier into the live decision path (replacing or
  augmenting `RoutingHint`, or gating tool selection) is an explicit,
  separate, future decision — to be made from the benchmark numbers this
  milestone produces, not assumed by building the classifier.

See `app/ml_router/contract.py` for the classification contract (what
DIRECT/WEB/TIME/DATE mean and why), `app/ml_router/dataset.py` for the
versioned training data, `app/ml_router/classifier.py` for the model
itself, and `app/ml_router/train.py` for the training/evaluation entry
point.
"""
