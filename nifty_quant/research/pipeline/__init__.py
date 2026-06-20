"""Daily Research Pipeline: a staged, read-only orchestration layer.

This sub-package turns a trading day's collected option-chain and candle data
into a concise daily research report and an evidence-based research journal. It
is a thin orchestration layer over the frozen ``nifty_quant`` v0.2 components
(``Storage``, ``FeatureEngine``, ``FeatureStore``, ``analytics.options``,
``ValidationEngine``, ``ResearchJournal``, ``research.regime``) and introduces
no new architecture and modifies no existing interface.

The pipeline is read-only analysis: it never places, modifies, or simulates a
trade or order. No broker/order modules are imported from this package.

The orchestrator runs eight stages in a fixed order:
``quality-gate -> read -> features -> validation -> comparison -> evidence
-> ideas -> report``.
"""
