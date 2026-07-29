"""avtext — Aviation-Text LLM Lab.

A trust-laddered pipeline for turning raw aviation text (METAR/TAF first) into
structured data, an eval harness to score models on that task, and the machinery
to fine-tune and quantize a small open model against it.

Data flows one direction through the layers:

    ingest -> schema -> oracles -> quality -> consensus -> tasks -> harness
                                                                       |
                                                            finetune  <┘ (Phase 4+)

Each layer is importable and testable without the next.
"""

__version__ = "0.0.1"
