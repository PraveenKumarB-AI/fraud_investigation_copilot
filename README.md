# Real-Time Agentic Fraud-Investigation Copilot

An ML and LLM system that scores transactions for fraud risk using a Graph Neural Network, replays transactions in near real time, and uses an LLM agent to investigate and explain flagged transactions — retrieving relevant context and producing a structured verdict (risk score, explanation, recommended action).

> This is a learning and portfolio project. It runs on a public, anonymized benchmark dataset (the Elliptic Bitcoin dataset), not real financial institution data, and it is not a production fraud-prevention system. Nothing here should be used to make real fraud decisions.

## Modules

### Core
- [ ] **Module 1 — Project Setup.** Repo, environment, folder structure.
- [x] **Module 2 — Data & Graph Construction.** Elliptic Bitcoin dataset: 203,769 transactions (nodes), 234,355 payment flows (edges), 165 features per node, 49 time steps. Class balance: 2.2% illicit, 20.6% licit, 77.1% unlabeled — the real, imbalanced shape of fraud data, not an artifact to fix.

  Split by time step (train: steps 1-34, test: steps 35-49), matching the honest, no-shuffle evaluation discipline from the stock sentiment project. The labeled illicit rate shifts from 11.6% (train) to 6.5% (test) — a real change in fraud concentration over time, and a good illustration of why a random split would be misleading here.
- [x] **Module 3 — GNN Fraud Detection Model.** A 2-layer GraphSAGE (PyTorch Geometric), trained on the full transaction graph on a Colab T4 GPU, evaluated on an honest time-based test split (train: steps 1-27, test: steps 35-49, never overlapping).

  | Metric | Value |
  |---|---|
  | Precision (illicit) | 74.6% |
  | Recall (illicit) | 57.1% |
  | F1 (illicit) | 64.7% |
  | AUC-PR | 62.1% |
  | Baseline ("always licit") F1 | 0.0% (93.5% accuracy — misleadingly high, catches zero fraud) |

  A follow-up experiment tried standard validation-based early stopping, expecting it to improve on this fixed-epoch result — it did not. Both a loss-selected and an F1-selected checkpoint scored substantially worse on the true test set (27.3% and 40.3% F1) despite one reaching 84.6% F1 on its own validation slice. AUC-PR declined steadily across all three attempts, ruling out a threshold-calibration explanation. The cause: the validation period sits temporally adjacent to training, while the test period is further out, and this dataset has documented temporal drift (Module 2's EDA already showed the labeled illicit rate shifting from 11.6% to 6.5% across that same boundary). The fixed 100-epoch run — never chosen by peeking at any held-out set — is the honest, non-cherry-picked result, and it also happens to be the best one. Full write-up in `models/RESULTS.md`.
- [x] **Module 4 — Baseline Model Comparison.** XGBoost trained on the same 165 features and identical time-based split as the GNN (train: steps 1-27, test: steps 35-49).

  | Model | Precision | Recall | F1 | AUC-PR |
  |---|---|---|---|---|
  | GraphSAGE (Module 3) | 74.6% | 57.1% | 64.7% | 62.1% |
  | **XGBoost (this module)** | **82.5%** | **70.6%** | **76.1%** | **78.4%** |

  XGBoost won decisively on every metric. The explanation: many of the dataset's 165 features are pre-aggregated 1-hop graph-neighborhood summaries built into the data itself, so the "non-graph" baseline was never actually blind to local graph structure — and a small, lightly-tuned 2-layer GraphSAGE didn't realize enough additional benefit from deeper structure to overcome that, especially given the temporal drift documented in Module 3. This matches published academic comparisons on this exact benchmark, where tree-based models have been reported to match or beat GNNs. XGBoost is the model carried forward for transaction scoring in the rest of this project — it's also the practical choice for real-time use, since it scores a transaction from its feature vector alone, with no graph neighborhood required at inference time. Full write-up in `models/RESULTS.md`.
- [x] **Module 5 — Streaming Layer.** A single-node Kafka broker (KRaft mode, Docker) with an independent producer and consumer. The producer replays all 203,769 transactions in original time-step order (1-49); the consumer scores each one live with the Module 4 XGBoost model.

  Verified over a full end-to-end run: all 203,769 transactions sent and scored, ~10% flagged for review — consistent with the model's 82.5% precision (some honest false positives are expected at that precision level, given the true illicit rate is ~6.5%).
- [x] **Module 6 — RAG Layer for Investigation.** Every transaction (203,769) summarized into a plain-English profile (features, time step, graph connectivity), embedded with all-MiniLM-L6-v2, and stored in a persistent ChromaDB index for retrieval by the investigation agent.

  Verified with a retrieval test: querying with a real illicit transaction returned 6/6 nearest neighbors also labeled illicit, spread across four different time steps — suggesting the embeddings capture a genuine, persistent fraud pattern rather than incidental similarity. The index itself (`rag/chroma_store/`) isn't committed -- it's regenerable local data that exceeds GitHub's file size limits, same as the raw/processed data folders. Regenerate with `python -m rag.build_index`.
- [ ] **Module 7 — LLM Agent Orchestration.** A LangGraph agent that retrieves context, checks account history, and produces a structured verdict.
- [ ] **Module 8 — LLMOps: Evaluation & Monitoring.** Langfuse tracing, MLflow experiment tracking, and a labeled evaluation set for verdict accuracy.
- [ ] **Module 9 — Dashboard & API.** A Streamlit interface and a FastAPI backend.
- [ ] **Module 10 — Deployment.** A live public demo.

### Capstone
- [ ] **Module 11 — Polish & Present.** Architecture diagram, final README, demo recording.

## Tech stack
Python, PyTorch Geometric, XGBoost/LightGBM, Kafka/Redpanda, FAISS/ChromaDB, sentence-transformers, LangGraph, Groq or Ollama, MLflow, Langfuse, FastAPI, Streamlit, Docker.

## Project structure
(filled in as modules are built)
