# Real-Time Agentic Fraud-Investigation Copilot

An ML and LLM system that scores transactions for fraud risk using a Graph Neural Network, replays transactions in near real time, and uses an LLM agent to investigate and explain flagged transactions — retrieving relevant context and producing a structured verdict (risk score, explanation, recommended action).

> This is a learning and portfolio project. It runs on a public, anonymized benchmark dataset (the Elliptic Bitcoin dataset), not real financial institution data, and it is not a production fraud-prevention system. Nothing here should be used to make real fraud decisions.

## Modules

### Core
- [ ] **Module 1 — Project Setup.** Repo, environment, folder structure.
- [ ] **Module 2 — Data & Graph Construction.** Elliptic Bitcoin dataset — EDA and graph schema (nodes = transactions, edges = payment flows).
- [ ] **Module 3 — GNN Fraud Detection Model.** GraphSAGE/GAT via PyTorch Geometric, trained on a Colab GPU.
- [ ] **Module 4 — Baseline Model Comparison.** XGBoost/LightGBM on the same features, compared against the GNN.
- [ ] **Module 5 — Streaming Layer.** Kafka/Redpanda in Docker, replaying transactions by time step and scoring them in near real time.
- [ ] **Module 6 — RAG Layer for Investigation.** A vector store of account history and case notes for flagged transactions.
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
