# 🧬 Mini AlphaEvolve v0.4

**Mini AlphaEvolve v0.4** is a lightweight, interactive code evolution system built in Python and Streamlit. It enables users to evolve Python functions toward an objective using a custom evaluation function and LLM-driven mutations. This project is heavily inspired by DeepMind’s [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve/), but scoped down to run locally, simply, and transparently.

---

## 🚀 What It Does

- Evolves Python functions (starting from a seed `solution(x)`) using LLM-generated mutations
- Scores each function with a user-defined `evaluate()` function
- Stores all candidates and scores in a SQLite database
- Reuses high-scoring examples to enrich prompts
- Provides a simple web UI (Streamlit) to define objectives and run evolution

---

## 🧠 How It Relates to DeepMind’s AlphaEvolve

| Feature | Mini AlphaEvolve (This Repo) | DeepMind AlphaEvolve |
|--------|-------------------------------|-----------------------|
| **Scale** | Single-machine, Streamlit UI | Distributed, async pipeline on clusters |
| **Language support** | Python-only | Any programming language |
| **Mutation strategy** | Prompt-only whole-function rewrites | Prompt + diff + structural edits |
| **Database** | SQLite with per-objective hash | Evolutionary program database with MAP-Elites sampling |
| **Evaluation** | Single `evaluate(fn)` function | Cascade of evaluators (fast → slow), supports multiple metrics |
| **Prompting** | Manual + inspirational snippets | Rich, meta-evolved prompts with diverse formatting |
| **LLM usage** | ChatCompletion API, user-chosen model | Gemini 2.0 Pro + Flash ensemble |
| **Applications** | Educational, heuristic design, exploration | Matrix mult., combinatorics, TPU circuit design, job scheduling |
| **Reusability** | Stateless except SQLite DB | Persistent, component-wise orchestration with annotations |

---

## 🎯 Key Design Decisions

- **Objective hashing**: Candidates are grouped by the user-defined objective to ensure separation between experiments.
- **Full-code rewrites only**: To keep things simple and clean, each LLM call is expected to output a full function, no diffs.
- **SQLite + Streamlit**: Persistence and visualization out-of-the-box with no external services required.
- **Multi-model selection**: You can mix fast and strong OpenAI models (e.g., GPT-4o, GPT-3.5) per candidate.

---

## 🔍 Why Build This?

While DeepMind’s AlphaEvolve is cutting-edge, it is highly complex, GPU-intensive, and deeply integrated into Google's infrastructure. This project aims to democratize the core idea: **LLM-guided evolution using code as the medium**, but with:
- Transparent internals
- Zero cloud infrastructure dependency
- Immediate feedback loop in the browser

---

## 🧪 Try It Out

1. Clone the repo
2. Run `streamlit run app.py`
3. Enter your OpenAI key, define a scoring function, and click “Start Evolution”

---

## 📦 Future Directions

- Add diff-based evolution
- Multi-objective scoring
- Richer prompt construction
- Recombination and crossover of multiple parents

---

Inspired by [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://deepmind.google/discover/blog/alphaevolve/)
