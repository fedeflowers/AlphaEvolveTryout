import streamlit as st
import openai
import difflib
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import textwrap
import random
import json
import hashlib
import plotly.express as px
import pandas as pd

################################################################################
# Helper utilities
################################################################################

def apply_unified_diff(original_code: str, diff_text: str) -> str:
    """Apply a unified diff string to *original_code* and return the patched text.
    Falls back to returning *original_code* unchanged if the diff does not apply."""
    try:
        patched = []
        # difflib.restore needs a sequence with lines prefixed by '  ', '+ ', or '- '
        # We convert the diff into a sequence of (tag, line) with difflib.ndiff
        diff_lines = diff_text.splitlines(keepends=True)
        # Build ndiff-style list to feed to difflib.restore
        ndiff_like = []
        for line in diff_lines:
            if line.startswith("+ "):
                ndiff_like.append("+" + line[2:])
            elif line.startswith("- "):
                ndiff_like.append("-" + line[2:])
            elif line.startswith("  "):
                ndiff_like.append(" " + line[2:])
        if not ndiff_like:
            raise ValueError("Empty diff")
        restored = difflib.restore(ndiff_like, 2)  # 2 => produce final version
        return "".join(restored)
    except Exception:
        return original_code  # safety – return unchanged


def get_objective_hash(objective: str) -> str:
    """Generate a hash of the objective to detect changes."""
    return hashlib.md5(objective.encode()).hexdigest()


def code_hash(code: str) -> str:
    """Generate a hash of the code for deduplication."""
    # Normalize whitespace to avoid trivial duplicates
    normalized = "\n".join(line.rstrip() for line in code.splitlines())
    return hashlib.md5(normalized.encode()).hexdigest()


def init_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS population (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gen INTEGER,
                score REAL,
                code TEXT,
                objective_hash TEXT,
                code_hash TEXT,
                quick_score REAL
            )"""
        )
    return conn


def save_candidate(conn, gen, score, code, objective_hash, quick_score=None):
    with conn:
        conn.execute(
            """INSERT INTO population 
               (gen, score, code, objective_hash, code_hash, quick_score) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (gen, score, code, objective_hash, code_hash(code), quick_score),
        )


def sample_inspirations(conn, objective_hash, k=3):
    """Return up to *k* high‑scoring distinct code snippets for the current objective."""
    cur = conn.cursor()
    cur.execute(
        "SELECT code FROM population WHERE objective_hash = ? ORDER BY score DESC LIMIT ?",
        (objective_hash, k),
    )
    rows = cur.fetchall()
    return [row[0] for row in rows]


def reset_db_for_objective(conn, objective_hash):
    """Remove all entries for a given objective hash."""
    with conn:
        conn.execute("DELETE FROM population WHERE objective_hash = ?", (objective_hash,))


################################################################################
# Streamlit UI
################################################################################

st.set_page_config(page_title="Mini AlphaEvolve v0.5", layout="wide")
st.title("🧬 Mini AlphaEvolve — v0.5")

# Initialize session state for API key if not present
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

# Sidebar – setup
st.sidebar.header("🔐 Setup")
api_key = st.sidebar.text_input("OpenAI API Key", 
                               type="password",
                               value=st.session_state.openai_api_key)

# Update session state when API key changes
if api_key != st.session_state.openai_api_key:
    st.session_state.openai_api_key = api_key

model_small = st.sidebar.selectbox("Fast model", ["gpt-4o-mini", "gpt-3.5-turbo"])
model_big = st.sidebar.selectbox("Strong model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
strong_every = st.sidebar.number_input("Use strong model every N candidates", 1, 20, 4)

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

openai.api_key = api_key

# Main inputs
st.subheader("📌 Define the Optimization Task")
objective = st.text_area(
    "Describe the goal",
    value="Evolve a function solution(x) that transforms an integer x into a target value using a mix of nonlinear rules, conditionals, and randomness — similar to what real-world heuristics might look like.",
)

# Calculate objective hash
objective_hash = get_objective_hash(objective)

# Option to reset DB for current objective
if "last_objective_hash" not in st.session_state:
    st.session_state.last_objective_hash = objective_hash

objective_changed = objective_hash != st.session_state.last_objective_hash
if objective_changed:
    st.warning("⚠️ Objective has changed since last run!")

reset_db = st.checkbox("Reset evolution database for this objective", 
                      value=objective_changed,
                      help="Clear previous results and start fresh")

# View past candidates
if st.checkbox("📜 Show previous candidates for this objective"):
    db_path = Path("alphaevolve.sqlite")
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """SELECT gen, score, code, quick_score 
               FROM population 
               WHERE objective_hash = ? 
               ORDER BY score DESC""", 
            (objective_hash,)
        )
        rows = cur.fetchall()
        if rows:
            for gen, score, code, quick_score in rows:
                score_text = f"Score: {score:.2f}"
                if quick_score is not None:
                    score_text += f" (Quick: {quick_score:.2f})"
                with st.expander(f"Gen {gen} — {score_text}"):
                    st.code(code, language="python")
        else:
            st.info("No previous candidates found for this objective.")
        conn.close()

st.markdown("#### Quick evaluation (cheap, optional)")
quick_eval_code = st.text_area(
    "Return a numeric score quickly; may raise on failure. Empty = skip.",
    value=textwrap.dedent(
        """
        def quick_evaluate(solution_fn):
            import math
            
            # Test only 5 key cases for quick feedback
            test_cases = [-5, -1, 0, 1, 5]
            score = 0
            
            def quick_target(x):
                if x < 0:
                    return int(math.sqrt(abs(x)) * 3 + 5)
                elif x == 0:
                    return 42
                else:
                    return int(math.log(x + 1) * 7 + x % 3)
            
            try:
                for x in test_cases:
                    target = quick_target(x)
                    guess = solution_fn(x)
                    # Use same scoring as full eval
                    diff = abs(guess - target)
                    score += max(0, 10 - diff)
                return score / len(test_cases)  # Normalize to 0-10 range
            except Exception:
                return 0
        """
    ),
)

quick_score_threshold = None
if quick_eval_code.strip():
    quick_score_threshold = st.slider(
        "Quick evaluation score threshold (skip if below)",
        0.0, 10.0, 3.0,
        help="Skip full evaluation for candidates scoring below this threshold"
    )

st.markdown("#### Full evaluation (expensive)")
full_eval_code = st.text_area(
    "Must return a numeric score (higher=better)",
    value=textwrap.dedent(
        """
        def evaluate(solution_fn):
            import math
            import random

            score = 0
            test_cases = list(range(-10, 11))  # test integers from -10 to 10

            def target_function(x):
                if x < 0:
                    return int(math.sqrt(abs(x)) * 3 + 5)
                elif x == 0:
                    return 42
                else:
                    return int(math.log(x + 1) * 7 + x % 3)

            for x in test_cases:
                try:
                    target = target_function(x)
                    guess = solution_fn(x)
                    # Penalize large deviations
                    diff = abs(guess - target)
                    score += max(0, 10 - diff)  # soft score
                except:
                    pass

            return score
                """
    ),
)

initial_code = st.text_area(
    "Initial function to evolve (must define `def solution(x): ...`)",
    value=textwrap.dedent(
        """
        def solution(x):
            return 2 * x 
        """
    ).strip(),
)

# Evolution hyper‑params
col1, col2 = st.columns(2)
num_generations = col1.slider("Generations", 1, 20, 4)
candidates_per_gen = col1.slider("Candidates / generation", 1, 10, 4)
population_size = col2.slider("Population size to keep", 1, 50, 10)
inspiration_k = col2.slider("Inspirations in prompt", 0, 5, 3)

run = st.button("🚀 Start Evolution")

################################################################################
# Evolution logic (v0.5)
################################################################################

def load_eval_functions():
    namespace = {}
    if quick_eval_code.strip():
        exec(quick_eval_code, {}, namespace)
        quick_evaluate = namespace.get("quick_evaluate")
    else:
        quick_evaluate = None

    namespace = {}
    exec(full_eval_code, {}, namespace)
    full_evaluate = namespace["evaluate"]
    return quick_evaluate, full_evaluate


def get_parent(pop):
    """Pick a parent with probability ∝ rank (lower rank => higher prob)."""
    if not pop:
        return initial_code
    weights = [1 / (i + 1) for i in range(len(pop))]  # rank‑based
    codes = [c for _, _, c in pop]
    return random.choices(codes, weights)[0]


if run:
    # Update last objective hash
    st.session_state.last_objective_hash = objective_hash

    # Prepare evaluation functions
    quick_evaluate, full_evaluate = load_eval_functions()

    # DB setup in ./alphaevolve.sqlite (persistent across runs)
    db_path = Path("alphaevolve.sqlite")
    conn = init_db(db_path)

    # Reset DB for this objective if requested
    if reset_db:
        reset_db_for_objective(conn, objective_hash)
        st.success("🗑️ Database cleared for current objective")

    # Track seen code hashes for deduplication
    seen_hashes = set()
    
    # Load existing code hashes for this objective
    cur = conn.cursor()
    cur.execute("SELECT code_hash FROM population WHERE objective_hash = ?", (objective_hash,))
    seen_hashes.update(row[0] for row in cur.fetchall())

    # Seed population with initial code
    population = []  # list of tuples (gen, score, code)

    # Evaluate initial code
    ns = {}
    exec(initial_code, {}, ns)
    sol_fn = ns["solution"]
    
    # Quick evaluate if available
    quick_score = None
    if quick_evaluate:
        try:
            quick_score = quick_evaluate(sol_fn)
        except Exception:
            pass
    
    base_score = full_evaluate(sol_fn)
    population.append((0, base_score, initial_code))
    save_candidate(conn, 0, base_score, initial_code, objective_hash, quick_score)
    
    # Add initial code hash
    seen_hashes.add(code_hash(initial_code))

    client = openai.OpenAI(api_key=api_key)

    st.subheader("🧬 Evolution Progress")
    
    # Setup progress tracking
    progress_data = []

    for gen in range(1, num_generations + 1):
        st.markdown(f"### Generation {gen}")

        parent_code = get_parent(population)
        inspirations = sample_inspirations(conn, objective_hash, k=inspiration_k)

        candidates = []  # (code, score)
        with st.spinner("Evolving generation…"):
            for cand_idx in range(candidates_per_gen):
                use_big = (cand_idx % strong_every == 0)
                model = model_big if use_big else model_small

                prompt_parts = [
                    "You are AlphaEvolve‑Mini, a code evolution system. Your ONLY task is to generate Python function variations.",
                    f"Objective:\n{objective}",
                    "---\nCURRENT IMPLEMENTATION:\n" + parent_code,
                ]
                if inspirations:
                    prompt_parts.append("---\nHIGH SCORING EXAMPLES:\n" + "\n---\n".join(inspirations))

                prompt_parts.append(
                    textwrap.dedent(
                        """
                        RESPONSE RULES:
                        1. Return ONLY a complete Python function definition
                        2. Function MUST start with exactly: def solution(x):
                        3. NO explanations, NO markdown, NO backticks
                        4. ONLY return the function code with proper indentation
                        
                        Example valid response:
                        def solution(x):
                            return x * 3
                        """
                    )
                )

                prompt = "\n\n".join(prompt_parts)

                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                raw_reply = completion.choices[0].message.content.strip()

                # Clean up the response
                lines = raw_reply.splitlines()
                cleaned_lines = []
                started = False
                
                for line in lines:
                    line = line.rstrip()
                    if line.startswith("def solution(x):"):
                        started = True
                        cleaned_lines = [line]  # Reset if we find another function def
                    elif started and line:
                        if not line[0].isspace() and "def" not in line:
                            break  # Stop if we hit non-indented non-def line
                        cleaned_lines.append(line)
                
                new_code = "\n".join(cleaned_lines)

                if not new_code.startswith("def solution(x):"):
                    st.text("⚠️ Candidate discarded – invalid function definition")
                    continue

                # Check for duplicates
                h = code_hash(new_code)
                if h in seen_hashes:
                    st.text("⚠️ Candidate discarded – duplicate solution")
                    continue
                seen_hashes.add(h)

                # Validate the function
                try:
                    ns = {}
                    exec(new_code, {}, ns)
                    if "solution" not in ns or not callable(ns["solution"]):
                        st.text("⚠️ Candidate discarded – invalid function")
                        continue
                except Exception as e:
                    st.text(f"⚠️ Candidate discarded – syntax error: {str(e)}")
                    continue

                candidates.append(new_code)

            # Full evaluation in parallel
            scored_candidates = []

            with ThreadPoolExecutor() as pool:
                future_to_code = {}
                for code in candidates:
                    ns = {}
                    try:
                        exec(code, {}, ns)
                        fn = ns["solution"]
                        
                        # Quick evaluate if available
                        if quick_evaluate and quick_score_threshold is not None:
                            try:
                                q_score = quick_evaluate(fn)
                                if q_score < quick_score_threshold:
                                    st.text(f"⚠️ Candidate discarded – quick score too low: {q_score:.2f}")
                                    continue
                            except Exception:
                                continue
                        
                    except Exception:
                        continue
                    future = pool.submit(full_evaluate, fn)
                    future_to_code[future] = (code, q_score if quick_evaluate else None)

                for future in as_completed(future_to_code):
                    code, quick_score = future_to_code[future]
                    try:
                        score = future.result()
                    except Exception:
                        continue
                    scored_candidates.append((code, score, quick_score))
                    score_text = f"Score: {score:.2f}"
                    if quick_score is not None:
                        score_text += f" (Quick: {quick_score:.2f})"
                    st.text(f"Candidate {score_text}")

            # Merge population + new candidates, keep best unique by score
            for code, score, quick_score in scored_candidates:
                population.append((gen, score, code))
                save_candidate(conn, gen, score, code, objective_hash, quick_score)
                progress_data.append({"Generation": gen, "Score": score})

            # Sort & truncate population
            population.sort(key=lambda t: t[1], reverse=True)
            if len(population) > population_size:
                population = population[:population_size]

            best_gen_score = population[0][1]
            st.write(f"**Best score after generation {gen}: {best_gen_score}**")

    # Final best
    best_overall = population[0]
    st.success("🎉 Evolution complete")
    st.markdown("### 🏆 Best discovered function")
    st.code(best_overall[2], language="python")
    st.write(f"**Final score:** {best_overall[1]}")

    # Plot score progression
    if progress_data:
        st.subheader("📈 Score Progression")
        progress_df = pd.DataFrame(progress_data)
        fig = px.line(progress_df, 
                     x="Generation", 
                     y="Score",
                     title="Evolution Progress",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)

    # Close DB connection
    conn.close()
