import streamlit as st
import openai

#alphaevolve-env\Scripts\activate.bat


st.set_page_config(page_title="Mini AlphaEvolve", layout="wide")
st.title("🧬 Mini AlphaEvolve — Evolve Python Code with OpenAI")

# Sidebar: API key and setup
st.sidebar.header("🔐 Setup")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
model = st.sidebar.selectbox("OpenAI Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

openai.api_key = api_key

# Main inputs
st.subheader("📌 Define the Optimization Task")
objective = st.text_area(
    "Describe what you want the function to do (e.g., 'Return triple of input')",
    value="Evolve a function solution(x) that transforms an integer x into a target value using a mix of nonlinear rules, conditionals, and randomness — similar to what real-world heuristics might look like."
)

eval_code = st.text_area(
    "Write an evaluation function to score candidates.\nIt should return a score (higher = better).",
    value="""
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
)

initial_code = st.text_area(
    "Initial function to evolve (must define `def solution(x): ...`)",
    value="""
def solution(x):
    # Initial stub that just returns something plausible
    return x * 2
"""
)

num_generations = st.slider("Number of generations", 1, 10, 3)
candidates_per_gen = st.slider("Candidates per generation", 1, 5, 2)
run = st.button("🚀 Start Evolution")

# Evolution logic
if run:
    st.subheader("🧬 Evolution Progress")
    best_code = initial_code
    namespace = {}
    exec(eval_code, {}, namespace)
    evaluate = namespace["evaluate"]

    namespace = {}
    exec(best_code, {}, namespace)
    solution_fn = namespace["solution"]
    best_score = evaluate(solution_fn)

    with st.spinner("Evolving..."):
        for gen in range(num_generations):
            st.markdown(f"### Generation {gen + 1}")
            st.code(best_code, language="python")
            st.write(f"**Current best score:** {best_score}")

            candidates = []
            for _ in range(candidates_per_gen):
                prompt = f"""
You are an AI trying to improve a Python function based on the following objective:

**Objective**: {objective}

Current implementation:
{best_code}

Your task is to suggest a better implementation of the solution function.

IMPORTANT:
1. Only return the Python function code starting with 'def solution(x):'
2. Do not include any explanation, markdown formatting, or additional text
3. The function must be named 'solution' and take a single parameter 'x'
4. Return only valid Python code

Example of correct response format:
def solution(x):
    return x * 3
"""
                client = openai.OpenAI(api_key=api_key)

                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                new_code = response.choices[0].message.content.strip()
                
                # Clean up the code response
                if "```python" in new_code:
                    # Extract code from markdown code blocks
                    new_code = new_code.split("```python")[1].split("```")[0].strip()
                elif "```" in new_code:
                    # Extract code from generic code blocks
                    new_code = new_code.split("```")[1].strip()
                
                # Ensure the code starts with the function definition
                if not new_code.startswith("def solution"):
                    st.text("Invalid response format - missing function definition")
                    continue

                try:
                    namespace = {}  # Create a fresh namespace for each candidate
                    exec(new_code, {}, namespace)
                    new_score = evaluate(namespace["solution"])
                    candidates.append((new_code, new_score))
                    st.text(f"Candidate score: {new_score}")
                except Exception as e:
                    st.text(f"Candidate failed to execute. Error: {str(e)}")
                    st.code(new_code, language="python")  # Show the failing code

            # Keep the best
            best_candidate = max(candidates, key=lambda x: x[1], default=(best_code, best_score))
            if best_candidate[1] > best_score:
                best_code, best_score = best_candidate

    st.success("🎉 Evolution Complete!")
    st.markdown("### 🏆 Best Discovered Function:")
    st.code(best_code, language="python")
    st.write(f"**Final score:** {best_score}")
