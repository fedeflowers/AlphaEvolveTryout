import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="Population Viewer", layout="wide")
st.title("🔍 Population History Viewer")

# Access the shared API key from session state
api_key = st.session_state.get("openai_api_key", "")
if not api_key:
    st.warning("No API key found. Please enter your OpenAI API key in the main page.")

# Connect to the database
db_path = Path("alphaevolve.sqlite")
if not db_path.exists():
    st.error("No evolution database found. Please run some evolution first!")
    st.stop()

conn = sqlite3.connect(db_path)

# Check if we have the new schema with objective_hash
cur = conn.cursor()
has_objective_hash = False
has_quick_score = False
try:
    cur.execute("SELECT objective_hash FROM population LIMIT 1")
    has_objective_hash = True
    cur.execute("SELECT quick_score FROM population LIMIT 1")
    has_quick_score = True
except sqlite3.OperationalError:
    pass

# Load all population data
if has_objective_hash:
    df = pd.read_sql_query("SELECT * FROM population", conn)
else:
    # Old schema - create a dummy objective hash column
    df = pd.read_sql_query("SELECT *, 'legacy' as objective_hash FROM population", conn)

if len(df) == 0:
    st.warning("No evolution data found. Try running some evolution first!")
    st.stop()

# Get unique objectives
objectives = df.groupby('objective_hash')['gen'].count().sort_values(ascending=False)

# Let user select which objective to analyze
if len(objectives) > 1:
    st.subheader("🎯 Select Objective")
    selected_hash = st.selectbox(
        "Choose which evolution objective to analyze:",
        options=objectives.index,
        format_func=lambda x: f"{'Legacy Data' if x == 'legacy' else f'Objective {objectives.index.get_loc(x) + 1}'} ({objectives[x]} solutions)"
    )
    df = df[df['objective_hash'] == selected_hash]

# Display summary statistics
st.subheader("📊 Population Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Solutions", len(df))
    
with col2:
    st.metric("Best Score", f"{df['score'].max():.2f}")
    
with col3:
    st.metric("Generations", df['gen'].max() + 1)

with col4:
    if has_quick_score:
        quick_scores = df['quick_score'].dropna()
        if len(quick_scores) > 0:
            st.metric("Best Quick Score", f"{quick_scores.max():.2f}")

# Score distribution plot
st.subheader("Score Distribution Over Generations")
fig = px.scatter(df, x='gen', y='score', 
                 title='Evolution Progress',
                 labels={'gen': 'Generation', 'score': 'Score'},
                 trendline="lowess")
st.plotly_chart(fig, use_container_width=True)

if has_quick_score:
    # Quick vs Full Score Analysis
    quick_score_df = df[df['quick_score'].notna()]
    if len(quick_score_df) > 0:
        st.subheader("🔄 Quick vs Full Score Correlation")
        fig_corr = px.scatter(quick_score_df, 
                            x='quick_score', 
                            y='score',
                            title='Quick vs Full Score Correlation',
                            labels={'quick_score': 'Quick Score', 'score': 'Full Score'},
                            trendline="ols")
        st.plotly_chart(fig_corr, use_container_width=True)

# View top solutions
st.subheader("🏆 Top Solutions")
top_n = st.slider("Number of top solutions to view", 1, 20, 5)

top_solutions = df.nlargest(top_n, 'score')
for idx, row in top_solutions.iterrows():
    score_text = f"Score: {row['score']:.2f}"
    if has_quick_score and pd.notna(row['quick_score']):
        score_text += f" (Quick: {row['quick_score']:.2f})"
    with st.expander(f"Gen {row['gen']} — {score_text}"):
        st.code(row['code'], language='python')

# Generation analysis
st.subheader("📈 Generation Analysis")
gen_stats = df.groupby('gen').agg({
    'score': ['mean', 'max', 'min', 'count']
}).reset_index()
gen_stats.columns = ['Generation', 'Mean Score', 'Max Score', 'Min Score', 'Solutions']

st.dataframe(gen_stats)

# Score distribution histogram
st.subheader("Score Distribution")
fig_hist = px.histogram(df, x='score', nbins=30,
                       title='Distribution of Solution Scores',
                       labels={'score': 'Score', 'count': 'Number of Solutions'})
st.plotly_chart(fig_hist, use_container_width=True)

# Close the database connection
conn.close() 