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

# Load all population data
df = pd.read_sql_query("SELECT * FROM population", conn)

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
        format_func=lambda x: f"Objective {objectives.index.get_loc(x) + 1} ({objectives[x]} solutions)"
    )
    df = df[df['objective_hash'] == selected_hash]

# Display summary statistics
st.subheader("📊 Population Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Solutions", len(df))
    
with col2:
    st.metric("Best Score", f"{df['score'].max():.2f}")
    
with col3:
    st.metric("Generations", df['gen'].max() + 1)

# Score distribution plot
st.subheader("Score Distribution Over Generations")
fig = px.scatter(df, x='gen', y='score', 
                 title='Evolution Progress',
                 labels={'gen': 'Generation', 'score': 'Score'},
                 trendline="lowess")
st.plotly_chart(fig, use_container_width=True)

# View top solutions
st.subheader("🏆 Top Solutions")
top_n = st.slider("Number of top solutions to view", 1, 20, 5)

top_solutions = df.nlargest(top_n, 'score')
for idx, row in top_solutions.iterrows():
    with st.expander(f"Score: {row['score']:.2f} (Generation {row['gen']})"):
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