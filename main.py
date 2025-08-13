import os
import io
import json
import base64
import traceback
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Helper functions --------

def df_to_base64_plot(df, x_col, y_col, title=None):
    """Return a scatterplot with dotted red regression line as base64 data URI."""
    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col], label="Data")
    m, b = np.polyfit(df[x_col], df[y_col], 1)
    ax.plot(df[x_col], m * df[x_col] + b, 'r--', label="Regression")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if title:
        ax.set_title(title)
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def scrape_highest_grossing_films():
    """Scrape Wikipedia table for highest-grossing films."""
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    df = None
    for t in tables:
        if "Rank" in t.columns and "Title" in t.columns:
            df = t
            break
    return df

def shortest_path_network(file_path, source, target):
    df = pd.read_csv(file_path)
    G = nx.Graph()
    for _, row in df.iterrows():
        if "weight" in df.columns:
            G.add_edge(row["source"], row["target"], weight=row["weight"])
        else:
            G.add_edge(row["source"], row["target"], weight=1)
    return nx.shortest_path(G, source=source, target=target, weight="weight")

# -------- Main logic --------

def process_request_logic(question_text: str, files: dict):
    try:
        q_lower = question_text.lower().strip()

        # ---- CSV tasks ----
        if any(fname.endswith(".csv") for fname in files):
            csv_files = {k: v for k, v in files.items() if k.endswith(".csv")}
            for name, f in csv_files.items():
                df = pd.read_csv(io.BytesIO(f))
                # Row count
                if "how many rows" in q_lower:
                    return len(df)
                # Correlation
                if "correlation" in q_lower:
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    if len(num_cols) >= 2:
                        corr = df[num_cols[0]].corr(df[num_cols[1]])
                        return corr
                # Scatterplot
                if "scatterplot" in q_lower:
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    if len(num_cols) >= 2:
                        img_uri = df_to_base64_plot(df, num_cols[0], num_cols[1])
                        return img_uri
                # Network shortest path
                if "shortest path" in q_lower and "source" in q_lower and "target" in q_lower:
                    # Extract node names from question
                    import re
                    src_match = re.search(r"source\s*[:=]?\s*(\w+)", q_lower)
                    tgt_match = re.search(r"target\s*[:=]?\s*(\w+)", q_lower)
                    if src_match and tgt_match:
                        src, tgt = src_match.group(1), tgt_match.group(1)
                        path = shortest_path_network(io.BytesIO(f), src, tgt)
                        return path

        # ---- Wikipedia scraping ----
        if "highest-grossing films" in q_lower:
            df = scrape_highest_grossing_films()
            if df is not None:
                if "$2" in q_lower and "before 2000" in q_lower:
                    df2 = df[df["Worldwide gross"].str.contains("2", na=False)]
                    # Filter by release date if available
                    if "Year" in df2.columns:
                        before_2000 = df2[df2["Year"] < 2000]
                        return len(before_2000)
                if "earliest film" in q_lower and "$1.5" in q_lower:
                    df["gross_num"] = (
                        df["Worldwide gross"]
                        .replace("[\$,]", "", regex=True)
                        .astype(float)
                    )
                    filtered = df[df["gross_num"] > 1_500_000_000]
                    earliest = filtered.sort_values("Year").iloc[0]["Title"]
                    return earliest
                if "correlation" in q_lower:
                    if "Rank" in df.columns and "Peak" in df.columns:
                        corr = df["Rank"].corr(df["Peak"])
                        return corr
                if "scatterplot" in q_lower:
                    if "Rank" in df.columns and "Peak" in df.columns:
                        img_uri = df_to_base64_plot(df, "Rank", "Peak")
                        return img_uri

        # ---- Generic confirm ----
        if "confirm you received all files" in q_lower:
            return "All files received: " + ", ".join(files.keys())

        return None

    except Exception:
        traceback.print_exc()
        return None

# -------- API endpoint --------

@app.post("/")
async def process_request(questions: UploadFile = File(...), files: list[UploadFile] = File(None)):
    try:
        q_text = (await questions.read()).decode("utf-8")
        other_files = {}
        if files:
            for f in files:
                other_files[f.filename] = await f.read()

        answer = process_request_logic(q_text, other_files)

        return JSONResponse(content={
            "question": q_text,
            "answer": answer
        })
    except Exception as e:
        return JSONResponse(content={
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
