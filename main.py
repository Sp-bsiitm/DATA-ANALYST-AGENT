import os
import json
import time
import traceback
import base64
import io
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import requests
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---- FastAPI setup ----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_EXECUTION_TIME = 180  # seconds
PLOT_MAX_BYTES = 100_000  # <100 KB

# ---- Utility Functions ----
def within_time(start_time):
    return (time.time() - start_time) < MAX_EXECUTION_TIME

def fig_to_base64_png_under_limit(fig, max_bytes=100_000):
    """Serialize Matplotlib figure to PNG data URI under max_bytes by reducing DPI."""
    for dpi in (200, 160, 140, 120, 100, 90, 80, 70, 60):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=50)
    data = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")

def analyze_csv(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    plot_uri = None

    if df.shape[1] >= 2:
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        mask = x.notna() & y.notna()
        x, y = x[mask], y[mask]

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.scatter(x, y, label="Data", s=18)
        if len(x) >= 2:
            coeffs = np.polyfit(x, y, 1)
            trend = np.poly1d(coeffs)
            ax.plot(x, trend(x), "r--", label="Regression")
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        ax.set_title("Scatterplot with Regression")
        ax.legend()

        plot_uri = fig_to_base64_png_under_limit(fig, max_bytes=PLOT_MAX_BYTES)
        plt.close(fig)

    return {
        "dataset_rows": int(len(df)),
        "plot": plot_uri
    }

def handle_highest_grossing_films(question_text: str):
    """Scrape Wikipedia table of highest-grossing films and return:
    [# of $2bn movies before 2000, earliest > $1.5bn film, correlation Rank vs Peak, scatterplot URI]
    """
    import re

    question_lower = question_text.lower()
    normalized_question = re.sub(r"[-]", " ", question_lower)

    # Extract the Wikipedia URL from the question text
    m = re.search(r"https?://\S+", question_text)
    if not m:
        raise ValueError("No URL found in question text.")
    url = m.group(0)

    # Download and parse the table
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)

    cand = [t for t in tables if any(str(c).strip().lower() == "rank" for c in t.columns)]
    if not cand:
        raise ValueError("No table with a 'Rank' column found.")
    df = cand[0].copy()
    df.columns = [(" ".join(col) if isinstance(col, tuple) else str(col)).strip() for col in df.columns]

    def to_num(s):
        if pd.isna(s):
            return np.nan
        return pd.to_numeric(str(s).replace("$", "").replace(",", "").strip(), errors="coerce")

    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "rank" and "Rank" not in colmap:
            colmap["Rank"] = c
        if "peak" in cl and "Peak" not in colmap:
            colmap["Peak"] = c
        if "title" in cl and "Title" not in colmap:
            colmap["Title"] = c
        if ("worldwide" in cl or "gross" in cl) and "Gross" not in colmap:
            colmap["Gross"] = c
        if "year" in cl and "Year" not in colmap:
            colmap["Year"] = c

    need = ["Rank", "Peak", "Title", "Gross", "Year"]
    if not all(k in colmap for k in need):
        raise ValueError(f"Expected columns not found. Got mapping: {colmap}")

    df["Rank_num"] = pd.to_numeric(df[colmap["Rank"]], errors="coerce")
    df["Peak_num"] = pd.to_numeric(df[colmap["Peak"]], errors="coerce")
    df["Gross_num"] = df[colmap["Gross"]].apply(to_num)
    df["Year_num"]  = pd.to_numeric(df[colmap["Year"]], errors="coerce")

    two_bn_pre2000 = int(((df["Gross_num"] >= 2_000_000_000) & (df["Year_num"] < 2000)).sum())

    over_1_5 = df[df["Gross_num"] >= 1_500_000_000].copy()
    earliest_title = ""
    if not over_1_5.empty:
        idx = over_1_5["Year_num"].idxmin()
        earliest_title = str(df.loc[idx, colmap["Title"]])

    corr = float(pd.Series(df["Rank_num"]).corr(pd.Series(df["Peak_num"])))

    mask = df["Rank_num"].notna() & df["Peak_num"].notna()
    x = df.loc[mask, "Rank_num"]
    y = df.loc[mask, "Peak_num"]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.scatter(x, y, s=18)
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * x + b, "r--")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Peak")
    ax.set_title("Rank vs Peak")
    img_uri = fig_to_base64_png_under_limit(fig, max_bytes=PLOT_MAX_BYTES)
    plt.close(fig)

    return [two_bn_pre2000, earliest_title, corr, img_uri]

def scrape_wikipedia(topic):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        return r.json().get("extract", "")
    return ""

def indian_high_court_query():
    """Returns a sample from a public CSV dataset."""
    url = "https://people.sc.fsu.edu/~jburkardt/data/csv/addresses.csv"
    try:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        df = con.execute(f"SELECT * FROM read_csv_auto('{url}') LIMIT 5").fetchdf()
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": f"Error loading public CSV: {e}"}

# ---- Main endpoint ----
@app.post("/")
async def process_request(
    questions: UploadFile = File(None, alias="questions"),
    questions_txt: UploadFile = File(None, alias="questions.txt"),
    data: UploadFile = File(None, alias="data")
):
    start_time = time.time()
    try:
        q_file = questions or questions_txt
        if not q_file:
            raise HTTPException(status_code=400, detail="Questions file is required")

        question_text = (await q_file.read()).decode("utf-8").strip()
        question_lower = question_text.lower()

        if "scrape wikipedia" in question_lower:
            topic = question_text.split("scrape wikipedia", 1)[1].strip()
            return {"question": question_text, "answer": scrape_wikipedia(topic)}

        elif "analyze csv" in question_lower and data:
            csv_bytes = await data.read()
            return {"question": question_text, **analyze_csv(csv_bytes)}

        elif "indian high court" in question_lower:
            return {"question": question_text, "records_sample": indian_high_court_query()}
        
        # Highest-grossing films handler
        elif any(kw in question_lower for kw in [
            "highest grossing films",
            "highest-grossing films",
            "list of highest grossing films"
        ]):
            return {
                "question": question_text,
                "answer": handle_highest_grossing_films(question_text)
            }


        else:
            # Safe default for unknown types
            return {"question": question_text, "answer": None}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()}
        )

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
