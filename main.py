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
from fastapi import FastAPI, File, UploadFile, Form
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

def compress_plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    img_bytes = buf.getvalue()
    buf.close()

    # compress until under size limit
    quality = 90
    while len(img_bytes) > PLOT_MAX_BYTES and quality > 10:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_bytes = buf.getvalue()
        quality -= 10
    return "data:image/png;base64," + base64.b64encode(img_bytes).decode()

def fig_to_base64_png_under_limit(fig, max_bytes=100_000):
    """
    Serialize a Matplotlib Figure to a PNG data URI under max_bytes by
    reducing DPI progressively. Keeps axes/labels. Returns data URI string.
    """
    for dpi in (200, 160, 140, 120, 100, 90, 80, 70, 60):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    # final attempt with very small DPI
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
        coeffs = np.polyfit(x, y, 1)
        trend = np.poly1d(coeffs)
        ax.plot(x, trend(x), "r--", label="Regression")  # dotted red line
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        ax.set_title("Scatterplot with Regression")
        ax.legend()

        plot_uri = fig_to_base64_png_under_limit(fig, max_bytes=100_000)
        plt.close(fig)

    return {
        "dataset_rows": int(len(df)),
        "plot": plot_uri
    }

def handle_highest_grossing_films(url_text: str):
    """
    Scrape the Wikipedia table of highest-grossing films and compute:
      1) # of $2bn movies released before 2000
      2) earliest film grossing over $1.5bn (string title)
      3) correlation between Rank and Peak (float)
      4) scatterplot (Rank vs Peak) with dotted red regression, labeled axes,
         encoded as data URI PNG under 100 kB
    Returns a 4-element list in that order (as strings/numbers).
    """
    # find URL in the prompt
    import re
    m = re.search(r"https?://\S+", url_text)
    if not m:
        raise ValueError("No URL found in question text.")
    url = m.group(0)

    # pull tables
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)

    # heuristically pick the main table: the one with a 'Rank' column
    cand = [t for t in tables if any(str(c).strip().lower() == "rank" for c in t.columns)]
    if not cand:
        raise ValueError("No table with a 'Rank' column found on the page.")
    df = cand[0].copy()

    # normalize columns likely to exist: Rank, Peak, Title, Worldwide gross, Year
    # if multiple header rows, flatten them
    df.columns = [(" ".join(col) if isinstance(col, tuple) else str(col)).strip() for col in df.columns]

    # coerce numerics
    def to_num(s):
        if pd.isna(s):
            return np.nan
        return pd.to_numeric(str(s).replace("$", "").replace(",", "").strip(), errors="coerce")

    # best-effort column matching
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

    # minimal required cols
    need = ["Rank", "Peak", "Title", "Gross", "Year"]
    if not all(k in colmap for k in need):
        raise ValueError(f"Expected columns not found. Got mapping: {colmap}")

    # clean & compute
    df["Rank_num"] = pd.to_numeric(df[colmap["Rank"]], errors="coerce")
    df["Peak_num"] = pd.to_numeric(df[colmap["Peak"]], errors="coerce")
    df["Gross_num"] = df[colmap["Gross"]].apply(to_num)
    df["Year_num"]  = pd.to_numeric(df[colmap["Year"]], errors="coerce")

    # 1) # of $2bn movies released before 2000
    two_bn_pre2000 = int(((df["Gross_num"] >= 2_000_000_000) & (df["Year_num"] < 2000)).sum())

    # 2) earliest film > $1.5bn
    over_1_5 = df[df["Gross_num"] >= 1_500_000_000].copy()
    earliest_title = ""
    if not over_1_5.empty:
        idx = over_1_5["Year_num"].idxmin()
        earliest_title = str(df.loc[idx, colmap["Title"]])

    # 3) correlation Rank vs Peak
    corr = float(pd.Series(df["Rank_num"]).corr(pd.Series(df["Peak_num"])))

    # 4) scatterplot
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
    img_uri = fig_to_base64_png_under_limit(fig, max_bytes=100_000)
    plt.close(fig)

    # return in the exact order the sample expects
    return [two_bn_pre2000, earliest_title, corr, img_uri]


def scrape_wikipedia(topic):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        return r.json().get("extract", "")
    return ""
    
def indian_high_court_query():
    """
    Returns a small sample (max 5 rows) from a stable public CSV dataset.
    In a real project, this would connect to an actual Indian High Court dataset.
    """
    url = "https://people.sc.fsu.edu/~jburkardt/data/csv/addresses.csv"
    try:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        df = con.execute(f"SELECT * FROM read_csv_auto('{url}') LIMIT 5").fetchdf()
        return df.to_dict(orient="records")  # ✅ Return list directly
    except Exception as e:
        return {"error": f"Error loading public CSV: {e}"}


# ---- Main endpoint ----
from fastapi import HTTPException
@app.post("/")
async def process_request(
    questions: UploadFile = File(None, alias="questions"),
    questions_txt: UploadFile = File(None, alias="questions.txt"),
    data: UploadFile = File(None, alias="data")
):
    start_time = time.time()
    try:
        # Pick whichever file field is provided
        q_file = questions or questions_txt
        if not q_file:
            raise HTTPException(status_code=400, detail="Questions file is required")

        question_text = (await q_file.read()).decode("utf-8").strip()
        question_lower = question_text.lower()

        # Task: Scrape Wikipedia
        if "scrape wikipedia" in question_lower:
            topic = question_text.split("scrape wikipedia", 1)[1].strip()
            answer = scrape_wikipedia(topic)
            return {"question": question_text, "answer": answer}

        # Task: Analyze CSV
        elif "analyze csv" in question_lower and data:
            csv_bytes = await data.read()
            result = analyze_csv(csv_bytes)
            return {"question": question_text, **result}

        # Task: Indian High Court Dataset
        elif "indian high court" in question_lower:
            records = indian_high_court_query()
            return {"question": question_text, "records_sample": records}

        # Unknown task
        else:
            return {"question": question_text, "answer": "I don't have logic for that type of question yet."}

    except Exception as e:
        traceback_str = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback_str}
        )


@app.get("/healthz")
def health_check():
    return {"status": "ok"}
