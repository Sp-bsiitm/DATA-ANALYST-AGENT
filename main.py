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

def analyze_csv(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    plot_uri = None
    if df.shape[1] >= 2:
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        plt.scatter(x, y, label="Data")
        coeffs = np.polyfit(x, y, 1)
        trend = np.poly1d(coeffs)
        plt.plot(x, trend(x), "r--", label="Trend line")
        plt.legend()
        plot_uri = compress_plot_to_base64()
    return {
        "dataset_rows": len(df),
        "plot": plot_uri
    }

def scrape_wikipedia(topic):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        return r.json().get("extract", "")
    return ""

def indian_high_court_query():
    # Example using DuckDB with HTTPFS parquet
    url = "https://github.com/datablist/sample-csv-files/raw/main/files/people/people-100.csv"
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    df = con.execute(f"SELECT * FROM read_csv_auto('{url}') LIMIT 5").fetchdf()
    return df.to_dict(orient="records")

# ---- Main endpoint ----
@app.post("/")
async def process_request(
    questions: UploadFile = File(...),
    data: UploadFile = File(None)
):
    start_time = time.time()
    try:
        question_text = (await questions.read()).decode("utf-8").strip()

        # Decide task type
        if "scrape wikipedia" in question_text.lower():
            topic = question_text.split("scrape wikipedia", 1)[1].strip()
            answer = scrape_wikipedia(topic)
            return {"question": question_text, "answer": answer}

        elif "analyze csv" in question_text.lower() and data:
            csv_bytes = await data.read()
            result = analyze_csv(csv_bytes)
            return {"question": question_text, **result}

        elif "indian high court" in question_text.lower():
            records = indian_high_court_query()
            return {"question": question_text, "records_sample": records}

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
