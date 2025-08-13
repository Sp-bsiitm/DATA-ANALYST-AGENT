import os
import io
import time
import base64
import asyncio
import requests
from bs4 import BeautifulSoup
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import io, base64
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

MAX_PROCESS_SECONDS = 180  # 3 minutes

app = FastAPI(title="Data Analyst Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def health():
    return {"status": "ok"}

def read_uploaded_file(upload: UploadFile) -> bytes:
    return upload.file.read()

async def run_with_timeout(coro, timeout=MAX_PROCESS_SECONDS):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Processing exceeded {timeout} seconds")



async def process_request_logic(files: dict, questions_txt: str) -> dict:
    question = questions_txt.strip()
    output = {"question": question}

    # Wikipedia scraping
    if question.lower().startswith("scrape wikipedia"):
        try:
            topic = question.split(" ", 2)[-1]
            url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                output["answer"] = f"Failed to fetch Wikipedia page. Status: {r.status_code}"
                return output

            soup = BeautifulSoup(r.text, "html.parser")
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
            if paragraphs:
                output["answer"] = paragraphs[0][:1000]  # first paragraph, limit length
            else:
                output["answer"] = "No content found on the Wikipedia page."
            return output
        except Exception as e:
            output["answer"] = f"Error scraping Wikipedia: {str(e)}"
            return output

    # CSV analysis
    for fname, content in files.items():
        if fname.lower().endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content))
                row_count = len(df)
                output["dataset_rows"] = row_count

                fig, ax = plt.subplots()
                ax.plot(np.arange(row_count), np.random.rand(row_count))
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode('ascii')
                output["plot"] = f"data:image/png;base64,{img_b64}"
                plt.close(fig)
                return output
            except Exception as e:
                output["error"] = f"Error processing CSV: {str(e)}"
                return output

    # Fallback
    output["answer"] = "I don't have logic for that type of question yet."
    return output

@app.post("/", response_class=JSONResponse)
async def handle_post(
    request: Request,
    questions_file: UploadFile = File(..., alias="questions.txt"),
    other_files: List[UploadFile] = File(None)
):
    files_map = {}

    # Read main questions.txt
    questions_content_bytes = await run_in_threadpool(read_uploaded_file, questions_file)
    questions_txt_content = questions_content_bytes.decode("utf-8", errors="ignore")
    files_map[questions_file.filename] = questions_content_bytes

    # Grab all other uploaded files, regardless of their field names
    form = await request.form()
    for field_name, file_obj in form.items():
        if field_name != "questions.txt" and hasattr(file_obj, "filename"):
            content = await run_in_threadpool(read_uploaded_file, file_obj)
            files_map[file_obj.filename] = content

    # Process and return
    result_obj = await run_with_timeout(process_request_logic(files_map, questions_txt_content))
    return JSONResponse(content=result_obj)

