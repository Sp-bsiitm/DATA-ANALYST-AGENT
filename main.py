import os
import io
import time
import base64
import asyncio
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

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
    """
    PLACEHOLDER: implement the real logic here.

    files: dict mapping filename -> bytes
    questions_txt: the string contents of questions.txt

    Expected to return a Python object that can be JSON serialized:
      - e.g. list or dict depending on the requested output format.

    Suggestions for real implementation:
      - If scraping: use requests + beautifulsoup4
      - If heavy parquet/DuckDB: use duckdb, pyarrow, pandas
      - For plotting: matplotlib, save to PNG, base64 encode -> "data:image/png;base64,..."
      - Keep outputs base64 image <100k bytes if tests require size constraint.
    """
    # ---- simple example stub logic ----
    # If questions_txt contains specific keywords you can branch behaviour.
    out = {
        "received_files": list(files.keys()),
        "questions_preview": questions_txt[:100],
        "timestamp": time.time(),
        "note": "This is a stub response. Replace process_request_logic with your real analysis."
    }

    # Example: produce a small example plot encoded as a PNG data URI (demo only)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        x = np.linspace(0, 2 * np.pi, 30)
        y = np.sin(x)
        ax.plot(x, y, marker='o')
        ax.set_title("Demo plot")
        ax.set_xlabel("x")
        ax.set_ylabel("sin(x)")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('ascii')
        out["demo_plot"] = f"data:image/png;base64,{img_b64}"
    except Exception as e:
        out["demo_plot_error"] = str(e)

    return out

@app.post("/", response_class=JSONResponse)
async def handle_post(request: Request, files: List[UploadFile] = File(None)):
    """
    Accept a multipart/form-data POST:
      - questions.txt must be included (server will search for an uploaded file named questions.txt)
      - additional files optional
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded. 'questions.txt' is required.")

    # find questions.txt
    questions_txt_content = None
    files_map = {}
    for upload in files:
        filename = upload.filename or ""
        content = await run_in_threadpool(read_uploaded_file, upload)
        files_map[filename] = content
        if filename.lower() == "questions.txt" or filename.lower().endswith("/questions.txt"):
            questions_txt_content = content.decode("utf-8", errors="ignore")

    if not questions_txt_content:
        # try to parse raw body (edge-case clients)
        raise HTTPException(status_code=400, detail="Missing 'questions.txt' file in upload.")

    # Run user analysis logic with timeout
    result_obj = await run_with_timeout(process_request_logic(files_map, questions_txt_content))
    # Finalize: ensure JSON serializable (caller should ensure that during implementation)
    return JSONResponse(content=result_obj)
