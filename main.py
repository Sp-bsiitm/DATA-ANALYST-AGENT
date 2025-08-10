import os
import io
import time
import base64
import asyncio
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
    out = {
        "received_files": list(files.keys()),
        "questions_preview": questions_txt[:100],
        "timestamp": time.time(),
        "note": "This is a stub response. Replace process_request_logic with your real analysis."
    }

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

    # Add any other uploaded files
    if other_files:
        for upload in other_files:
            content = await run_in_threadpool(read_uploaded_file, upload)
            files_map[upload.filename] = content

    # Process and return
    result_obj = await run_with_timeout(process_request_logic(files_map, questions_txt_content))
    return JSONResponse(content=result_obj)
