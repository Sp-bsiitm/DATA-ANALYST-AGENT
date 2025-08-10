# Data Analyst Agent API (example starter)

POST `https://<host>/` with multipart/form-data:
 - `questions.txt` file (REQUIRED)
 - zero or more additional files (data.csv, image.png, etc.)

Example curl:
```bash
curl -X POST "https://your-host/" -F "questions.txt=@question.txt" -F "data.csv=@data.csv"
```

Response: JSON object. Replace `process_request_logic` in `main.py` with real analysis logic (scrape/parquet/plot).

## Deployment

### Option A — Render / Railway / Fly (Docker)

1. Push to GitHub.
2. Create a new Web Service on Render (or similar).
3. Choose Docker or Web Service, connect your repo.
4. Build command: leave empty (Dockerfile used).
5. Start command: Dockerfile CMD runs uvicorn.
6. Set port 8080.

### Option B — Local + ngrok

1. Run:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080
   ```
2. Start ngrok:
   ```bash
   ngrok http 8080
   ```
   Use public URL shown by ngrok.

## Example question.txt

```
Test question
Return a JSON object with keys: foo, timestamp
```

Test locally:
```bash
curl -X POST "http://localhost:8080/" -F "questions.txt=@question.txt"
```

## Production Notes

- App enforces a 3-minute timeout.
- Sanitize inputs in production.
- Limit image output to <100k bytes if needed.
- For DuckDB: avoid loading huge files, push down filters.
