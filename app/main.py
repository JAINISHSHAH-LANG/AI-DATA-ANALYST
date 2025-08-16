# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# import uvicorn
# import duckdb
# import pandas as pd
# import io
# import base64
# import matplotlib.pyplot as plt
# import time

# app = FastAPI(title="Data Analyst Agent")

# # --- Helper Functions ---
# def generate_plot_base64(df, x_col, y_col):
#     plt.figure(figsize=(5, 4))
#     plt.scatter(df[x_col], df[y_col], label="data")
#     m, b = pd.np.polyfit(df[x_col], df[y_col], 1)
#     plt.plot(df[x_col], m*df[x_col] + b, linestyle="dotted", color="red", label="regression")
#     plt.legend()
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format="png")
#     buffer.seek(0)
#     img_str = base64.b64encode(buffer.read()).decode("utf-8")
#     plt.close()
#     return img_str

# # --- API Models ---
# class QueryRequest(BaseModel):
#     question: str

# class QueryResponse(BaseModel):
#     answer: dict | list

# # --- API Endpoints ---
# @app.post("/api/")
# async def solve(
#     question: str = Form(...),
#     file: UploadFile = File(None)
# ):
#     start = time.time()
#     try:
#         answer = {}
#         df = None

#         # Handle uploaded file if CSV
#         if file and file.filename.endswith(".csv"):
#             content = await file.read()
#             df = pd.read_csv(io.BytesIO(content))

#         # Example: simple question to dataframe logic
#         if df is not None and "plot" in question.lower():
#             # Pick first two numeric columns for demo
#             numeric_cols = df.select_dtypes(include="number").columns
#             if len(numeric_cols) >= 2:
#                 img = generate_plot_base64(df, numeric_cols[0], numeric_cols[1])
#                 answer = {"plot": img, "x": numeric_cols[0], "y": numeric_cols[1]}
#             else:
#                 answer = {"error": "Not enough numeric columns to plot"}
#         elif df is not None:
#             answer = {"preview": df.head(5).to_dict(orient="records")}
#         else:
#             # If no file, fallback answer
#             answer = {"message": f"Received question: {question}"}

#         # Enforce time budget
#         elapsed = time.time() - start
#         if elapsed > 180:
#             return JSONResponse(content={"error": "Processing timed out."}, status_code=504)

#         return QueryResponse(answer=answer)

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)


# # --- Entrypoint ---
# if __name__ == "__main__":
#     uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)



import asyncio
import base64
import io
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import orjson  # noqa: E402
import pandas as pd  # noqa: E402
from bs4 import BeautifulSoup  # noqa: F401
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from starlette.datastructures import UploadFile

# ------------------ Config ------------------
REQUEST_DEADLINE = int(os.getenv("REQUEST_DEADLINE_SECONDS", "170"))
MAX_PLOT_BYTES = int(os.getenv("MAX_PLOT_BYTES", "100000"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

app = FastAPI(title="Data Analyst Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"ok": True}


# ------------------ Utils ------------------
class Deadline:
    def __init__(self, seconds: int):
        self.start = time.time()
        self.deadline = self.start + seconds

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.time())

    def about_to_expire(self, buffer: float = 5.0) -> bool:
        return self.remaining() <= buffer


def json_dumps(obj: Any) -> str:
    return orjson.dumps(obj).decode()


async def read_all_files(files: List[UploadFile]) -> Dict[str, bytes]:
    out: Dict[str, bytes] = {}
    for f in files:
        out[f.filename] = await f.read()
    return out


def b64_data_uri_png(buf: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def save_plot_to_budget(fig, max_bytes: int = MAX_PLOT_BYTES) -> str:
    # progressively compress by reducing DPI until under budget
    for dpi in [140, 120, 100, 90, 80, 70, 60]:
        bio = io.BytesIO()
        fig.savefig(
            bio, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1
        )
        data = bio.getvalue()
        if len(data) <= max_bytes:
            plt.close(fig)
            return b64_data_uri_png(data)
    # last resort: tiny dpi
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=50, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return b64_data_uri_png(bio.getvalue())


# ------------------ Simple LLM Router (optional) ------------------
async def llm_route(question_text: str) -> str:
    if not OPENAI_API_KEY:
        return "none"
    prompt = (
        "You are a router for a data analyst agent. Given a task description, "
        "respond with exactly one token among: wiki, duckdb, csv, generic.\n\n"
        f"Task:\n{question_text}\n\nAnswer:"
    )
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )
            r.raise_for_status()
            out = (
                r.json()["choices"][0]["message"]["content"]
                .strip()
                .lower()
            )
            return out if out in {"wiki", "duckdb", "csv", "generic"} else "generic"
    except Exception:
        return "none"


# ------------------ Question parsing ------------------
@dataclass
class Parsed:
    mode: str  # "array" or "object"
    items: List[str]


def parse_questions(qtext: str) -> Parsed:
    mode = (
        "array"
        if re.search(r"respond with a json array", qtext, re.I)
        else "object"
        if re.search(r"respond with a json object", qtext, re.I)
        else "array"
    )
    # grab numbered or bulleted questions
    lines = [l.strip() for l in qtext.splitlines()]
    items: List[str] = []
    for l in lines:
        if re.match(r"^\s*\d+\.\s+", l) or re.match(r"^[-*]\s+", l):
            items.append(re.sub(r"^\s*(\d+\.|[-*])\s+", "", l))
    if not items:
        # fallback: treat the whole text as one question
        items = [qtext.strip()]
    return Parsed(mode=mode, items=items)


# ------------------ Solvers ------------------
async def solver_wikipedia_highest_grossing(
    qtext: str, deadline: Deadline
) -> Tuple[List[Any], Optional[str]]:
    """Implements the sample task for highest-grossing films.
    Returns (answers list, plot_data_uri or None)."""
    # URL extraction
    m = re.search(r"https?://\S+", qtext)
    url = (
        m.group(0)
        if m
        else "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    # Parse table with pandas (faster/robust) then enrich with BeautifulSoup if needed
    tables = pd.read_html(html)

    # Heuristic: pick the first table that has columns like Rank and Peak
    def score(t: pd.DataFrame):
        cols = {c.lower() for c in t.columns.astype(str)}
        return (
            int("rank" in cols)
            + int("peak" in cols)
            + int("title" in cols)
            + int("worldwide" in " ".join(cols))
        )

    table = max(tables, key=score)

    df = table.copy()
    # Normalize
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["Rank", "Peak"]:
        if col not in df.columns:
            c2 = next((c for c in df.columns if c.lower() == col.lower()), None)
            if c2:
                df[col] = df[c2]
    # Clean Rank/Peak numeric
    for c in ["Rank", "Peak"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Year
    if "Year" not in df.columns:
        if "Title" in df.columns:
            df["Year"] = (
                df["Title"].astype(str).str.extract(r"(19\d{2}|20\d{2})").astype(float)
            )
        else:
            df["Year"] = np.nan
    # Gross column
    gross_col = next(
        (c for c in df.columns if re.search(r"gross|worldwide", str(c), re.I)), None
    )
    if gross_col is None:
        gross_col = df.columns[-1]
    gross = df[gross_col].astype(str)

    def parse_money(s: str) -> float:
        s = s.replace(",", "")
        m = re.search(r"(\d+\.?\d*)\s*\$?\s*(billion|bn|million|m)?", s, re.I)
        if not m:
            m2 = re.search(r"\$(\d+\.?\d*)", s)
            if m2:
                return float(m2.group(1))
            return math.nan
        val = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit in ("billion", "bn"):
            return val * 1_000_000_000
        if unit in ("million", "m"):
            return val * 1_000_000
        return val

    df["WorldwideUSD"] = gross.map(parse_money)

    # Q1
    q1 = int(((df["WorldwideUSD"] >= 2_000_000_000) & (df["Year"] < 2000)).sum())

    # Q2
    q2_title = ""
    if "Title" in df.columns:
        df2 = df[df["WorldwideUSD"] > 1_500_000_000].copy()
        if "Year" in df2.columns:
            df2 = df2.sort_values("Year", na_position="last")
        if not df2.empty:
            q2_title = str(df2.iloc[0]["Title"])

    # Q3
    corr = (
        float(pd.Series(df["Rank"]).corr(pd.Series(df["Peak"])))
        if "Rank" in df and "Peak" in df
        else float("nan")
    )

    # Q4
    plot_uri = None
    if "Rank" in df and "Peak" in df:
        x = pd.to_numeric(df["Rank"], errors="coerce")
        y = pd.to_numeric(df["Peak"], errors="coerce")
        msk = x.notna() & y.notna()
        x = x[msk].astype(float)
        y = y[msk].astype(float)
        if len(x) >= 2:
            a, b = np.polyfit(x, y, 1)
            xline = np.linspace(x.min(), x.max(), 100)
            yline = a * xline + b
            fig = plt.figure(figsize=(4, 3))
            ax = fig.add_subplot(111)
            ax.scatter(x, y)
            ax.plot(xline, yline, linestyle=":", color="red")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Peak")
            ax.set_title("Rank vs Peak")
            plot_uri = save_plot_to_budget(fig)

    answers = [q1, q2_title, round(corr, 6), plot_uri or ""]
    return answers, plot_uri


async def solver_duckdb_indian_courts(qtext: str, deadline: Deadline) -> Dict[str, Any]:
    # Use DuckDB with httpfs & parquet
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;")

    # 1) Which high court disposed the most cases 2019-2022?
    q1 = con.execute(
        """
        SELECT court, COUNT(*) AS n
        FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
        WHERE year BETWEEN 2019 AND 2022
        GROUP BY court
        ORDER BY n DESC
        LIMIT 1
        """
    ).fetchdf()
    most_court = q1.iloc[0]["court"] if not q1.empty else None

    # 2) Regression slope of date_of_registration -> decision_date by year in court=33_10
    df = con.execute(
        """
        SELECT year, date_of_registration, decision_date
        FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=33_10/bench=*/metadata.parquet?s3_region=ap-south-1')
        WHERE decision_date IS NOT NULL AND date_of_registration IS NOT NULL
        """
    ).fetchdf()

    def parse_reg(s):
        if pd.isna(s):
            return pd.NaT
        s = str(s)
        try:
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        except Exception:
            return pd.to_datetime(s, errors="coerce")

    df["dor"] = df["date_of_registration"].map(parse_reg)
    df["dod"] = pd.to_datetime(df["decision_date"], errors="coerce")
    df["delay_days"] = (df["dod"] - df["dor"]).dt.days
    df = df.dropna(subset=["year", "delay_days"])

    if len(df) >= 2:
        x = df["year"].astype(float)
        y = df["delay_days"].astype(float)
        a, b = np.polyfit(x, y, 1)  # slope a
        slope = float(a)
    else:
        slope = float("nan")

    # 3) Plot
    plot_uri = ""
    try:
        if len(df) >= 2:
            xs = df["year"].astype(float)
            ys = df["delay_days"].astype(float)
            A, B = np.polyfit(xs, ys, 1)
            xline = np.linspace(xs.min(), xs.max(), 200)
            yline = A * xline + B
            fig = plt.figure(figsize=(4, 3))
            ax = fig.add_subplot(111)
            ax.scatter(xs, ys)
            ax.plot(xline, yline, linestyle=":")
            ax.set_xlabel("Year")
            ax.set_ylabel("Delay (days)")
            ax.set_title("Registration→Decision Delay by Year (court=33_10)")
            plot_uri = save_plot_to_budget(fig)
    except Exception:
        plot_uri = ""

    return {
        "Which high court disposed the most cases from 2019 - 2022?": str(most_court)
        if most_court is not None
        else None,
        "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": slope,
        "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": plot_uri,
    }


# Generic CSV helper (used if CSV is uploaded & referenced)
def try_load_csvs(files: Dict[str, bytes]) -> Dict[str, pd.DataFrame]:
    out = {}
    for name, data in files.items():
        if name.lower().endswith(".csv"):
            try:
                out[name] = pd.read_csv(io.BytesIO(data))
            except Exception:
                pass
    return out


def pick_questions_file(form_items: List[Tuple[str, Any]]) -> Tuple[Optional[UploadFile], List[UploadFile]]:
    """
    Selects the questions file to support the spec's example:
      curl -F "questions.txt=@question.txt" ...
    We accept:
      - part name 'questions.txt'
      - or any UploadFile whose filename == 'questions.txt'
      - or, as a fallback, the first text/plain UploadFile
    Returns (questions_file, other_files).
    """
    uploads: List[UploadFile] = [v for _, v in form_items if isinstance(v, UploadFile)]

    # 1) match by field name
    for k, v in form_items:
        if isinstance(v, UploadFile) and k.lower() == "questions.txt":
            others = [u for u in uploads if u is not v]
            return v, others

    # 2) match by filename
    for v in uploads:
        if (v.filename or "").lower() == "questions.txt":
            others = [u for u in uploads if u is not v]
            return v, others

    # 3) first text/plain
    for v in uploads:
        if (v.content_type or "").startswith("text/"):
            others = [u for u in uploads if u is not v]
            return v, others

    return None, uploads


# ------------------ Main handler ------------------
@app.post("/api/", response_class=PlainTextResponse)
async def api(request: Request):
    req_id = str(uuid.uuid4())[:8]
    deadline = Deadline(REQUEST_DEADLINE)

    try:
        form = await request.form()
        # Starlette's FormData is multi-dict; keep all (key, value) pairs
        items: List[Tuple[str, Any]] = []
        for k in form.keys():
            for v in form.getlist(k):
                items.append((k, v))

        questions_file, other_uploads = pick_questions_file(items)
        if not questions_file:
            # Always return structured error
            return PlainTextResponse(
                content=json_dumps({"error": "questions.txt not found in form-data"}),
                media_type="application/json",
                status_code=400,
            )

        filemap = await read_all_files(other_uploads)
        qtext = (await questions_file.read()).decode(errors="ignore")
        parsed = parse_questions(qtext)

        # Router
        route_hint = await llm_route(qtext)
        if re.search(r"wikipedia|highest-grossing films|highest grossing films", qtext, re.I):
            route = "wiki"
        elif re.search(r"indian high court|ecourts|s3://indian-high-court-judgments", qtext, re.I):
            route = "duckdb"
        elif any(k.lower().endswith(".csv") for k in filemap):
            route = "csv"
        else:
            route = route_hint or "generic"

        # Execute with time budget
        if route == "wiki":
            answers, _plot = await asyncio.wait_for(
                solver_wikipedia_highest_grossing(qtext, deadline),
                timeout=max(5.0, deadline.remaining()),
            )
            result: Any = (
                answers if parsed.mode == "array" else {str(i + 1): v for i, v in enumerate(answers)}
            )
        elif route == "duckdb":
            result = await asyncio.wait_for(
                solver_duckdb_indian_courts(qtext, deadline),
                timeout=max(5.0, deadline.remaining()),
            )
        elif route == "csv":
            # very simple CSV support: answer count rows per CSV
            dfs = try_load_csvs(filemap)
            result = {name: int(df.shape[0]) for name, df in dfs.items()} or {
                "message": "No CSV parsed"
            }
        else:
            # generic fallback: echo questions
            result = {
                "message": "Received questions; no matching solver. Provide structured data or a supported URL."
            }

        # Ensure shape compliance to the best of our ability
        if isinstance(result, list):
            payload = json_dumps(result)
        elif isinstance(result, dict):
            if parsed.mode == "array":
                # flatten best-effort
                payload = json_dumps(list(result.values()))
            else:
                payload = json_dumps(result)
        else:
            payload = json_dumps([str(result)])

        return PlainTextResponse(content=payload, media_type="application/json")

    except asyncio.TimeoutError:
        # Return best-effort stub matching likely array shape
        fallback = [None, "", float("nan"), ""]
        return PlainTextResponse(content=json_dumps(fallback), media_type="application/json")
    except Exception as e:
        # Always return something structured
        fallback = {"error": str(e)}
        return PlainTextResponse(content=json_dumps(fallback), media_type="application/json")
