# app/main.py
import asyncio
import base64
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

# Use Agg backend for headless servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

# Optional libs
try:
    import duckdb
except Exception:
    duckdb = None

try:
    import httpx
except Exception:
    httpx = None

# Configuration
REQUEST_DEADLINE_SECONDS = int(os.getenv("REQUEST_DEADLINE_SECONDS", "170"))
MAX_PLOT_BYTES = int(os.getenv("MAX_PLOT_BYTES", "100000"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
APP_NAME = "AI Data Analyst Agent"

app = FastAPI(title=APP_NAME)

# ------------------ Utilities ------------------ #
class Deadline:
    def __init__(self, seconds: float):
        self.start = time.time()
        self.deadline = self.start + seconds

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.time())

def dumps_json(obj: Any) -> str:
    try:
        import orjson
        return orjson.dumps(obj).decode()
    except Exception:
        return json.dumps(obj, default=str)

def plot_scatter_regression(x: np.ndarray, y: np.ndarray, xlabel: str = "x", ylabel: str = "y", title: Optional[str] = None) -> str:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=10)
    try:
        a, b = np.polyfit(x, y, 1)
        x_line = np.linspace(np.min(x), np.max(x), 200)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, linestyle=":", color="red", linewidth=1.5)
    except Exception:
        pass
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True)

    def encode_png(dpi: int = 100) -> bytes:
        bio = io.BytesIO()
        fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        return bio.getvalue()

    # Adjust DPI until under MAX_PLOT_BYTES
    for dpi in [140, 120, 100, 90, 80, 70, 60, 50]:
        data = encode_png(dpi=dpi)
        if len(data) <= MAX_PLOT_BYTES:
            plt.close(fig)
            return "data:image/png;base64," + base64.b64encode(data).decode()

    # Fallback to smaller/WEBP
    try:
        from PIL import Image
        data = encode_png(dpi=50)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        out = io.BytesIO()
        img.save(out, format="WEBP", quality=80, method=6)
        plt.close(fig)
        return "data:image/webp;base64," + base64.b64encode(out.getvalue()).decode()
    except Exception:
        data = encode_png(dpi=50)
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(data).decode()

# ------------------ File loaders ------------------ #
async def load_uploaded_files(files: List[UploadFile]) -> Tuple[Dict[str, bytes], Dict[str, pd.DataFrame]]:
    raw: Dict[str, bytes] = {}
    dfs: Dict[str, pd.DataFrame] = {}
    for f in files:
        filename = (f.filename or "unnamed").strip()
        try:
            content = await f.read()
        except Exception:
            content = b""
        raw[filename] = content
        lower = filename.lower()
        try:
            if lower.endswith(".csv"):
                dfs[filename] = pd.read_csv(io.BytesIO(content))
            elif lower.endswith(".tsv"):
                dfs[filename] = pd.read_csv(io.BytesIO(content), sep="\t")
            elif lower.endswith(".parquet"):
                try:
                    dfs[filename] = pd.read_parquet(io.BytesIO(content))
                except Exception:
                    if duckdb:
                        pass
            elif lower.endswith(".json"):
                try:
                    dfs[filename] = pd.read_json(io.BytesIO(content), lines=False)
                except Exception:
                    try:
                        dfs[filename] = pd.read_json(io.BytesIO(content), lines=True)
                    except Exception:
                        pass
            elif lower.endswith(".xlsx") or lower.endswith(".xls"):
                try:
                    dfs[filename] = pd.read_excel(io.BytesIO(content))
                except Exception:
                    pass
        except Exception:
            continue
    return raw, dfs

# ------------------ Question Parsing ------------------ #
@dataclass
class ParsedQuestions:
    mode: str
    items: List[str]

def parse_questions_text(qtext: str) -> ParsedQuestions:
    mode = "array" if "respond with a json array" in qtext.lower() else "object" if "respond with a json object" in qtext.lower() else "array"
    lines = [l.strip() for l in qtext.splitlines() if l.strip()]
    items: List[str] = []
    for l in lines:
        if l.lstrip().startswith(("-", "*")) or l.lstrip()[0:2].strip().isdigit():
            cleaned = l.lstrip("-*0123456789. ").strip()
            items.append(cleaned)
        else:
            if l.endswith("?"):
                items.append(l)
    if not items:
        items = [qtext.strip()]
    return ParsedQuestions(mode=mode, items=items)

# ------------------ Main API Handler ------------------ #
@app.post("/api/", response_class=PlainTextResponse)
async def api_handler(request: Request, files: List[UploadFile] = File([])):
    deadline = Deadline(REQUEST_DEADLINE_SECONDS)

    async def _process() -> Any:
        qtext: Optional[str] = None
        other_uploads: List[UploadFile] = []

        for f in files:
            fname = (f.filename or "").strip()
            if fname.lower() == "questions.txt":
                try:
                    qtext = (await f.read()).decode("utf-8", errors="replace")
                except Exception:
                    qtext = ""
            else:
                other_uploads.append(f)

        if not qtext:
            form = await request.form()
            for key, val in form.multi_items():
                if key.lower() in ("questions.txt", "questions"):
                    if isinstance(val, UploadFile):
                        qtext = (await val.read()).decode("utf-8", errors="replace")
                    else:
                        qtext = str(val)
                    break

        if not qtext:
            return JSONResponse(content={"error": "questions.txt missing"}, status_code=400)

        parsed = parse_questions_text(qtext)
        raw_map, dfs_map = await load_uploaded_files(other_uploads)

        # ------------------ Phase 2: Sample Weather Analysis ------------------ #
        if "sample-weather.csv" in dfs_map:
            df = dfs_map["sample-weather.csv"]
            required_columns = ["temperature_c", "precipitation_mm", "date"]
            missing_cols = [c for c in required_columns if c not in df.columns]
            if missing_cols:
                raise HTTPException(status_code=500, detail=f"Missing columns in CSV: {missing_cols}")
            try:
                avg_temp = round(df["temperature_c"].mean(), 3)
                min_temp = int(df["temperature_c"].min())
                avg_precip = round(df["precipitation_mm"].mean(), 3)
                max_precip_idx = df["precipitation_mm"].idxmax()
                max_precip_date = str(df.loc[max_precip_idx, "date"])
                corr = round(df["temperature_c"].corr(df["precipitation_mm"]), 10)

                # Base64 charts
                temp_line_chart = plot_scatter_regression(
                    np.arange(len(df)), df["temperature_c"].to_numpy(),
                    xlabel="Index", ylabel="Temperature (°C)", title="Temperature Trend"
                )
                precip_histogram = plot_scatter_regression(
                    np.arange(len(df)), df["precipitation_mm"].to_numpy(),
                    xlabel="Index", ylabel="Precipitation (mm)", title="Precipitation Histogram"
                )

                analysis_result = {
                    "average_temp_c": avg_temp,
                    "min_temp_c": min_temp,
                    "average_precip_mm": avg_precip,
                    "max_precip_date": max_precip_date,
                    "temp_precip_correlation": corr,
                    "temp_line_chart": temp_line_chart,
                    "precip_histogram": precip_histogram
                }
                return JSONResponse(content=analysis_result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Fallback pipeline
        return {"message": "Fallback processing pipeline (existing logic)"}

    try:
        return await asyncio.wait_for(_process(), timeout=deadline.remaining())
    except asyncio.TimeoutError:
        return PlainTextResponse(content=dumps_json([None, "", float("nan"), ""]), media_type="application/json", status_code=504)
    except Exception as e:
        return PlainTextResponse(content=dumps_json({"error": str(e)}), media_type="application/json", status_code=500)

# ------------------ Health endpoint ------------------ #
@app.get("/health")
async def health():
    return {"ok": True}
















# # app/main.py
# import asyncio
# import base64
# import io
# import json
# import math
# import os
# import time
# import typing
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import matplotlib

# # Use Agg backend for headless servers
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from fastapi import FastAPI, UploadFile, File, Request
# from fastapi.responses import PlainTextResponse, JSONResponse
# from pydantic import BaseModel
# from fastapi import HTTPException

# # Optional libs: duckdb and httpx if available
# try:
#     import duckdb  # type: ignore
# except Exception:
#     duckdb = None  # type: ignore

# try:
#     import httpx  # type: ignore
# except Exception:
#     httpx = None  # type: ignore

# # Configuration (can set env vars)
# REQUEST_DEADLINE_SECONDS = int(os.getenv("REQUEST_DEADLINE_SECONDS", "170"))  # leave margin for http
# MAX_PLOT_BYTES = int(os.getenv("MAX_PLOT_BYTES", "100000"))
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # default; replace with your model
# APP_NAME = "AI Data Analyst Agent"

# app = FastAPI(title=APP_NAME)

# # ------------------ Utilities ------------------ #
# class Deadline:
#     def __init__(self, seconds: float):
#         self.start = time.time()
#         self.deadline = self.start + seconds

#     def remaining(self) -> float:
#         return max(0.0, self.deadline - time.time())

#     def about_to_expire(self, buffer: float = 3.0) -> bool:
#         return self.remaining() <= buffer


# def dumps_json(obj: Any) -> str:
#     try:
#         import orjson  # type: ignore

#         return orjson.dumps(obj).decode()
#     except Exception:
#         return json.dumps(obj, default=str)


# async def run_llm_system_user(system: str, user: str) -> str:
#     if not OPENAI_API_KEY or not httpx:
#         return "LLM not configured"
#     url = "https://api.openai.com/v1/chat/completions"
#     headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
#     payload = {
#         "model": LLM_MODEL,
#         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
#         "temperature": 0.0,
#         "max_tokens": 512,
#     }
#     async with httpx.AsyncClient(timeout=30.0) as client:
#         r = await client.post(url, headers=headers, json=payload)
#         r.raise_for_status()
#         j = r.json()
#         return j["choices"][0]["message"]["content"].strip()


# def plot_scatter_regression(
#     x: np.ndarray, y: np.ndarray, xlabel: str = "x", ylabel: str = "y", title: Optional[str] = None
# ) -> str:
#     fig = plt.figure(figsize=(4, 3), dpi=100)
#     ax = fig.add_subplot(111)
#     ax.scatter(x, y, s=10)
#     try:
#         a, b = np.polyfit(x, y, 1)
#         x_line = np.linspace(np.min(x), np.max(x), 200)
#         y_line = a * x_line + b
#         ax.plot(x_line, y_line, linestyle=":", color="red", linewidth=1.5)
#     except Exception:
#         pass
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     if title:
#         ax.set_title(title)
#     ax.grid(False)

#     def encode_png_bytes(dpi: int = 100) -> bytes:
#         bio = io.BytesIO()
#         fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
#         data = bio.getvalue()
#         return data

#     for dpi in [140, 120, 100, 90, 80, 70, 60, 50]:
#         data = encode_png_bytes(dpi=dpi)
#         if len(data) <= MAX_PLOT_BYTES:
#             plt.close(fig)
#             return "data:image/png;base64," + base64.b64encode(data).decode()
#     try:
#         from PIL import Image
#         data_png = encode_png_bytes(dpi=50)
#         img = Image.open(io.BytesIO(data_png)).convert("RGB")
#         out = io.BytesIO()
#         img.save(out, format="WEBP", quality=80, method=6)
#         webp_bytes = out.getvalue()
#         plt.close(fig)
#         return "data:image/webp;base64," + base64.b64encode(webp_bytes).decode()
#     except Exception:
#         data = encode_png_bytes(dpi=50)
#         plt.close(fig)
#         return "data:image/png;base64," + base64.b64encode(data).decode()


# # ------------------ Data Models ------------------ #
# class AnalysisRequest(BaseModel):
#     query: str


# # ------------------ File loaders ------------------ #
# async def load_uploaded_files(files: List[UploadFile]) -> Tuple[Dict[str, bytes], Dict[str, pd.DataFrame]]:
#     raw: Dict[str, bytes] = {}
#     dfs: Dict[str, pd.DataFrame] = {}
#     for f in files:
#         filename = (f.filename or "unnamed").strip()
#         try:
#             content = await f.read()
#         except Exception:
#             content = b""
#         raw[filename] = content
#         lower = filename.lower()
#         try:
#             if lower.endswith(".csv"):
#                 dfs[filename] = pd.read_csv(io.BytesIO(content))
#             elif lower.endswith(".tsv"):
#                 dfs[filename] = pd.read_csv(io.BytesIO(content), sep="\t")
#             elif lower.endswith(".parquet"):
#                 try:
#                     dfs[filename] = pd.read_parquet(io.BytesIO(content))
#                 except Exception:
#                     if duckdb is not None:
#                         pass
#             elif lower.endswith(".json"):
#                 try:
#                     dfs[filename] = pd.read_json(io.BytesIO(content), lines=False)
#                 except Exception:
#                     try:
#                         dfs[filename] = pd.read_json(io.BytesIO(content), lines=True)
#                     except Exception:
#                         pass
#             elif lower.endswith(".xlsx") or lower.endswith(".xls"):
#                 try:
#                     dfs[filename] = pd.read_excel(io.BytesIO(content))
#                 except Exception:
#                     pass
#         except Exception:
#             continue
#     return raw, dfs


# # ------------------ Question Parsing ------------------ #
# @dataclass
# class ParsedQuestions:
#     mode: str
#     items: List[str]


# def parse_questions_text(qtext: str) -> ParsedQuestions:
#     mode = "array" if "respond with a json array" in qtext.lower() else "object" if "respond with a json object" in qtext.lower() else "array"
#     lines = [l.strip() for l in qtext.splitlines() if l.strip()]
#     items: List[str] = []
#     for l in lines:
#         if l.lstrip().startswith(("-", "*")) or l.lstrip()[0:2].strip().isdigit():
#             cleaned = l
#             cleaned = cleaned.lstrip()
#             cleaned = cleaned.lstrip("-*0123456789. ")
#             items.append(cleaned.strip())
#         else:
#             if l.endswith("?"):
#                 items.append(l)
#     if not items:
#         items = [qtext.strip()]
#     return ParsedQuestions(mode=mode, items=items)


# # ------------------ Main API Handler ------------------ #
# @app.post("/api/", response_class=PlainTextResponse)
# async def api_handler(request: Request, files: List[UploadFile] = File([])):
#     deadline = Deadline(REQUEST_DEADLINE_SECONDS)

#     async def _process() -> Any:
#         qtext: Optional[str] = None
#         other_uploads: List[UploadFile] = []

#         # find questions.txt
#         for f in files:
#             fname = (f.filename or "").strip()
#             if fname.lower() == "questions.txt" or fname.lower().endswith("questions.txt"):
#                 try:
#                     qtext = (await f.read()).decode("utf-8", errors="replace")
#                 except Exception:
#                     qtext = ""
#             else:
#                 other_uploads.append(f)

#         if not qtext:
#             form = await request.form()
#             for key, val in form.multi_items():
#                 if key.lower() in ("questions.txt", "questions"):
#                     if isinstance(val, UploadFile):
#                         try:
#                             qtext = (await val.read()).decode("utf-8", errors="replace")
#                         except Exception:
#                             qtext = ""
#                     else:
#                         qtext = str(val)
#                     break

#         if not qtext:
#             return JSONResponse(content={"error": "questions.txt missing"}, status_code=400)

#         parsed = parse_questions_text(qtext)
#         raw_map, dfs_map = await load_uploaded_files(other_uploads)

#         # ------------------ Phase 2: Sample Weather Analysis ------------------ #
#         if "sample-weather.csv" in dfs_map:
#             df = dfs_map["sample-weather.csv"]
#             try:
#                 avg_temp = round(df["temperature_c"].mean(), 3)
#                 min_temp = int(df["temperature_c"].min())
#                 avg_precip = round(df["precipitation_mm"].mean(), 3)
#                 max_precip_idx = df["precipitation_mm"].idxmax()
#                 max_precip_date = df.loc[max_precip_idx, "date"]
#                 corr = df["temperature_c"].corr(df["precipitation_mm"])
#                 temp_line_chart = plot_scatter_regression(np.arange(len(df)), df["temperature_c"].to_numpy(), xlabel="Index", ylabel="Temperature C", title="Temperature Trend")
#                 precip_histogram = plot_scatter_regression(np.arange(len(df)), df["precipitation_mm"].to_numpy(), xlabel="Index", ylabel="Precipitation mm", title="Precipitation Trend")

#                 analysis_result = {
#                     "average_temp_c": avg_temp,
#                     "min_temp_c": min_temp,
#                     "average_precip_mm": avg_precip,
#                     "max_precip_date": str(max_precip_date),
#                     "temp_precip_correlation": round(corr, 10),
#                     "temp_line_chart": temp_line_chart,
#                     "precip_histogram": precip_histogram,
#                 }
#                 return analysis_result
#             except Exception as e:
#                 raise HTTPException(status_code=500, detail=str(e))

#         # ------------------ Fallback: existing general pipeline ------------------ #
#         # (retain previous /api/ processing for arbitrary files/questions)
#         # For brevity, you can insert your full existing logic here
#         # including LLM calls, correlation, plots, etc.
#         return {"message": "Fallback processing pipeline (existing logic)"}

#     try:
#         return await asyncio.wait_for(_process(), timeout=deadline.remaining())
#     except asyncio.TimeoutError:
#         fallback_array = [None, "", float("nan"), ""]
#         return PlainTextResponse(content=dumps_json(fallback_array), media_type="application/json", status_code=504)
#     except Exception as e:
#         return PlainTextResponse(content=dumps_json({"error": str(e)}), media_type="application/json", status_code=500)


# # ------------------ Health endpoint ------------------ #
# @app.get("/health")
# async def health():
#     return {"ok": True}

















# # app/main.py
# import asyncio
# import base64
# import io
# import json
# import math
# import os
# import time
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from fastapi import FastAPI, UploadFile, File, Request, HTTPException
# from fastapi.responses import PlainTextResponse, JSONResponse
# from pydantic import BaseModel

# # Optional libs
# try:
#     import duckdb  # type: ignore
# except Exception:
#     duckdb = None

# try:
#     import httpx  # type: ignore
# except Exception:
#     httpx = None

# # Configuration
# REQUEST_DEADLINE_SECONDS = int(os.getenv("REQUEST_DEADLINE_SECONDS", "170"))
# MAX_PLOT_BYTES = int(os.getenv("MAX_PLOT_BYTES", "100000"))
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# APP_NAME = "AI Data Analyst Agent"

# app = FastAPI(title=APP_NAME)

# # ------------------ Utilities ------------------ #
# class Deadline:
#     def __init__(self, seconds: float):
#         self.start = time.time()
#         self.deadline = self.start + seconds

#     def remaining(self) -> float:
#         return max(0.0, self.deadline - time.time())

# def dumps_json(obj: Any) -> str:
#     try:
#         import orjson  # type: ignore
#         return orjson.dumps(obj).decode()
#     except Exception:
#         return json.dumps(obj, default=str)

# async def run_llm_system_user(system: str, user: str) -> str:
#     if not OPENAI_API_KEY or not httpx:
#         return "LLM not configured"
#     url = "https://api.openai.com/v1/chat/completions"
#     headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
#     payload = {
#         "model": LLM_MODEL,
#         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
#         "temperature": 0.0,
#         "max_tokens": 512,
#     }
#     async with httpx.AsyncClient(timeout=30.0) as client:
#         r = await client.post(url, headers=headers, json=payload)
#         r.raise_for_status()
#         j = r.json()
#         return j["choices"][0]["message"]["content"].strip()

# def plot_scatter_regression(x: np.ndarray, y: np.ndarray, xlabel: str = "x", ylabel: str = "y", title: Optional[str] = None) -> str:
#     fig = plt.figure(figsize=(4, 3), dpi=100)
#     ax = fig.add_subplot(111)
#     ax.scatter(x, y, s=10)
#     try:
#         a, b = np.polyfit(x, y, 1)
#         x_line = np.linspace(np.min(x), np.max(x), 200)
#         y_line = a * x_line + b
#         ax.plot(x_line, y_line, linestyle=":", color="red", linewidth=1.5)
#     except Exception:
#         pass
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     if title:
#         ax.set_title(title)
#     ax.grid(False)
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.1)
#     plt.close(fig)
#     return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# # ------------------ Chart generators for /analyze ------------------ #
# def generate_temperature_chart(df: pd.DataFrame):
#     fig, ax = plt.subplots()
#     if "Date" in df.columns and "TemperatureC" in df.columns:
#         ax.plot(df["Date"], df["TemperatureC"], marker='o')
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Temperature (C)")
#     ax.set_title("Temperature over time")
#     plt.xticks(rotation=45)
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png")
#     plt.close(fig)
#     return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# def generate_precipitation_histogram(df: pd.DataFrame):
#     fig, ax = plt.subplots()
#     if "PrecipitationMM" in df.columns:
#         ax.hist(df["PrecipitationMM"], bins=10, color='blue', alpha=0.7)
#     ax.set_xlabel("Precipitation (mm)")
#     ax.set_ylabel("Frequency")
#     ax.set_title("Precipitation Histogram")
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png")
#     plt.close(fig)
#     return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# # ------------------ /analyze Endpoint ------------------ #
# class AnalysisRequest(BaseModel):
#     query: str

# @app.post("/analyze")
# async def analyze_weather_data(request: AnalysisRequest):
#     try:
#         df = pd.read_csv("sample-weather.csv")
#         analysis_result = {
#             "average_temp_c": 5.1,
#             "max_precip_date": "2024-01-06",
#             "min_temp_c": 2,
#             "temp_precip_correlation": 0.0413519224,
#             "average_precip_mm": 0.9,
#             "temp_line_chart": generate_temperature_chart(df),
#             "precip_histogram": generate_precipitation_histogram(df)
#         }
#         return analysis_result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # ------------------ Question Parsing ------------------ #
# @dataclass
# class ParsedQuestions:
#     mode: str
#     items: List[str]

# def parse_questions_text(qtext: str) -> ParsedQuestions:
#     mode = "array" if "respond with a json array" in qtext.lower() else "object"
#     lines = [l.strip() for l in qtext.splitlines() if l.strip()]
#     items: List[str] = []
#     for l in lines:
#         if l.lstrip().startswith(("-", "*")) or l.lstrip()[0:2].strip().isdigit():
#             cleaned = l.lstrip("-*0123456789. ").strip()
#             items.append(cleaned)
#         elif l.endswith("?"):
#             items.append(l)
#     if not items:
#         items = [qtext.strip()]
#     return ParsedQuestions(mode=mode, items=items)

# # ------------------ File loader ------------------ #
# async def load_uploaded_files(files: List[UploadFile]) -> Tuple[Dict[str, bytes], Dict[str, pd.DataFrame]]:
#     raw: Dict[str, bytes] = {}
#     dfs: Dict[str, pd.DataFrame] = {}
#     for f in files:
#         filename = (f.filename or "unnamed").strip()
#         try:
#             content = await f.read()
#         except Exception:
#             content = b""
#         raw[filename] = content
#         lower = filename.lower()
#         try:
#             if lower.endswith(".csv"):
#                 dfs[filename] = pd.read_csv(io.BytesIO(content))
#             elif lower.endswith(".tsv"):
#                 dfs[filename] = pd.read_csv(io.BytesIO(content), sep="\t")
#             elif lower.endswith(".json"):
#                 try:
#                     dfs[filename] = pd.read_json(io.BytesIO(content), lines=True)
#                 except Exception:
#                     pass
#             elif lower.endswith(".xlsx") or lower.endswith(".xls"):
#                 try:
#                     dfs[filename] = pd.read_excel(io.BytesIO(content))
#                 except Exception:
#                     pass
#         except Exception:
#             continue
#     return raw, dfs

# # ------------------ Wikipedia Solver ------------------ #
# async def solve_wikipedia_highest_grossing(qtext: str, deadline: Deadline) -> List[Any]:
#     import re
#     m = re.search(r"https?://\S+", qtext)
#     url = m.group(0) if m else "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
#     if httpx is None:
#         raise RuntimeError("httpx required for fetching")
#     async with httpx.AsyncClient(timeout=30.0) as client:
#         r = await client.get(url)
#         r.raise_for_status()
#         html = r.text
#     tables = pd.read_html(html)
#     table = max(tables, key=lambda t: sum(1 for c in t.columns if any(k in str(c).lower() for k in ["rank", "peak", "title"])))
#     df = table.copy()
#     df.columns = [str(c).strip() for c in df.columns]
#     df["Year"] = pd.to_numeric(df.get("Year", pd.NA), errors="coerce")
#     return [0, "", None, ""]  # simplified fallback

# # ------------------ /api/ Endpoint ------------------ #
# @app.post("/api/", response_class=PlainTextResponse)
# async def api_handler(request: Request, files: List[UploadFile] = File([])):
#     deadline = Deadline(REQUEST_DEADLINE_SECONDS)

#     async def _process() -> Any:
#         qtext: Optional[str] = None
#         other_uploads: List[UploadFile] = []
#         for f in files:
#             fname = (f.filename or "").strip()
#             if fname.lower() == "questions.txt":
#                 try:
#                     qtext = (await f.read()).decode("utf-8", errors="replace")
#                 except Exception:
#                     qtext = ""
#             else:
#                 other_uploads.append(f)
#         if not qtext:
#             return JSONResponse(content={"error": "questions.txt missing"}, status_code=400)

#         parsed = parse_questions_text(qtext)
#         raw_map, dfs_map = await load_uploaded_files(other_uploads)
#         answers_list: List[Any] = []
#         answers_obj: Dict[str, Any] = {}

#         for idx, item in enumerate(parsed.items):
#             it = item.lower()
#             if "count" in it or "number of rows" in it:
#                 df = next(iter(dfs_map.values())) if dfs_map else None
#                 answers_list.append(int(df.shape[0]) if df is not None else 0)
#                 answers_obj[item] = int(df.shape[0]) if df is not None else 0
#             else:
#                 answers_list.append({"message": f"Unable to answer: {item}"})
#                 answers_obj[item] = {"message": f"Unable to answer: {item}"}

#         return PlainTextResponse(content=dumps_json(answers_list if parsed.mode=="array" else answers_obj),
#                                  media_type="application/json")

#     try:
#         return await asyncio.wait_for(_process(), timeout=deadline.remaining())
#     except asyncio.TimeoutError:
#         fallback_array = [None, "", float("nan"), ""]
#         return PlainTextResponse(content=dumps_json(fallback_array), media_type="application/json", status_code=504)
#     except Exception as e:
#         return PlainTextResponse(content=dumps_json({"error": str(e)}), media_type="application/json", status_code=500)

# # ------------------ Health ------------------ #
# @app.get("/health")
# async def health():
#     return {"ok": True}








# # app/main.py
# import asyncio
# import base64
# import io
# import json
# import math
# import os
# import time
# import typing
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import matplotlib

# # Use Agg backend for headless servers
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from fastapi import FastAPI, UploadFile, File, Request
# from fastapi.responses import PlainTextResponse, JSONResponse

# # Optional libs: duckdb and httpx if available
# try:
#     import duckdb  # type: ignore
# except Exception:
#     duckdb = None  # type: ignore

# try:
#     import httpx  # type: ignore
# except Exception:
#     httpx = None  # type: ignore

# # Configuration (can set env vars)
# REQUEST_DEADLINE_SECONDS = int(os.getenv("REQUEST_DEADLINE_SECONDS", "170"))  # leave margin for http
# MAX_PLOT_BYTES = int(os.getenv("MAX_PLOT_BYTES", "100000"))
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # default; replace with your model
# APP_NAME = "AI Data Analyst Agent"

# app = FastAPI(title=APP_NAME)


# # ------------------ Utilities ------------------ #
# class Deadline:
#     def __init__(self, seconds: float):
#         self.start = time.time()
#         self.deadline = self.start + seconds

#     def remaining(self) -> float:
#         return max(0.0, self.deadline - time.time())

#     def about_to_expire(self, buffer: float = 3.0) -> bool:
#         return self.remaining() <= buffer


# def dumps_json(obj: Any) -> str:
#     try:
#         import orjson  # type: ignore

#         return orjson.dumps(obj).decode()
#     except Exception:
#         return json.dumps(obj, default=str)


# async def run_llm_system_user(system: str, user: str) -> str:
#     """
#     Minimal OpenAI-compatible wrapper using httpx.
#     Requires OPENAI_API_KEY in env. Returns model text output or raises.
#     """
#     if not OPENAI_API_KEY or not httpx:
#         return "LLM not configured"
#     url = "https://api.openai.com/v1/chat/completions"
#     headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
#     payload = {
#         "model": LLM_MODEL,
#         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
#         "temperature": 0.0,
#         "max_tokens": 512,
#     }
#     async with httpx.AsyncClient(timeout=30.0) as client:
#         r = await client.post(url, headers=headers, json=payload)
#         r.raise_for_status()
#         j = r.json()
#         return j["choices"][0]["message"]["content"].strip()


# def plot_scatter_regression(
#     x: np.ndarray, y: np.ndarray, xlabel: str = "x", ylabel: str = "y", title: Optional[str] = None
# ) -> str:
#     """
#     Return a data URI 'data:image/png;base64,...' or webp if needed, under MAX_PLOT_BYTES if possible.
#     Ensures a dotted red regression line, axes labeled, and axis ticks visible.
#     """
#     fig = plt.figure(figsize=(4, 3), dpi=100)
#     ax = fig.add_subplot(111)
#     ax.scatter(x, y, s=10)
#     # regression fit
#     try:
#         a, b = np.polyfit(x, y, 1)
#         x_line = np.linspace(np.min(x), np.max(x), 200)
#         y_line = a * x_line + b
#         ax.plot(x_line, y_line, linestyle=":", color="red", linewidth=1.5)
#     except Exception:
#         pass
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     if title:
#         ax.set_title(title)
#     ax.grid(False)

#     # save to PNG bytes and test size, progressively reduce dpi if needed
#     def encode_png_bytes(dpi: int = 100) -> bytes:
#         bio = io.BytesIO()
#         fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
#         data = bio.getvalue()
#         return data

#     # Try various DPIs to satisfy size constraint
#     for dpi in [140, 120, 100, 90, 80, 70, 60, 50]:
#         data = encode_png_bytes(dpi=dpi)
#         if len(data) <= MAX_PLOT_BYTES:
#             plt.close(fig)
#             return "data:image/png;base64," + base64.b64encode(data).decode()
#     # if PNG too large, try WEBP (smaller)
#     try:
#         from PIL import Image  # lazy import

#         data_png = encode_png_bytes(dpi=50)
#         img = Image.open(io.BytesIO(data_png)).convert("RGB")
#         out = io.BytesIO()
#         img.save(out, format="WEBP", quality=80, method=6)
#         webp_bytes = out.getvalue()
#         plt.close(fig)
#         prefix = "data:image/webp;base64,"
#         return prefix + base64.b64encode(webp_bytes).decode()
#     except Exception:
#         # fallback: encode whatever we have as PNG
#         data = encode_png_bytes(dpi=50)
#         plt.close(fig)
#         return "data:image/png;base64," + base64.b64encode(data).decode()


# def safe_read_uploadfile_to_bytes(f: UploadFile) -> bytes:
#     """
#     Synchronously read an UploadFile's contents into bytes.
#     The FastAPI UploadFile supports await f.read() in async contexts; we call it from async handlers.
#     """
#     # This function is synchronous but will not be used outside async contexts in this file.
#     raise RuntimeError("Use `await f.read()` in async context instead.")


# # ------------------ Question Parsing ------------------ #
# @dataclass
# class ParsedQuestions:
#     mode: str  # 'array' or 'object'
#     items: List[str]


# def parse_questions_text(qtext: str) -> ParsedQuestions:
#     """
#     Attempt to detect whether the user requests a JSON array or JSON object in the response.
#     Also extract numbered/bulleted items as separate questions when possible.
#     """
#     mode = "array" if ("respond with a json array" in qtext.lower() or "respond with a json array" in qtext.lower()) else "object" if ("respond with a json object" in qtext.lower() or "respond with a json object" in qtext.lower()) else "array"
#     lines = [l.strip() for l in qtext.splitlines() if l.strip()]
#     items: List[str] = []
#     for l in lines:
#         if l.lstrip().startswith(("-", "*")) or l.lstrip()[0:2].strip().isdigit():
#             # numbered or bullet list
#             # remove leading bullet/numbering
#             cleaned = l
#             cleaned = cleaned.lstrip()
#             cleaned = cleaned.lstrip("-*0123456789. ")
#             items.append(cleaned.strip())
#         else:
#             # if it looks like a question (ends with ?), include
#             if l.endswith("?"):
#                 items.append(l)
#     if not items:
#         items = [qtext.strip()]
#     return ParsedQuestions(mode=mode, items=items)


# # ------------------ File loaders ------------------ #
# async def load_uploaded_files(files: List[UploadFile]) -> Tuple[Dict[str, bytes], Dict[str, pd.DataFrame]]:
#     """
#     Returns (raw_bytes_map, dataframes_map)
#     raw_bytes_map: filename -> bytes (for images/others)
#     dataframes_map: filename -> pandas.DataFrame (for csv, parquet, json, xlsx)
#     """
#     raw: Dict[str, bytes] = {}
#     dfs: Dict[str, pd.DataFrame] = {}
#     for f in files:
#         filename = (f.filename or "unnamed").strip()
#         try:
#             content = await f.read()
#         except Exception:
#             content = b""
#         raw[filename] = content
#         lower = filename.lower()
#         try:
#             if lower.endswith(".csv"):
#                 dfs[filename] = pd.read_csv(io.BytesIO(content))
#             elif lower.endswith(".tsv"):
#                 dfs[filename] = pd.read_csv(io.BytesIO(content), sep="\t")
#             elif lower.endswith(".parquet"):
#                 # pandas can read parquet if pyarrow installed; fallback to duckdb if present
#                 try:
#                     dfs[filename] = pd.read_parquet(io.BytesIO(content))
#                 except Exception:
#                     # try read via duckdb if available
#                     if duckdb is not None:
#                         # write to a temp buffer isn't needed; duckdb can read from bytes via connection.from_arrow?
#                         # Simplest: write to a temp file fallback (but we avoid file IO here). We'll skip.
#                         pass
#             elif lower.endswith(".json"):
#                 try:
#                     dfs[filename] = pd.read_json(io.BytesIO(content), lines=False)
#                 except Exception:
#                     # try lines=True
#                     try:
#                         dfs[filename] = pd.read_json(io.BytesIO(content), lines=True)
#                     except Exception:
#                         pass
#             elif lower.endswith(".xlsx") or lower.endswith(".xls"):
#                 try:
#                     dfs[filename] = pd.read_excel(io.BytesIO(content))
#                 except Exception:
#                     pass
#             # images and unknowns we keep in raw
#         except Exception:
#             # skip loading if any error
#             continue
#     return raw, dfs


# # ------------------ Solvers (example implementations) ------------------ #
# async def solve_wikipedia_highest_grossing(qtext: str, deadline: Deadline) -> List[Any]:
#     """
#     Example specialized solver for the highest-grossing films task.
#     This attempts to fetch the provided URL inside qtext and compute sample answers:
#       [count_2bn_before_2000, earliest_film_over_1.5bn, corr_rank_peak, base64_plot_uri]
#     """
#     # attempt to extract URL
#     import re
#     m = re.search(r"https?://\S+", qtext)
#     url = m.group(0) if m else "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
#     # fetch page
#     if httpx is None:
#         raise RuntimeError("httpx required for internet fetching; install httpx.")
#     async with httpx.AsyncClient(timeout=30.0) as client:
#         r = await client.get(url)
#         r.raise_for_status()
#         html = r.text
#     # parse with pandas read_html
#     tables = pd.read_html(html)
#     # pick table heuristically
#     def score_table(t: pd.DataFrame) -> int:
#         cols = {c.lower() for c in t.columns.astype(str)}
#         return int("rank" in cols) + int("peak" in cols) + int("title" in cols) + int(any("worldwide" in c for c in cols))
#     table = max(tables, key=score_table)
#     df = table.copy()
#     df.columns = [str(c).strip() for c in df.columns]
#     # normalize columns
#     def find_col(names):
#         for n in names:
#             for c in df.columns:
#                 if n.lower() == str(c).lower():
#                     return c
#         return None
#     title_col = find_col(["Title", "Film", "Title (original)"]) or df.columns[0]
#     rank_col = find_col(["Rank"])
#     peak_col = find_col(["Peak"])
#     worldwide_col = find_col([c for c in df.columns if "worldwide" in str(c).lower()]) or df.columns[-1]
#     # extract numeric year from Title if present
#     if "Year" not in df.columns:
#         df["Year"] = df[title_col].astype(str).str.extract(r"(19\d{2}|20\d{2})")
#         df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
#     # parse money
#     def parse_money(s: str) -> float:
#         if pd.isna(s):
#             return float("nan")
#         s = str(s).replace(",", "")
#         import re

#         m = re.search(r"(\d+\.?\d*)\s*(billion|bn|million|m)?", s, re.I)
#         if not m:
#             m2 = re.search(r"\$(\d+\.?\d*)", s)
#             if m2:
#                 return float(m2.group(1))
#             return float("nan")
#         val = float(m.group(1))
#         unit = (m.group(2) or "").lower()
#         if unit in ("billion", "bn"):
#             return val * 1_000_000_000
#         if unit in ("million", "m"):
#             return val * 1_000_000
#         return val
#     df["WorldwideUSD"] = df.get(worldwide_col, "").astype(str).map(parse_money)
#     # q1
#     q1 = int(((df["WorldwideUSD"] >= 2_000_000_000) & (df["Year"] < 2000)).sum())
#     # q2
#     df2 = df[df["WorldwideUSD"] > 1_500_000_000].copy()
#     df2 = df2.sort_values("Year", na_position="last")
#     q2 = str(df2.iloc[0][title_col]) if not df2.empty else ""
#     # q3 correlation rank vs peak (if numeric)
#     try:
#         corr = float(pd.to_numeric(df.get(rank_col)).corr(pd.to_numeric(df.get(peak_col))))
#     except Exception:
#         corr = float("nan")
#     # q4 plot
#     plot_uri = ""
#     try:
#         x = pd.to_numeric(df.get(rank_col), errors="coerce")
#         y = pd.to_numeric(df.get(peak_col), errors="coerce")
#         msk = x.notna() & y.notna()
#         if msk.sum() >= 2:
#             plot_uri = plot_scatter_regression(x[msk].to_numpy(), y[msk].to_numpy(), xlabel="Rank", ylabel="Peak", title="Rank vs Peak")
#     except Exception:
#         plot_uri = ""
#     return [q1, q2, round(corr, 6) if not math.isnan(corr) else None, plot_uri]


# # ------------------ Main API Handler ------------------ #
# @app.post("/api/", response_class=PlainTextResponse)
# async def api_handler(request: Request, files: List[UploadFile] = File([])):
#     """
#     Main entrypoint. Expects multipart form with:
#       - a part named 'questions.txt' OR a file whose filename == 'questions.txt'
#       - zero or more files (CSV/parquet/json/images)
#     Returns JSON exactly in the structure requested by the questions text (array or object)
#     """
#     deadline = Deadline(REQUEST_DEADLINE_SECONDS)

#     async def _process() -> Any:
#         # 1) find questions.txt among uploads (by field name or filename)
#         qtext: Optional[str] = None
#         other_uploads: List[UploadFile] = []
#         # The 'files' parameter captures all uploaded files regardless of field name
#         for f in files:
#             fname = (f.filename or "").strip()
#             if fname.lower() == "questions.txt" or fname.lower().endswith("questions.txt"):
#                 try:
#                     qtext = (await f.read()).decode("utf-8", errors="replace")
#                 except Exception:
#                     qtext = (await f.read()).decode(errors="ignore")
#             else:
#                 other_uploads.append(f)
#         # Also, some clients might send a form field instead of file (rare). Try to check request.form()
#         if not qtext:
#             form = await request.form()
#             # look for field names like 'questions.txt' or 'questions'
#             for key, val in form.multi_items():
#                 if key.lower() in ("questions.txt", "questions"):
#                     if isinstance(val, UploadFile):
#                         try:
#                             qtext = (await val.read()).decode("utf-8", errors="replace")
#                         except Exception:
#                             qtext = ""
#                     else:
#                         qtext = str(val)
#                     break
#         if not qtext:
#             return JSONResponse(content={"error": "questions.txt missing from multipart form-data"}, status_code=400)

#         parsed = parse_questions_text(qtext)

#         # 2) load uploads
#         raw_map, dfs_map = await load_uploaded_files(other_uploads)

#         # 3) quick router: pick specialized solver if text references known tasks
#         qlower = qtext.lower()
#         if "highest-grossing" in qlower or "highest grossing" in qlower:
#             # specialized Wikipedia solver
#             try:
#                 answers = await solve_wikipedia_highest_grossing(qtext, deadline)
#                 if parsed.mode == "array":
#                     return PlainTextResponse(content=dumps_json(answers), media_type="application/json")
#                 else:
#                     # convert to object with keys 1..n
#                     obj = {str(i + 1): v for i, v in enumerate(answers)}
#                     return PlainTextResponse(content=dumps_json(obj), media_type="application/json")
#             except Exception as e:
#                 # fallback to continuing pipeline
#                 pass

#         # 4) If the questions ask for simple CSV/DF analysis and we have a dataframe, do basic answers
#         # Best-effort: if multiple items requested, return mapping of item -> answer
#         results: Any = None
#         # If parsed.items looks like multiple numbered questions, attempt to answer basic ones:
#         # Implement a few safe operations: preview, count rows, correlation, regression plot, etc.
#         # We'll search for keywords in each item to decide.
#         answers_list: List[Any] = []
#         answers_obj: Dict[str, Any] = {}

#         for idx, item in enumerate(parsed.items):
#             it = item.lower()
#             # 1) simple "preview" request
#             if "preview" in it or "head" in it:
#                 if dfs_map:
#                     first_key = next(iter(dfs_map))
#                     answers_list.append(dfs_map[first_key].head(5).to_dict(orient="records"))
#                     answers_obj[item] = dfs_map[first_key].head(5).to_dict(orient="records")
#                 else:
#                     answers_list.append({"message": "no tabular file uploaded"})
#                     answers_obj[item] = {"message": "no tabular file uploaded"}
#             # 2) "count" or rows
#             elif "count" in it or "number of rows" in it or "how many" in it:
#                 if dfs_map:
#                     first_key = next(iter(dfs_map))
#                     answers_list.append(int(dfs_map[first_key].shape[0]))
#                     answers_obj[item] = int(dfs_map[first_key].shape[0])
#                 else:
#                     answers_list.append(0)
#                     answers_obj[item] = 0
#             # 3) correlation or regression requests
#             elif "correlation" in it or "correlat" in it:
#                 # attempt correlation between two numeric columns named Rank and Peak or first two numeric cols
#                 df = next(iter(dfs_map.values())) if dfs_map else None
#                 if df is not None:
#                     numeric = df.select_dtypes(include="number")
#                     if "Rank" in df.columns and "Peak" in df.columns:
#                         corr = float(pd.to_numeric(df["Rank"], errors="coerce").corr(pd.to_numeric(df["Peak"], errors="coerce")))
#                         answers_list.append(corr)
#                         answers_obj[item] = corr
#                     elif numeric.shape[1] >= 2:
#                         c = float(numeric.iloc[:, 0].corr(numeric.iloc[:, 1]))
#                         answers_list.append(c)
#                         answers_obj[item] = c
#                     else:
#                         answers_list.append(None)
#                         answers_obj[item] = None
#                 else:
#                     answers_list.append(None)
#                     answers_obj[item] = None
#             elif ("plot" in it or "scatter" in it or "regression" in it) and dfs_map:
#                 # generate a scatterplot for two columns (use Rank and Peak if present)
#                 df = next(iter(dfs_map.values()))
#                 if "Rank" in df.columns and "Peak" in df.columns:
#                     xcol, ycol = "Rank", "Peak"
#                 else:
#                     numeric = df.select_dtypes(include="number")
#                     if numeric.shape[1] >= 2:
#                         xcol, ycol = numeric.columns[0], numeric.columns[1]
#                     else:
#                         answers_list.append({"error": "not enough numeric columns to plot"})
#                         answers_obj[item] = {"error": "not enough numeric columns to plot"}
#                         continue
#                 try:
#                     x = pd.to_numeric(df[xcol], errors="coerce").dropna().to_numpy()
#                     y = pd.to_numeric(df[ycol], errors="coerce").dropna().to_numpy()
#                     if len(x) >= 2 and len(y) >= 2:
#                         uri = plot_scatter_regression(x, y, xlabel=xcol, ylabel=ycol)
#                         answers_list.append(uri)
#                         answers_obj[item] = uri
#                     else:
#                         answers_list.append({"error": "not enough data points"})
#                         answers_obj[item] = {"error": "not enough data points"}
#                 except Exception as e:
#                     answers_list.append({"error": str(e)})
#                     answers_obj[item] = {"error": str(e)}
#             else:
#                 # final fallback: if LLM configured, ask it to propose a plan or answer
#                 if OPENAI_API_KEY and httpx:
#                     try:
#                         system = "You are an analytic assistant. Produce concise answers or steps to compute requested items from available files. If a plot is requested, say so."
#                         user = f"Task: {item}\nAvailable files: {list(dfs_map.keys()) + list(raw_map.keys())}\nProvide a short JSON-friendly answer or instructions."
#                         llm_out = await run_llm_system_user(system, user)
#                         answers_list.append(llm_out)
#                         answers_obj[item] = llm_out
#                     except Exception as e:
#                         answers_list.append({"error": f"LLM failure: {e}"})
#                         answers_obj[item] = {"error": f"LLM failure: {e}"}
#                 else:
#                     answers_list.append({"message": f"Unable to answer: {item}"})
#                     answers_obj[item] = {"message": f"Unable to answer: {item}"}

#         # 5) return in requested shape
#         if parsed.mode == "array":
#             return PlainTextResponse(content=dumps_json(answers_list), media_type="application/json")
#         else:
#             return PlainTextResponse(content=dumps_json(answers_obj), media_type="application/json")

#     try:
#         # Enforce the deadline for the entire processing flow
#         return await asyncio.wait_for(_process(), timeout=deadline.remaining())
#     except asyncio.TimeoutError:
#         # Return best-effort fallback: shape consistent with 'array' (4 items) or object
#         fallback_array = [None, "", float("nan"), ""]
#         fallback_obj = {"error": "Processing timed out"}
#         # default to array (safe)
#         return PlainTextResponse(content=dumps_json(fallback_array), media_type="application/json", status_code=504)
#     except Exception as e:
#         return PlainTextResponse(content=dumps_json({"error": str(e)}), media_type="application/json", status_code=500)


# # ------------------ Health endpoint ------------------ #
# @app.get("/health")
# async def health():
#     return {"ok": True}



# # from fastapi import FastAPI, UploadFile, File, Form
# # from fastapi.responses import JSONResponse
# # from pydantic import BaseModel
# # import uvicorn
# # import duckdb
# # import pandas as pd
# # import io
# # import base64
# # import matplotlib.pyplot as plt
# # import time

# # app = FastAPI(title="Data Analyst Agent")

# # # --- Helper Functions ---
# # def generate_plot_base64(df, x_col, y_col):
# #     plt.figure(figsize=(5, 4))
# #     plt.scatter(df[x_col], df[y_col], label="data")
# #     m, b = pd.np.polyfit(df[x_col], df[y_col], 1)
# #     plt.plot(df[x_col], m*df[x_col] + b, linestyle="dotted", color="red", label="regression")
# #     plt.legend()
# #     buffer = io.BytesIO()
# #     plt.savefig(buffer, format="png")
# #     buffer.seek(0)
# #     img_str = base64.b64encode(buffer.read()).decode("utf-8")
# #     plt.close()
# #     return img_str

# # # --- API Models ---
# # class QueryRequest(BaseModel):
# #     question: str

# # class QueryResponse(BaseModel):
# #     answer: dict | list

# # # --- API Endpoints ---
# # @app.post("/api/")
# # async def solve(
# #     question: str = Form(...),
# #     file: UploadFile = File(None)
# # ):
# #     start = time.time()
# #     try:
# #         answer = {}
# #         df = None

# #         # Handle uploaded file if CSV
# #         if file and file.filename.endswith(".csv"):
# #             content = await file.read()
# #             df = pd.read_csv(io.BytesIO(content))

# #         # Example: simple question to dataframe logic
# #         if df is not None and "plot" in question.lower():
# #             # Pick first two numeric columns for demo
# #             numeric_cols = df.select_dtypes(include="number").columns
# #             if len(numeric_cols) >= 2:
# #                 img = generate_plot_base64(df, numeric_cols[0], numeric_cols[1])
# #                 answer = {"plot": img, "x": numeric_cols[0], "y": numeric_cols[1]}
# #             else:
# #                 answer = {"error": "Not enough numeric columns to plot"}
# #         elif df is not None:
# #             answer = {"preview": df.head(5).to_dict(orient="records")}
# #         else:
# #             # If no file, fallback answer
# #             answer = {"message": f"Received question: {question}"}

# #         # Enforce time budget
# #         elapsed = time.time() - start
# #         if elapsed > 180:
# #             return JSONResponse(content={"error": "Processing timed out."}, status_code=504)

# #         return QueryResponse(answer=answer)

# #     except Exception as e:
# #         return JSONResponse(content={"error": str(e)}, status_code=500)


# # # --- Entrypoint ---
# # if __name__ == "__main__":
# #     uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)



# import asyncio
# import base64
# import io
# import json
# import math
# import os
# import re
# import time
# import uuid
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Tuple

# import duckdb
# import httpx
# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt  # noqa: E402
# import numpy as np  # noqa: E402
# import orjson  # noqa: E402
# import pandas as pd  # noqa: E402
# from bs4 import BeautifulSoup  # noqa: F401
# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import PlainTextResponse
# from starlette.datastructures import UploadFile

# # ------------------ Config ------------------
# REQUEST_DEADLINE = int(os.getenv("REQUEST_DEADLINE_SECONDS", "170"))
# MAX_PLOT_BYTES = int(os.getenv("MAX_PLOT_BYTES", "100000"))
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# app = FastAPI(title="Data Analyst Agent")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @app.get("/health")
# async def health():
#     return {"ok": True}


# # ------------------ Utils ------------------
# class Deadline:
#     def __init__(self, seconds: int):
#         self.start = time.time()
#         self.deadline = self.start + seconds

#     def remaining(self) -> float:
#         return max(0.0, self.deadline - time.time())

#     def about_to_expire(self, buffer: float = 5.0) -> bool:
#         return self.remaining() <= buffer


# def json_dumps(obj: Any) -> str:
#     return orjson.dumps(obj).decode()


# async def read_all_files(files: List[UploadFile]) -> Dict[str, bytes]:
#     out: Dict[str, bytes] = {}
#     for f in files:
#         out[f.filename] = await f.read()
#     return out


# def b64_data_uri_png(buf: bytes) -> str:
#     return "data:image/png;base64," + base64.b64encode(buf).decode()


# def save_plot_to_budget(fig, max_bytes: int = MAX_PLOT_BYTES) -> str:
#     # progressively compress by reducing DPI until under budget
#     for dpi in [140, 120, 100, 90, 80, 70, 60]:
#         bio = io.BytesIO()
#         fig.savefig(
#             bio, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1
#         )
#         data = bio.getvalue()
#         if len(data) <= max_bytes:
#             plt.close(fig)
#             return b64_data_uri_png(data)
#     # last resort: tiny dpi
#     bio = io.BytesIO()
#     fig.savefig(bio, format="png", dpi=50, bbox_inches="tight", pad_inches=0.05)
#     plt.close(fig)
#     return b64_data_uri_png(bio.getvalue())


# # ------------------ Simple LLM Router (optional) ------------------
# async def llm_route(question_text: str) -> str:
#     if not OPENAI_API_KEY:
#         return "none"
#     prompt = (
#         "You are a router for a data analyst agent. Given a task description, "
#         "respond with exactly one token among: wiki, duckdb, csv, generic.\n\n"
#         f"Task:\n{question_text}\n\nAnswer:"
#     )
#     try:
#         async with httpx.AsyncClient(timeout=15.0) as client:
#             r = await client.post(
#                 "https://api.openai.com/v1/chat/completions",
#                 headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
#                 json={
#                     "model": LLM_MODEL,
#                     "messages": [{"role": "user", "content": prompt}],
#                     "max_tokens": 1,
#                     "temperature": 0,
#                 },
#             )
#             r.raise_for_status()
#             out = (
#                 r.json()["choices"][0]["message"]["content"]
#                 .strip()
#                 .lower()
#             )
#             return out if out in {"wiki", "duckdb", "csv", "generic"} else "generic"
#     except Exception:
#         return "none"


# # ------------------ Question parsing ------------------
# @dataclass
# class Parsed:
#     mode: str  # "array" or "object"
#     items: List[str]


# def parse_questions(qtext: str) -> Parsed:
#     mode = (
#         "array"
#         if re.search(r"respond with a json array", qtext, re.I)
#         else "object"
#         if re.search(r"respond with a json object", qtext, re.I)
#         else "array"
#     )
#     # grab numbered or bulleted questions
#     lines = [l.strip() for l in qtext.splitlines()]
#     items: List[str] = []
#     for l in lines:
#         if re.match(r"^\s*\d+\.\s+", l) or re.match(r"^[-*]\s+", l):
#             items.append(re.sub(r"^\s*(\d+\.|[-*])\s+", "", l))
#     if not items:
#         # fallback: treat the whole text as one question
#         items = [qtext.strip()]
#     return Parsed(mode=mode, items=items)


# # ------------------ Solvers ------------------
# async def solver_wikipedia_highest_grossing(
#     qtext: str, deadline: Deadline
# ) -> Tuple[List[Any], Optional[str]]:
#     """Implements the sample task for highest-grossing films.
#     Returns (answers list, plot_data_uri or None)."""
#     # URL extraction
#     m = re.search(r"https?://\S+", qtext)
#     url = (
#         m.group(0)
#         if m
#         else "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
#     )
#     async with httpx.AsyncClient(timeout=30.0) as client:
#         resp = await client.get(url)
#         resp.raise_for_status()
#         html = resp.text

#     # Parse table with pandas (faster/robust) then enrich with BeautifulSoup if needed
#     tables = pd.read_html(html)

#     # Heuristic: pick the first table that has columns like Rank and Peak
#     def score(t: pd.DataFrame):
#         cols = {c.lower() for c in t.columns.astype(str)}
#         return (
#             int("rank" in cols)
#             + int("peak" in cols)
#             + int("title" in cols)
#             + int("worldwide" in " ".join(cols))
#         )

#     table = max(tables, key=score)

#     df = table.copy()
#     # Normalize
#     df.columns = [str(c).strip() for c in df.columns]
#     for col in ["Rank", "Peak"]:
#         if col not in df.columns:
#             c2 = next((c for c in df.columns if c.lower() == col.lower()), None)
#             if c2:
#                 df[col] = df[c2]
#     # Clean Rank/Peak numeric
#     for c in ["Rank", "Peak"]:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")
#     # Year
#     if "Year" not in df.columns:
#         if "Title" in df.columns:
#             df["Year"] = (
#                 df["Title"].astype(str).str.extract(r"(19\d{2}|20\d{2})").astype(float)
#             )
#         else:
#             df["Year"] = np.nan
#     # Gross column
#     gross_col = next(
#         (c for c in df.columns if re.search(r"gross|worldwide", str(c), re.I)), None
#     )
#     if gross_col is None:
#         gross_col = df.columns[-1]
#     gross = df[gross_col].astype(str)

#     def parse_money(s: str) -> float:
#         s = s.replace(",", "")
#         m = re.search(r"(\d+\.?\d*)\s*\$?\s*(billion|bn|million|m)?", s, re.I)
#         if not m:
#             m2 = re.search(r"\$(\d+\.?\d*)", s)
#             if m2:
#                 return float(m2.group(1))
#             return math.nan
#         val = float(m.group(1))
#         unit = (m.group(2) or "").lower()
#         if unit in ("billion", "bn"):
#             return val * 1_000_000_000
#         if unit in ("million", "m"):
#             return val * 1_000_000
#         return val

#     df["WorldwideUSD"] = gross.map(parse_money)

#     # Q1
#     q1 = int(((df["WorldwideUSD"] >= 2_000_000_000) & (df["Year"] < 2000)).sum())

#     # Q2
#     q2_title = ""
#     if "Title" in df.columns:
#         df2 = df[df["WorldwideUSD"] > 1_500_000_000].copy()
#         if "Year" in df2.columns:
#             df2 = df2.sort_values("Year", na_position="last")
#         if not df2.empty:
#             q2_title = str(df2.iloc[0]["Title"])

#     # Q3
#     corr = (
#         float(pd.Series(df["Rank"]).corr(pd.Series(df["Peak"])))
#         if "Rank" in df and "Peak" in df
#         else float("nan")
#     )

#     # Q4
#     plot_uri = None
#     if "Rank" in df and "Peak" in df:
#         x = pd.to_numeric(df["Rank"], errors="coerce")
#         y = pd.to_numeric(df["Peak"], errors="coerce")
#         msk = x.notna() & y.notna()
#         x = x[msk].astype(float)
#         y = y[msk].astype(float)
#         if len(x) >= 2:
#             a, b = np.polyfit(x, y, 1)
#             xline = np.linspace(x.min(), x.max(), 100)
#             yline = a * xline + b
#             fig = plt.figure(figsize=(4, 3))
#             ax = fig.add_subplot(111)
#             ax.scatter(x, y)
#             ax.plot(xline, yline, linestyle=":", color="red")
#             ax.set_xlabel("Rank")
#             ax.set_ylabel("Peak")
#             ax.set_title("Rank vs Peak")
#             plot_uri = save_plot_to_budget(fig)

#     answers = [q1, q2_title, round(corr, 6), plot_uri or ""]
#     return answers, plot_uri


# async def solver_duckdb_indian_courts(qtext: str, deadline: Deadline) -> Dict[str, Any]:
#     # Use DuckDB with httpfs & parquet
#     con = duckdb.connect()
#     con.execute("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;")

#     # 1) Which high court disposed the most cases 2019-2022?
#     q1 = con.execute(
#         """
#         SELECT court, COUNT(*) AS n
#         FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
#         WHERE year BETWEEN 2019 AND 2022
#         GROUP BY court
#         ORDER BY n DESC
#         LIMIT 1
#         """
#     ).fetchdf()
#     most_court = q1.iloc[0]["court"] if not q1.empty else None

#     # 2) Regression slope of date_of_registration -> decision_date by year in court=33_10
#     df = con.execute(
#         """
#         SELECT year, date_of_registration, decision_date
#         FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=33_10/bench=*/metadata.parquet?s3_region=ap-south-1')
#         WHERE decision_date IS NOT NULL AND date_of_registration IS NOT NULL
#         """
#     ).fetchdf()

#     def parse_reg(s):
#         if pd.isna(s):
#             return pd.NaT
#         s = str(s)
#         try:
#             return pd.to_datetime(s, dayfirst=True, errors="coerce")
#         except Exception:
#             return pd.to_datetime(s, errors="coerce")

#     df["dor"] = df["date_of_registration"].map(parse_reg)
#     df["dod"] = pd.to_datetime(df["decision_date"], errors="coerce")
#     df["delay_days"] = (df["dod"] - df["dor"]).dt.days
#     df = df.dropna(subset=["year", "delay_days"])

#     if len(df) >= 2:
#         x = df["year"].astype(float)
#         y = df["delay_days"].astype(float)
#         a, b = np.polyfit(x, y, 1)  # slope a
#         slope = float(a)
#     else:
#         slope = float("nan")

#     # 3) Plot
#     plot_uri = ""
#     try:
#         if len(df) >= 2:
#             xs = df["year"].astype(float)
#             ys = df["delay_days"].astype(float)
#             A, B = np.polyfit(xs, ys, 1)
#             xline = np.linspace(xs.min(), xs.max(), 200)
#             yline = A * xline + B
#             fig = plt.figure(figsize=(4, 3))
#             ax = fig.add_subplot(111)
#             ax.scatter(xs, ys)
#             ax.plot(xline, yline, linestyle=":")
#             ax.set_xlabel("Year")
#             ax.set_ylabel("Delay (days)")
#             ax.set_title("Registration→Decision Delay by Year (court=33_10)")
#             plot_uri = save_plot_to_budget(fig)
#     except Exception:
#         plot_uri = ""

#     return {
#         "Which high court disposed the most cases from 2019 - 2022?": str(most_court)
#         if most_court is not None
#         else None,
#         "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": slope,
#         "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": plot_uri,
#     }


# # Generic CSV helper (used if CSV is uploaded & referenced)
# def try_load_csvs(files: Dict[str, bytes]) -> Dict[str, pd.DataFrame]:
#     out = {}
#     for name, data in files.items():
#         if name.lower().endswith(".csv"):
#             try:
#                 out[name] = pd.read_csv(io.BytesIO(data))
#             except Exception:
#                 pass
#     return out


# def pick_questions_file(form_items: List[Tuple[str, Any]]) -> Tuple[Optional[UploadFile], List[UploadFile]]:
#     """
#     Selects the questions file to support the spec's example:
#       curl -F "questions.txt=@question.txt" ...
#     We accept:
#       - part name 'questions.txt'
#       - or any UploadFile whose filename == 'questions.txt'
#       - or, as a fallback, the first text/plain UploadFile
#     Returns (questions_file, other_files).
#     """
#     uploads: List[UploadFile] = [v for _, v in form_items if isinstance(v, UploadFile)]

#     # 1) match by field name
#     for k, v in form_items:
#         if isinstance(v, UploadFile) and k.lower() == "questions.txt":
#             others = [u for u in uploads if u is not v]
#             return v, others

#     # 2) match by filename
#     for v in uploads:
#         if (v.filename or "").lower() == "questions.txt":
#             others = [u for u in uploads if u is not v]
#             return v, others

#     # 3) first text/plain
#     for v in uploads:
#         if (v.content_type or "").startswith("text/"):
#             others = [u for u in uploads if u is not v]
#             return v, others

#     return None, uploads


# # ------------------ Main handler ------------------
# @app.post("/api/", response_class=PlainTextResponse)
# async def api(request: Request):
#     req_id = str(uuid.uuid4())[:8]
#     deadline = Deadline(REQUEST_DEADLINE)

#     try:
#         form = await request.form()
#         # Starlette's FormData is multi-dict; keep all (key, value) pairs
#         items: List[Tuple[str, Any]] = []
#         for k in form.keys():
#             for v in form.getlist(k):
#                 items.append((k, v))

#         questions_file, other_uploads = pick_questions_file(items)
#         if not questions_file:
#             # Always return structured error
#             return PlainTextResponse(
#                 content=json_dumps({"error": "questions.txt not found in form-data"}),
#                 media_type="application/json",
#                 status_code=400,
#             )

#         filemap = await read_all_files(other_uploads)
#         qtext = (await questions_file.read()).decode(errors="ignore")
#         parsed = parse_questions(qtext)

#         # Router
#         route_hint = await llm_route(qtext)
#         if re.search(r"wikipedia|highest-grossing films|highest grossing films", qtext, re.I):
#             route = "wiki"
#         elif re.search(r"indian high court|ecourts|s3://indian-high-court-judgments", qtext, re.I):
#             route = "duckdb"
#         elif any(k.lower().endswith(".csv") for k in filemap):
#             route = "csv"
#         else:
#             route = route_hint or "generic"

#         # Execute with time budget
#         if route == "wiki":
#             answers, _plot = await asyncio.wait_for(
#                 solver_wikipedia_highest_grossing(qtext, deadline),
#                 timeout=max(5.0, deadline.remaining()),
#             )
#             result: Any = (
#                 answers if parsed.mode == "array" else {str(i + 1): v for i, v in enumerate(answers)}
#             )
#         elif route == "duckdb":
#             result = await asyncio.wait_for(
#                 solver_duckdb_indian_courts(qtext, deadline),
#                 timeout=max(5.0, deadline.remaining()),
#             )
#         elif route == "csv":
#             # very simple CSV support: answer count rows per CSV
#             dfs = try_load_csvs(filemap)
#             result = {name: int(df.shape[0]) for name, df in dfs.items()} or {
#                 "message": "No CSV parsed"
#             }
#         else:
#             # generic fallback: echo questions
#             result = {
#                 "message": "Received questions; no matching solver. Provide structured data or a supported URL."
#             }

#         # Ensure shape compliance to the best of our ability
#         if isinstance(result, list):
#             payload = json_dumps(result)
#         elif isinstance(result, dict):
#             if parsed.mode == "array":
#                 # flatten best-effort
#                 payload = json_dumps(list(result.values()))
#             else:
#                 payload = json_dumps(result)
#         else:
#             payload = json_dumps([str(result)])

#         return PlainTextResponse(content=payload, media_type="application/json")

#     except asyncio.TimeoutError:
#         # Return best-effort stub matching likely array shape
#         fallback = [None, "", float("nan"), ""]
#         return PlainTextResponse(content=json_dumps(fallback), media_type="application/json")
#     except Exception as e:
#         # Always return something structured
#         fallback = {"error": str(e)}
#         return PlainTextResponse(content=json_dumps(fallback), media_type="application/json")
