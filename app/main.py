from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import duckdb
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import time

app = FastAPI(title="Data Analyst Agent")

# --- Helper Functions ---
def generate_plot_base64(df, x_col, y_col):
    plt.figure(figsize=(5, 4))
    plt.scatter(df[x_col], df[y_col], label="data")
    m, b = pd.np.polyfit(df[x_col], df[y_col], 1)
    plt.plot(df[x_col], m*df[x_col] + b, linestyle="dotted", color="red", label="regression")
    plt.legend()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return img_str

# --- API Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: dict | list

# --- API Endpoints ---
@app.post("/api/")
async def solve(
    question: str = Form(...),
    file: UploadFile = File(None)
):
    start = time.time()
    try:
        answer = {}
        df = None

        # Handle uploaded file if CSV
        if file and file.filename.endswith(".csv"):
            content = await file.read()
            df = pd.read_csv(io.BytesIO(content))

        # Example: simple question to dataframe logic
        if df is not None and "plot" in question.lower():
            # Pick first two numeric columns for demo
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) >= 2:
                img = generate_plot_base64(df, numeric_cols[0], numeric_cols[1])
                answer = {"plot": img, "x": numeric_cols[0], "y": numeric_cols[1]}
            else:
                answer = {"error": "Not enough numeric columns to plot"}
        elif df is not None:
            answer = {"preview": df.head(5).to_dict(orient="records")}
        else:
            # If no file, fallback answer
            answer = {"message": f"Received question: {question}"}

        # Enforce time budget
        elapsed = time.time() - start
        if elapsed > 180:
            return JSONResponse(content={"error": "Processing timed out."}, status_code=504)

        return QueryResponse(answer=answer)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# --- Entrypoint ---
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
