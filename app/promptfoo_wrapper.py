from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

@app.get("/analyze-weather")
def analyze_weather():
    # 1. Load CSV
    df = pd.read_csv("sample-weather.csv")

    # 2. Compute metrics
    average_temp_c = df["temp_c"].mean()
    min_temp_c = df["temp_c"].min()
    max_precip_date = df.loc[df["precip_mm"].idxmax(), "date"]
    temp_precip_correlation = df["temp_c"].corr(df["precip_mm"])
    average_precip_mm = df["precip_mm"].mean()

    # 3. Create base64 line chart for temperature
    plt.figure(figsize=(6, 4))
    plt.plot(pd.to_datetime(df["date"]), df["temp_c"], color="red")
    plt.xlabel("Date")
    plt.ylabel("Temperature (C)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=80)
    plt.close()
    temp_line_chart = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 4. Create base64 histogram for precipitation
    plt.figure(figsize=(6, 4))
    plt.hist(df["precip_mm"], color="orange", bins=10)
    plt.xlabel("Precipitation (mm)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=80)
    plt.close()
    precip_histogram = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 5. Return JSON with all required keys
    return JSONResponse({
        "average_temp_c": round(average_temp_c, 2),
        "min_temp_c": round(min_temp_c, 2),
        "max_precip_date": str(max_precip_date),
        "temp_precip_correlation": round(temp_precip_correlation, 8),
        "average_precip_mm": round(average_precip_mm, 2),
        "temp_line_chart": temp_line_chart,
        "precip_histogram": precip_histogram
    })

