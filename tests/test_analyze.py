import pytest
from fastapi.testclient import TestClient
import base64
import os
import pandas as pd
from main import app

client = TestClient(app)

# Create a small sample CSV for testing
@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    df = pd.DataFrame({
        "source": ["Alice", "Bob", "Charlie", "Alice"],
        "target": ["Bob", "Charlie", "Eve", "Eve"]
    })
    file_path = tmp_path_factory.mktemp("data") / "edges.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_analyze_with_file_path(sample_csv):
    """Test API with file path input"""
    response = client.post("/analyze", data={"file_path": str(sample_csv)})
    assert response.status_code == 200
    result = response.json()

    # Check metrics
    assert "edge_count" in result
    assert "highest_degree_node" in result
    assert "average_degree" in result
    assert "density" in result
    assert "shortest_path_alice_eve" in result

    # Check images are valid base64
    for key in ["network_graph", "degree_histogram"]:
        try:
            base64.b64decode(result[key])
        except Exception:
            pytest.fail(f"{key} is not valid base64")

def test_analyze_with_upload(sample_csv):
    """Test API with file upload"""
    with open(sample_csv, "rb") as f:
        response = client.post("/analyze", files={"file": ("edges.csv", f, "text/csv")})
    assert response.status_code == 200
    result = response.json()

    # Must contain same keys as before
    assert "edge_count" in result
    assert "network_graph" in result
    assert "degree_histogram" in result

def test_missing_file_error():
    """Test API with missing file path"""
    response = client.post("/analyze", data={"file_path": "nonexistent.csv"})
    assert response.status_code == 400
    assert "error" in response.json()

def test_no_input_error():
    """Test API with no input at all"""
    response = client.post("/analyze", data={})
    assert response.status_code == 400
    assert "error" in response.json()
