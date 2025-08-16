import requests
import base64
import argparse
import os
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Client for Network Analysis API")
    parser.add_argument("--file", required=True, help="Path to CSV file containing edges")
    parser.add_argument("--url", default="http://127.0.0.1:8000/analyze", help="API endpoint URL")
    parser.add_argument("--outdir", default="results", help="Directory to save outputs")
    args = parser.parse_args()

    Path(args.outdir).mkdir(exist_ok=True)

    data = {"file_path": args.file}
    try:
        response = requests.post(args.url, json=data, timeout=60)
    except requests.exceptions.RequestException as e:
        print("❌ Failed to connect to API:", e)
        return

    if response.status_code == 200:
        try:
            result = response.json()
        except json.JSONDecodeError:
            print("❌ Invalid JSON response from server")
            return

        print("\n📊 Network Metrics:")
        for key in ["edge_count", "highest_degree_node", "average_degree", "density", "shortest_path_alice_eve"]:
            print(f"{key}: {result.get(key, 'N/A')}")

        # Save images
        for name in ["network_graph", "degree_histogram"]:
            if name in result:
                file_path = Path(args.outdir) / f"{name}.png"
                with open(file_path, "wb") as f:
                    f.write(base64.b64decode(result[name]))
                print(f"✅ Saved {file_path}")
            else:
                print(f"⚠️ {name} not found in response")

    else:
        print("❌ Error:", response.status_code, response.text)


if __name__ == "__main__":
    main()

















# import requests
# import base64

# url = "http://127.0.0.1:8000/analyze"
# data = {"file_path": "edges.csv"}

# response = requests.post(url, json=data)

# if response.status_code == 200:
#     result = response.json()
#     print("Network Metrics:")
#     print(f"Edge count: {result['edge_count']}")
#     print(f"Highest degree node: {result['highest_degree_node']}")
#     print(f"Average degree: {result['average_degree']}")
#     print(f"Density: {result['density']}")
#     print(f"Shortest path Alice→Eve: {result['shortest_path_alice_eve']}")
    
#     # Save images
#     with open("network_graph.png", "wb") as f:
#         f.write(base64.b64decode(result["network_graph"]))
#     with open("degree_histogram.png", "wb") as f:
#         f.write(base64.b64decode(result["degree_histogram"]))
#     print("Network graph and degree histogram saved as PNGs.")
# else:
#     print("Error:", response.text)
