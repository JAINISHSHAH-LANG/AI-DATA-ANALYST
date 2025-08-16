import requests
import base64

url = "http://127.0.0.1:8000/analyze"
data = {"file_path": "edges.csv"}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("Network Metrics:")
    print(f"Edge count: {result['edge_count']}")
    print(f"Highest degree node: {result['highest_degree_node']}")
    print(f"Average degree: {result['average_degree']}")
    print(f"Density: {result['density']}")
    print(f"Shortest path Alice→Eve: {result['shortest_path_alice_eve']}")
    
    # Save images
    with open("network_graph.png", "wb") as f:
        f.write(base64.b64decode(result["network_graph"]))
    with open("degree_histogram.png", "wb") as f:
        f.write(base64.b64decode(result["degree_histogram"]))
    print("Network graph and degree histogram saved as PNGs.")
else:
    print("Error:", response.text)
