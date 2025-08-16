import sys
import requests
import json

def main(api_url):
    """
    Wrapper for Promptfoo to fetch the latest task and return JSON result.
    """
    try:
        # 1. Get the list of tasks or progress (adjust endpoint if you have a dedicated tasks endpoint)
        response = requests.get(f"{api_url}/progress/latest")
        if response.status_code != 200:
            print(json.dumps({"error": "Failed to get latest task"}))
            return

        data = response.json()
        task_id = data.get("task_id")
        if not task_id:
            print(json.dumps({"error": "No task_id found"}))
            return

        # 2. Fetch full progress/result of the latest task
        progress_resp = requests.get(f"{api_url}/progress/{task_id}")
        if progress_resp.status_code != 200:
            print(json.dumps({"error": "Failed to get task progress"}))
            return

        progress_data = progress_resp.json()
        result = progress_data.get("result", {})

        # 3. Safely extract average_precip_mm
        avg_precip = result.get("average_precip_mm", None)

        output = {"average_precip_mm": avg_precip}
        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "API URL argument required"}))
    else:
        main(sys.argv[1])
