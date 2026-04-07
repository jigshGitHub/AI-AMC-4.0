import sys
import json
print("Hello, World!")
print(f"Python version: {sys.version}")
print(f"Current working directory: {sys.path[0]}")

sessionInfo = {
    "course": "AI-AMC-4.0",
    "batch": 4.0,
    "instructor": "Mr Kadam"
}

with open("session_info.json", "w") as f:
    json.dump(sessionInfo, f, indent=4)

print("Session Information:")
with open("session_info.json", "r") as f:
    loaded_session_info = json.load(f)

print(json.dumps(loaded_session_info, indent=4))
print("Verification complete. You are ready to start the course!")
