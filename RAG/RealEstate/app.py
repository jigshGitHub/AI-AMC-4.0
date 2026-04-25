from __future__ import annotations

import sys
from pathlib import Path
import os
from flask import Flask, request, jsonify, render_template

# Ensure repository's RealEstate package is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

try:
    # import the agent module from the RealEstate folder
    from agent import query_rag
except Exception:
    # fallback: try package-style import if running from repo root
    from RAG.RealEstate.agent import query_rag


app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"success": False, "error": "Empty question"}), 400

    try:
        result = query_rag(question)
        # result is expected to be a dict with final_answer and retrieved_sources
        return jsonify({"success": True, "answer": result.get("final_answer", ""), "sources": result.get("retrieved_sources", "")})
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Use port 8501 by default to avoid conflicts
    port = int(os.getenv("PORT", 8501))
    app.run(host="0.0.0.0", port=port, debug=True)
