from flask import Blueprint, request, jsonify
import os
import uuid
import pandas as pd

dataset_routes = Blueprint("dataset_routes", __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _read_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


@dataset_routes.route("/datasets", methods=["POST"])
def upload_dataset():
    """
    Upload a dataset file.
    The dataset must contain both feature columns AND the prediction/score column.
    Returns: dataset_id, records, columns
    """
    if "file" not in request.files:
        return jsonify({"error": "No dataset file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    dataset_id = str(uuid.uuid4())
    filename   = f"{dataset_id}_{file.filename}"
    path       = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        df = _read_dataframe(path)
    except ValueError as e:
        os.remove(path)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        os.remove(path)
        return jsonify({"error": f"Failed to parse file: {str(e)}"}), 400

    records = len(df)
    columns = list(df.columns)

    return jsonify({
        "dataset_id": dataset_id,
        "file_path":  path,
        "records":    records,
        "columns":    columns,
        "message":    "Dataset uploaded successfully"
    }), 201


@dataset_routes.route("/models", methods=["POST"])
def create_model():
    """
    Register a model record.
    Schema: Model { id, name, version, status, userId }
    Called by frontend before /evaluate — returns model_id.
    """
    data    = request.json or {}
    name    = data.get("name", "Unnamed Model")
    version = data.get("version", "1.0.0")

    model_id = str(uuid.uuid4())

    return jsonify({
        "model_id": model_id,
        "name":     name,
        "version":  version,
        "status":   "ACTIVE",
        "message":  "Model registered"
    }), 201