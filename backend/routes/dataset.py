from flask import Blueprint, request, jsonify
import os
import uuid
import pandas as pd
import tempfile
import traceback
import google.generativeai as genai
    
from flask import send_file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key="AIzaSyAaj7-O56M5bS7ZzFCIUd9AGKgOy3MW3Uo")

dataset_routes = Blueprint("dataset_routes", __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def _find_file(dataset_id: str) -> str:
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.startswith(dataset_id):
            return os.path.join(UPLOAD_FOLDER, fname)
    raise FileNotFoundError

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
    
@dataset_routes.route("/datasets/<dataset_id>/filter-preview", methods=["POST"])
def filter_preview(dataset_id: str):
    """
    POST /datasets/<dataset_id>/filter-preview
    Body: { filter_code: str }
    
    Applies the pandas filter_code to the dataset and returns row counts.
    Does NOT save anything — purely a preview.
    """
    body = request.get_json(silent=True) or {}
    filter_code = body.get("filter_code", "").strip()

    if not filter_code:
        return jsonify({"error": "filter_code is required"}), 400

    if not filter_code.startswith("df_filtered"):
        return jsonify({"error": "filter_code must assign to df_filtered"}), 400

    # Find the uploaded file
    try:
        path = _find_file(dataset_id)
    except FileNotFoundError:
        return jsonify({"error": f"Dataset '{dataset_id}' not found"}), 404

    try:
        df = _read_dataframe(path)
        rows_before = len(df)

        # Execute the filter safely
        local_ns = {"df": df.copy(), "pd": pd}
        exec(filter_code, {"__builtins__": {}}, local_ns)
        df_filtered = local_ns.get("df_filtered")

        if df_filtered is None:
            return jsonify({"error": "Filter did not produce df_filtered"}), 400

        rows_after = len(df_filtered)

        # Value counts for the most-filtered column
        # Detect which column was filtered by comparing unique values
        value_counts = []
        if rows_after > 0:
            # Find columns with reduced unique values (likely the filtered column)
            for col in df_filtered.columns:
                if df_filtered[col].nunique() < df[col].nunique():
                    vc = df_filtered[col].value_counts().head(3)
                    value_counts = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
                    break
            # Fallback: first categorical column
            if not value_counts:
                for col in df_filtered.select_dtypes(include="object").columns[:1]:
                    vc = df_filtered[col].value_counts().head(3)
                    value_counts = [{"value": str(k), "count": int(v)} for k, v in vc.items()]

        # Suggestion if 0 rows
        suggestion = None
        if rows_after == 0:
            # Try to detect the column and show available values
            try:
                col_name = filter_code.split("['")[1].split("'")[0] if "'['" in filter_code else None
                if col_name and col_name in df.columns:
                    sample = df[col_name].value_counts().head(5)
                    suggestion = f"Column '{col_name}' has values: {', '.join(str(v) for v in sample.index)}. Try one of these."
            except Exception:
                suggestion = "No rows matched. Check the column name and value spelling."

        return jsonify({
            "rows_before":   rows_before,
            "rows_after":    rows_after,
            "pct_retained":  round(rows_after / rows_before * 100, 1) if rows_before else 0,
            "value_counts":  value_counts,
            "suggestion":    suggestion,
        }), 200

    except SyntaxError as e:
        return jsonify({"error": f"Syntax error in filter code: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Filter error: {str(e)}", "traceback": traceback.format_exc()}), 400


@dataset_routes.route("/datasets/<dataset_id>/filter-apply", methods=["POST"])
def filter_apply(dataset_id: str):
    """
    POST /datasets/<dataset_id>/filter-apply
    Body: { filter_code: str, description: str }
    
    Applies the filter, saves the filtered dataset as a new file, 
    and returns a new filtered_dataset_id to use for evaluation.
    """
    body = request.get_json(silent=True) or {}
    filter_code = body.get("filter_code", "").strip()
    description = body.get("description", "filtered subset")

    if not filter_code or not filter_code.startswith("df_filtered"):
        return jsonify({"error": "filter_code is required and must assign to df_filtered"}), 400

    try:
        path = _find_file(dataset_id)
    except FileNotFoundError:
        return jsonify({"error": f"Dataset '{dataset_id}' not found"}), 404

    try:
        df = _read_dataframe(path)
        rows_before = len(df)

        local_ns = {"df": df.copy(), "pd": pd}
        exec(filter_code, {"__builtins__": {}}, local_ns)
        df_filtered = local_ns.get("df_filtered")

        if df_filtered is None or len(df_filtered) == 0:
            return jsonify({"error": "Filter produced 0 rows — cannot evaluate empty dataset"}), 400

        rows_after = len(df_filtered)

        # Save filtered dataset as new CSV
        filtered_id       = str(uuid.uuid4())
        original_filename = os.path.basename(path).split("_", 1)[-1]  # strip old uuid prefix
        filtered_filename = f"{filtered_id}_filtered_{original_filename.replace('.xlsx', '.csv').replace('.xls', '.csv').replace('.json', '.csv')}"
        if not filtered_filename.endswith('.csv'):
            filtered_filename += '.csv'
        filtered_path = os.path.join(UPLOAD_FOLDER, filtered_filename)

        df_filtered.to_csv(filtered_path, index=False)

        return jsonify({
            "filtered_dataset_id": filtered_id,
            "original_dataset_id": dataset_id,
            "rows_before":         rows_before,
            "rows_after":          rows_after,
            "pct_retained":        round(rows_after / rows_before * 100, 1),
            "description":         description,
            "file_path":           filtered_path,
            "columns":             list(df_filtered.columns),
            "message":             f"Filter applied: {description}. {rows_after} rows saved.",
        }), 200

    except SyntaxError as e:
        return jsonify({"error": f"Syntax error in filter code: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Filter apply error: {str(e)}"}), 400

@dataset_routes.route("/datasets/<dataset_id>/download", methods=["GET"])
def download_dataset(dataset_id: str):
    """
    Download dataset file (original OR filtered)
    """

    try:
        path = _find_file(dataset_id)
    except FileNotFoundError:
        return jsonify({"error": f"Dataset '{dataset_id}' not found"}), 404

    try:
        filename = os.path.basename(path)

        return send_file(
            path,
            as_attachment=True,
            download_name=filename,
            mimetype="text/csv"
        )

    except Exception as e:
        return jsonify({
            "error": f"Download failed: {str(e)}"
        }), 500
        
@dataset_routes.route("/datasets/<dataset_id>/generate-filter", methods=["POST"])
def generate_filter(dataset_id: str):
    """
    Generate pandas filter code using Gemini
    """
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        path = _find_file(dataset_id)
        df = _read_dataframe(path)
        columns = list(df.columns)

        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(f"""
You are a pandas expert.

Dataset columns:
{columns}

User instruction:
"{prompt}"

Return ONLY one line of code.

Rules:
- Start with: df_filtered = df[
- Use ONLY columns from the list
- Use .str.lower() for strings
- No explanation
""")

        code = response.text.strip()

        # 🔒 safety
        if not code.startswith("df_filtered"):
            return jsonify({"error": "Invalid code generated"}), 400

        return jsonify({
            "code": code
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Gemini failed: {str(e)}"
        }), 500