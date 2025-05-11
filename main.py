from flask import Flask, request, jsonify
import joblib
import pandas as pd

# --- Caricamento modello ---
MODEL_PATH = 'soccer_ai_pro_final.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modello caricato: {MODEL_PATH}")
    print("Feature usate dal modello:", model.feature_names_in_)
except Exception as e:
    print(f"❌ Errore caricamento modello: {e}")
    model = None

# --- Mappatura delle classi ---
class_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model non disponibile"}), 500
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Seleziona e ordina le colonne esattamente come richiesto
        features = list(model.feature_names_in_)
        df = df[features]

        # Predizione e probabilità
        preds_mapped = model.predict(df)[0]  # 0,1,2
        probs = model.predict_proba(df)[0]

        # Rimappping back a -1,0,1
        inverse_map = {0: -1, 1: 0, 2: 1}
        pred_code = inverse_map[preds_mapped]
        pred_label = class_map[preds_mapped]

        return jsonify({
            "prediction_code": pred_code,
            "prediction_label": pred_label,
            "probabilities": {
                class_map[i]: float(p)
                for i, p in zip(model.classes_, probs)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # In prod usa un WSGI server; qui va bene per test
    app.run(host='0.0.0.0', port=5000)
