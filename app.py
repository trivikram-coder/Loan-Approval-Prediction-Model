from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("Prediction Model")

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/")
def health():
    return "ML SERVICE RUNNING SUCCESSFULLY"

@app.post("/predict")
def predict(data: PredictRequest):
    try:
        f = np.array(data.features, dtype=float)

        print("FEATURE COUNT:", len(f))
        print("FEATURE VECTOR:", f)

        # SAFETY CHECKS (PREVENT 500)
        if len(f) != model.n_features_in_:
            raise ValueError(
                f"Expected {model.n_features_in_} features, got {len(f)}"
            )

        if not np.isfinite(f).all():
            raise ValueError("Feature vector contains NaN or Infinity")

        prediction = model.predict([f])[0]

        # Feature mapping (for explanations only)
        credit = f[0]
        applicant_income_log = f[1]
        loan_amount_log = f[2]   # ⚠️ matches your Flask training
        not_graduate = f[10]
        employed_yes = f[11]

        reasons = []

        if prediction == "N":
            if credit == 0:
                reasons.append("No or poor credit history")

            if applicant_income_log < 8:
                reasons.append("Low applicant income")

            if loan_amount_log - applicant_income_log > 1.8:
                reasons.append("Requested loan amount is high compared to income")

            if employed_yes == 0:
                reasons.append("Applicant employment stability is low")

            if not_graduate == 1:
                reasons.append("Applicant education level increases risk profile")

            if not reasons:
                reasons.append("Model assessment indicates higher risk profile")

            return {
                "status": "NOT_ELIGIBLE",
                "message": "Sorry, You are not Eligible to avail loan services",
                "reasons": reasons
            }

        return {
            "status": "ELIGIBLE",
            "message": "Congratulations, You can avail loan services",
            "reasons": []
        }

    except Exception as e:
        print("❌ PREDICT ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
