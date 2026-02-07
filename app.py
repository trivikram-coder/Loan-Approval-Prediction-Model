from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load trained ML model
model = joblib.load("Prediction Model")

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/")
def health():
    return "ML SERVICE RUNNING SUCCESSFULLY"

@app.post("/predict")
def predict(data: PredictRequest):

    print("FEATURE VECTOR:", data.features)

    f = data.features
    prediction = model.predict([f])[0]

    # Feature mapping
    credit = f[0]
    applicant_income_log = f[1]
    loan_amount_log = f[3]
    not_graduate = f[10]
    employed_yes = f[11]

    reasons = []

    # 🔴 ONLY ADD REASONS IF MODEL REJECTS
    if prediction == "N":

        if credit == 0:
            reasons.append("No or poor credit history")

        # income < ~₹3000/month (log ≈ 8)
        if applicant_income_log < 8:
            reasons.append("Low applicant income")

        # loan > income * ~6 (more realistic)
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

    # ✅ ELIGIBLE CASE — NO NEGATIVE REASONS
    return {
        "status": "ELIGIBLE",
        "message": "Congratulations, You can avail loan services",
        "reasons": []
    }


    # DEBUG: print received features
    print("FEATURE VECTOR:", data.features)

    f = data.features

    # ML prediction
    prediction = model.predict([f])[0]

    reasons = []

    # ✅ CORRECT FEATURE MAPPING (VERY IMPORTANT)
    credit = f[0]
    applicant_income_log = f[1]
    loan_amount_log = f[3]
    print("Loan : ",loan_amount_log)        # ✅ FIXED INDEX
    not_graduate = f[10]
    employed_yes = f[11]

    # Rule-based explanation
    if credit == 0:
        reasons.append("No or poor credit history")

    if applicant_income_log < 8:
        reasons.append("Low applicant income")

    if loan_amount_log > 5.8:
        reasons.append("Requested loan amount is high")

    if employed_yes == 0:
        reasons.append("Applicant is not self-employed")

    if not_graduate == 1:
        reasons.append("Applicant is not a graduate")

    # Final decision
    if prediction == "N":
        if not reasons:
            reasons.append("Model assessment indicates higher risk profile")
        return {
            "status": "NOT_ELIGIBLE",
            "message": "Sorry, You are not Eligible to avail loan services",
            "reasons": reasons
        }


    return {
        "status": "ELIGIBLE",
        "message": "Congratulations, You Can avail loan services",
        "reasons": []
    }
