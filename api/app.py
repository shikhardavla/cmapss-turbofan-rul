from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictionRequest, PredictionResponse
from api.inference import predict_rul, WINDOW_SIZE

app = FastAPI(
    title="Turbofan RUL Prediction API",
    description="Predicts Remaining Useful Life (RUL) for CMAPSS turbofan engines.",
    version="1.0.0",
)

# Allows a dashboard served on a different origin/port to call this API.
# Tighten allow_origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if len(request.cycles) != WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected exactly {WINDOW_SIZE} cycles "
                f"(config.features.window_size), got {len(request.cycles)}."
            ),
        )

    try:
        predicted_rul = predict_rul(
            dataset=request.dataset,
            model_name=request.model,
            cycles=request.cycles,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return PredictionResponse(
        dataset=request.dataset,
        model=request.model,
        unit_id=request.unit_id,
        predicted_rul=predicted_rul,
    )