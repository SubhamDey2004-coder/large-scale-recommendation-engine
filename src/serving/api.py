
from fastapi import FastAPI, HTTPException, Query

from src.serving.recommend_embeddings import recommend

app = FastAPI(
    title="Nexvia Personalized Recommendation Engine",
    description="Two-stage movie retrieval and ranking API.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "Nexvia Recommendation Engine API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/recommend/{user_id}")
def get_recommendations(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
):
    try:
        results = recommend(user_id, top_k=top_k)
        return {
            "user_id": user_id,
            "count": len(results),
            "recommendations": results.to_dict(orient="records"),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
