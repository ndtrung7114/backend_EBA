"""
EBA Building Genome Web — FastAPI Application
================================================
Energy Baseline Adjustment — Normalized Usage with ElasticNet.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.analysis import router as analysis_router

app = FastAPI(
    title="EBA Building Genome — Energy Baseline API",
    description=(
        "Normalized Usage baseline analysis using ElasticNet regression. "
        "Building Genome Dataset with 15 representative electricity meters."
    ),
    version="1.0.0",
)

# CORS — allow the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router)


@app.get("/")
def root():
    return {
        "service": "EBA Building Genome — Energy Baseline API",
        "version": "1.0.0",
        "model": "ElasticNet (auto-tuned)",
        "meters": 15,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
