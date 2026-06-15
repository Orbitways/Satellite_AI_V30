from fastapi import FastAPI

app = FastAPI(
    title="Orbitways Insurer API",
    description="Prototype API for underwriting-oriented orbital risk assessment.",
    version="0.1.0",
)

@app.get("/")
def root():
    return {
        "service": "Orbitways Insurer API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Orbitways Insurer API",
    }