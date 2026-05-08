from __future__ import annotations

try:
    from fastapi import FastAPI
except ModuleNotFoundError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]

if FastAPI is not None:
    app = FastAPI(title="CerebRL API")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}


def main() -> None:
    print("CerebRL API placeholder. Install fastapi and uvicorn to serve the app.")


if __name__ == "__main__":
    main()

