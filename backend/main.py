# pylint: disable=missing-function-docstring

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi_limiter import FastAPILimiter

from app.routers.authentication import authentication
from app.routers.nse.derivatives import derivatives
from app.routers.nse.equity import equity
from app.routers.smartapi.smartapi import smartapi
from app.utils.common.logger import get_logger
from app.utils.redis_utils import init_redis_client
from app.utils.startup_utils import create_tokens_db

logger = get_logger(Path(__file__).name)


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    try:
        create_tokens_db()
        redis_client = await init_redis_client()
        await FastAPILimiter.init(redis_client)
    except Exception as e:
        logger.error("Failed to initialize startup event: %s", str(e))
        raise


app.include_router(derivatives.router)
app.include_router(equity.router)
app.include_router(smartapi.router)
app.include_router(authentication.router)


@app.get("/", response_model=str)
def index():
    return "This is main page"


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
