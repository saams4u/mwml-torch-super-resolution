import os
import json
import config, utils

from pydantic import BaseModel
from fastapi import FastAPI, Path
from fastapi.responses import RedirectResponse
from http import HTTPStatus


app = FastAPI(
    title="super_resolution",
    description="",
    version="1.0.0",
)


@utils.construct_response
@app.get("/")
async def _index():
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {}
    }
    config.logger.info(json.dumps(response, indent=2))
    return response


class PredictPayload(BaseModel):
    pass


@utils.construct_response
@app.post("/predict")
async def _predict(payload: PredictPayload):
    pass
