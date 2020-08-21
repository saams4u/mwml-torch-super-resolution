import os
import sys

sys.path.append(".")

from fastapi import FastAPI, Path
from fastapi.responses import RedirectResponse

from http import HTTPStatus
from pydantic import BaseModel

import wandb
import json

import config, data, eval, utils


app = FastAPI(
    title="super_resolution",
    description="PyTorch Super Resolution Using Made With ML",
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


class SuperResolvePayload(BaseModel):
    pass


@utils.construct_response
@app.post("/super_resolve")
async def _super_resolve(payload: SuperResolvePayload):
    pass
