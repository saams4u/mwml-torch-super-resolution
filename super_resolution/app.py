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

from opt_for_train import *


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


@app.get("/experiments")
async def _experiments():
    return RedirectResponse("https://app.wandb.ai/mahjouri-saamahn/mwml-torch-super-resolution")


class SuperResolvePayload(BaseModel):
    experiment_id: str = 'latest'
    inputs: list = [{"image": "leon.png"}]


@utils.construct_response
@app.post("/super_resolve")
async def _super_resolve(payload: SuperResolvePayload):
    super_resolution = super_resolve(model=Net, inputs=payload.inputs, cuda=False)

    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {"super_resolution": super_resolution}
    }
    config.logger.info(json.dumps(response, indent=2))

    return response
