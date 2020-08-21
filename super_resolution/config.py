import os
import json 

import logging
import logging.config


def create_dirs(dirpath):
    """Creating directories."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_json(filepath):
    """Load a json file."""
    with open(filepath, "r") as fp:
        json_obj = json.load(fp)
    return json_obj


# Directories
BASE_DIR = os.getcwd()  # project root
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')

# Create dirs
utils.create_dirs(LOGS_DIR)
utils.create_dirs(EXPERIMENTS_DIR)

# Loggers
log_config = utils.load_json(
    filepath=os.path.join(BASE_DIR, 'logging.json'))
logging.config.dictConfig(log_config)
logger = logging.getLogger('logger')
