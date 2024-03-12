import os

from .utils.ui import abspath


DATA_DIR = abspath(os.environ.get('DATA_DIR', os.path.join(os.path.dirname(__file__), '../../data')))




