"""
A lightweight, interactive client that calls CMAPSS RUL prediction API.

Allows users to either choose from existing CMAPSS data or upload a new data

"""

# Fundamental libraries
import argparse
import json
import sys
from pathlib import Path

# ML libraries
import pandas as pd
import requests
import yaml

# Pipeline libarary
from src.data.loader import load_cmapss

DATASET_CHOICES = {"1": "FD001", "2": "FD002", "3": "FD003", "4": "FD004"}
MODEL_CHOICES = {"1": "lstm", "2": "tcn", "3": "transformer", "4": "all"}
