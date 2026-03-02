import sys
from src import data_loader

# Make sure 'data_loader' is registered in sys.modules
sys.modules['data_loader'] = data_loader

# Expose everything to the root namespace
from src.data_loader import *
