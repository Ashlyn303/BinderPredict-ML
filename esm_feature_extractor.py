import sys
from src import esm_feature_extractor

# Make sure 'esm_feature_extractor' is registered in sys.modules
sys.modules['esm_feature_extractor'] = esm_feature_extractor

# Expose everything to the root namespace
from src.esm_feature_extractor import *
