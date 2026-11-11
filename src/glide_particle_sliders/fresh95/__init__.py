"""Explicit cached GLiDE normalized readout policy; no old cohort migration."""
PROFILE='glide-normalized-readout-explicit-cached-fresh95-v1'
def open_training(*args,**kwargs):
    from .api import Training
    return Training(*args,**kwargs)
def open_inference(*args,**kwargs):
    from .api import Inference
    return Inference(*args,**kwargs)
def save_prepared_cache(*args,**kwargs):
    from .cache import save_prepared_cache as save
    return save(*args,**kwargs)
from .util import Checkpoint,file_pin
__all__=['PROFILE','open_training','open_inference','save_prepared_cache','Checkpoint','file_pin']
