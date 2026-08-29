# Integration Package
__version__ = "1.0.0"

def get_integrated_pipeline(*args, **kwargs):
    from .phase2_phase3_pipeline import IntegratedPipeline
    return IntegratedPipeline(*args, **kwargs)
