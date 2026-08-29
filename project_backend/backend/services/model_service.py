"""
Model Registry & Service Manager.

Manages cached initialization of upstream models across Phases 4 to 11
and provides device allocation and health diagnostic status.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import torch

from phase_11_report_generator.pipeline import EndToEndReportPipeline
from phase_11_report_generator.config import ReportConfig
from phase_10_explainability.config import ExplainabilityConfig
from phase_10_explainability.explainability_engine import MultimodalExplainabilityEngine
from ..core.config import settings
from ..core.logging_config import logger


class ModelManager:
    """
    Singleton manager for deep learning pipelines and inference models.
    """
    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        logger.info(f"Initializing ModelManager on device: {self.device}")
        
        # Configure Explainability and Report Engines
        self.exp_config = ExplainabilityConfig(
            device=str(self.device),
            save_visualizations=True,
            shap_background_samples=10,
            mc_samples=10,
        )
        self.exp_engine = MultimodalExplainabilityEngine(
            config=self.exp_config,
            device=str(self.device),
        )
        
        self.report_config = ReportConfig(
            output_dir=str(settings.REPORTS_DIR),
        )
        self.pipeline = EndToEndReportPipeline(
            report_config=self.report_config,
            explainability_engine=self.exp_engine,
            device=str(self.device),
        )
        logger.info("ModelManager initialized successfully.")

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def get_model_status(self) -> Dict[str, str]:
        """
        Verify status of all pipeline components.
        """
        status = {
            "phase4_octa": "loaded" if "octa" in self.exp_engine.phase7_pipeline.retinal_feature_extractor.models else "missing",
            "phase4_octb": "loaded" if "octb" in self.exp_engine.phase7_pipeline.retinal_feature_extractor.models else "missing",
            "phase4_fundus": "loaded" if "fundus" in self.exp_engine.phase7_pipeline.retinal_feature_extractor.models else "missing",
            "phase5": "loaded" if self.exp_engine.phase7_pipeline.retinal_fusion_model is not None else "missing",
            "phase6": "loaded" if self.exp_engine.phase7_pipeline.clinical_extractor is not None else "missing",
            "phase7": "loaded" if self.exp_engine.phase7_pipeline.model is not None else "missing",
            "phase8": "loaded" if self.exp_engine.model is not None else "missing",
            "phase9": "available",
            "phase10": "available",
            "phase11": "available",
            "device": str(self.device),
        }
        return status
