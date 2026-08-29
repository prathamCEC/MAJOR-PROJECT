"""
End-to-End Multimodal Explainability Pipeline.

High-level interface coordinating Phases 4 through 10 into an explainable patient profile.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

from .config import ExplainabilityConfig, get_default_explainability_config
from .explainability_engine import MultimodalExplainabilityEngine


class EndToEndExplainabilityPipeline:
    """
    High-level explainability pipeline preparing complete diagnostic artifacts for Phase 11.
    """

    def __init__(
        self,
        config: Optional[ExplainabilityConfig] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        background_clinical_df: Optional[pd.DataFrame] = None,
        device: str = "auto",
    ):
        self.config = config or get_default_explainability_config()
        self.engine = MultimodalExplainabilityEngine(
            config=self.config,
            checkpoint_path=checkpoint_path,
            background_clinical_df=background_clinical_df,
            device=device,
        )

    def run_patient_explanation(
        self,
        patient_id: str,
        retinal_scans: Dict[str, Union[str, Path]],
        clinical_record: Dict[str, Any],
        save_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """
        Run end-to-end explainability suite and output structured explanation bundle.
        """
        return self.engine.explain_patient(
            patient_id=patient_id,
            retinal_scans=retinal_scans,
            clinical_record=clinical_record,
            save_plots=save_visualizations,
        )
