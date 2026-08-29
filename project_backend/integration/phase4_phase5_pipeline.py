"""
Integration Pipeline: Phase 4 (Swin Transformer Feature Extraction) -> Phase 5 (DMRA Multimodal Retinal Fusion).

Coordinates feature extraction from Phase 4 modality backbones and passes them
through Phase 5 Dynamic Modality Reliability Attention and Cross-Attention Fusion
to produce Unified Retinal Representations (URR) ready for Phase 6 / Phase 7.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch

from phase_5_retinal_fusion.config import get_default_fusion_config, get_fusion_outputs_dir
from phase_5_retinal_fusion.fusion_model import RetinalMultimodalFusionModel
from phase_5_retinal_fusion.feature_loader import Phase4FeatureExtractor


class Phase4ToPhase5Integrator:
    """
    Automates multimodal fusion across available patient scans.
    """

    def __init__(
        self,
        phase4_checkpoints: Optional[Dict[str, Union[str, Path]]] = None,
        fusion_config: Optional[Any] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.config = fusion_config or get_default_fusion_config()
        self.config.device = device

        self.extractor = Phase4FeatureExtractor(
            checkpoints=phase4_checkpoints,
            device=device,
            pretrained_backbone=False,
        )
        self.fusion_model = RetinalMultimodalFusionModel(config=self.config)
        self.fusion_model.to(torch.device(device))
        self.fusion_model.eval()

    def fuse_patient_scans(
        self,
        patient_id: str,
        scans: Dict[str, Union[str, Path]],
    ) -> Dict[str, Any]:
        """
        Fuse available retinal modalities for a single patient into a URR vector.
        """
        feats, mask = self.extractor.extract_multimodal_patient_features(scans, pool=False)
        if not feats:
            raise ValueError(f"No valid scans found for patient '{patient_id}'.")

        with torch.no_grad():
            res = self.fusion_model(modality_features=feats, modality_mask=mask)

        return {
            "patient_id": patient_id,
            "urr": res["urr"].cpu(),
            "modality_weights": {m: w.cpu().item() for m, w in res["modality_weights"].items()},
            "active_modalities": list(feats.keys()),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 -> Phase 5 Retinal Fusion Pipeline")
    parser.add_argument("--octa", type=str, default=None, help="Path to OCT-A scan")
    parser.add_argument("--octb", type=str, default=None, help="Path to OCT-B scan")
    parser.add_argument("--fundus", type=str, default=None, help="Path to Fundus scan")
    parser.add_argument("--patient-id", type=str, default="PATIENT_001")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    scans = {}
    if args.octa:
        scans["octa"] = args.octa
    if args.octb:
        scans["octb"] = args.octb
    if args.fundus:
        scans["fundus"] = args.fundus

    if not scans:
        print("Please provide at least one scan path (--octa, --octb, or --fundus).")
        return

    integrator = Phase4ToPhase5Integrator()
    result = integrator.fuse_patient_scans(args.patient_id, scans)

    print("\n============================================================")
    print(f"PATIENT {result['patient_id']} RETINAL FUSION (URR)")
    print("============================================================")
    print(f"Active Modalities      : {result['active_modalities']}")
    print(f"Unified Retinal Vector : Shape {tuple(result['urr'].shape)}")
    print("Modality Reliability Weights:")
    for m, w in result["modality_weights"].items():
        print(f"  - {m.upper():<8}: {w:.4f} ({w * 100:.2f}%)")

    out_p = args.output or (get_fusion_outputs_dir() / f"{result['patient_id']}_urr.pt")
    torch.save(result, str(out_p))
    print(f"Exported Patient URR to: {out_p}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
