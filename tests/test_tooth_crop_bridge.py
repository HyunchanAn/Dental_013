import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dental_core import PanoramaHiTLDocument, PanoramaImageMeta, ToothAnnotation
from predict import DEFAULT_CROP_CLASS_NAMES, map_raw_restoration_to_ssot
from tooth_crop_bridge import (
    attribute_detections_to_teeth,
    classify_restorations_for_hitl_doc,
)


class _DeterministicCrownClassifier(nn.Module):
    """Test double that outputs high confidence for index 1 ('Crown')."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((x.size(0), len(DEFAULT_CROP_CLASS_NAMES)), -5.0, device=x.device)
        logits[:, 1] = 8.0  # 'Crown' -> 'Crown/Bridge'
        return logits


def test_ssot_category_mapping():
    assert map_raw_restoration_to_ssot("Crown") == "Crown/Bridge"
    assert map_raw_restoration_to_ssot("Bridge") == "Crown/Bridge"
    assert map_raw_restoration_to_ssot("Filling") == "Inlay/Onlay"
    assert map_raw_restoration_to_ssot("Implant") == "Implant"
    assert map_raw_restoration_to_ssot("RCT") == "Endodontic (RCT)"
    assert map_raw_restoration_to_ssot("Natural") is None


def test_008_crop_and_spatial_restoration_fusion_to_015_ssot():
    pano = np.full((800, 1600, 3), 110, dtype=np.uint8)
    doc = PanoramaHiTLDocument(
        image_meta=PanoramaImageMeta(filename="pano_013.png", width=1600, height=800, sha256_hash="b" * 64),
        teeth=[
            ToothAnnotation(fdi=36, bbox_xyxy=[950.0, 420.0, 1040.0, 620.0]),
            ToothAnnotation(fdi=46, bbox_xyxy=[520.0, 420.0, 610.0, 620.0]),
        ],
    )

    spatial_dets = [
        {
            "raw_class": "Root Canal Treatment",
            "confidence": 0.89,
            "bbox": [965, 480, 1020, 610],
        },
        {
            "raw_class": "Implant",
            "confidence": 0.94,
            "bbox": [535, 450, 595, 615],
        },
    ]

    mapped = attribute_detections_to_teeth(
        spatial_dets,
        [{"fdi": t.fdi, "bbox_xyxy": t.bbox_xyxy} for t in doc.teeth],
    )
    assert mapped[36][0]["clinical_category"] == "Endodontic (RCT)"
    assert mapped[46][0]["clinical_category"] == "Implant"

    enriched = classify_restorations_for_hitl_doc(
        pano,
        doc,
        crop_model=_DeterministicCrownClassifier(),
        device=torch.device("cpu"),
        spatial_detections=spatial_dets,
        conf_threshold=0.5,
    )

    # #36 should have both Crown/Bridge (from crop model) and Endodontic (RCT) (from spatial fusion)
    assert "Crown/Bridge" in enriched.teeth[0].restoration_tags
    assert "Endodontic (RCT)" in enriched.teeth[0].restoration_tags
    # #46 should have Crown/Bridge + Implant
    assert "Implant" in enriched.teeth[1].restoration_tags

    ssot = enriched.to_ssot_findings()
    assert "Endodontic (RCT)" in ssot["teeth"][0]["restorationTags"]
    assert any(r["toothNumber"] == 46 and r["type"] == "Implant" for r in ssot["restorations"])
