from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch

from dental_core import LesionAnnotation, PanoramaHiTLDocument
try:
    from .predict import (
        DEFAULT_CROP_CLASS_NAMES,
        map_raw_restoration_to_ssot,
        predict_crop_array,
    )
except ImportError:
    from predict import (
        DEFAULT_CROP_CLASS_NAMES,
        map_raw_restoration_to_ssot,
        predict_crop_array,
    )


def extract_padded_tooth_crop(
    image_bgr: np.ndarray,
    bbox_xyxy: Sequence[float],
    margin_ratio: float = 0.25,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, bbox_xyxy[:4])
    tw = max(4.0, x2 - x1)
    th = max(4.0, y2 - y1)
    px1 = max(0, int(round(x1 - tw * margin_ratio)))
    py1 = max(0, int(round(y1 - th * margin_ratio)))
    px2 = min(w, int(round(x2 + tw * margin_ratio)))
    py2 = min(h, int(round(y2 + th * margin_ratio)))
    if px2 <= px1 or py2 <= py1:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    return image_bgr[py1:py2, px1:px2].copy()


def attribute_detections_to_teeth(
    detections: Sequence[Dict[str, Any]],
    teeth_boxes_fdi: Sequence[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Map spatial restoration detections ({bbox, raw_class/clinical_category, confidence})
    onto Dental_008 teeth ({fdi, bbox_xyxy}) via maximum intersection area & centroid containment.
    """
    per_tooth: Dict[int, List[Dict[str, Any]]] = {}
    for det in detections:
        raw_lbl = str(det.get("clinical_category") or det.get("raw_class") or det.get("type") or "")
        ssot_cat = map_raw_restoration_to_ssot(raw_lbl) or (
            raw_lbl if raw_lbl in ("Inlay/Onlay", "Crown/Bridge", "Implant", "Endodontic (RCT)") else None
        )
        if ssot_cat is None:
            continue
        rbox = det.get("bbox") or det.get("box")
        if not rbox or len(rbox) < 4:
            continue
        rx1, ry1, rx2, ry2 = map(float, rbox[:4])
        rcx, rcy = 0.5 * (rx1 + rx2), 0.5 * (ry1 + ry2)

        best_fdi: Optional[int] = None
        max_inter = 0.0
        for t in teeth_boxes_fdi:
            fdi = t.get("fdi")
            tbox = t.get("bbox_xyxy") or t.get("bbox")
            if fdi is None or not tbox or len(tbox) < 4:
                continue
            tx1, ty1, tx2, ty2 = map(float, tbox[:4])
            ix1, iy1 = max(rx1, tx1), max(ry1, ty1)
            ix2, iy2 = min(rx2, tx2), min(ry2, ty2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter > max_inter:
                    max_inter = inter
                    best_fdi = int(fdi)
            elif max_inter == 0.0 and tx1 <= rcx <= tx2 and ty1 <= rcy <= ty2:
                best_fdi = int(fdi)

        if best_fdi is not None:
            enriched = {
                "fdi": best_fdi,
                "clinical_category": ssot_cat,
                "raw_class": det.get("raw_class", raw_lbl),
                "confidence": float(det.get("confidence", 0.85)),
                "bbox": [round(rx1, 2), round(ry1, 2), round(rx2, 2), round(ry2, 2)],
            }
            per_tooth.setdefault(best_fdi, []).append(enriched)
    return per_tooth


def classify_restorations_for_hitl_doc(
    image_bgr: np.ndarray,
    doc: PanoramaHiTLDocument,
    crop_model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    spatial_detections: Optional[Sequence[Dict[str, Any]]] = None,
    conf_threshold: float = 0.45,
    margin_ratio: float = 0.25,
) -> PanoramaHiTLDocument:
    """
    Populate tooth.restoration_tags and LesionAnnotation(category='restoration') on a
    Dental_Core PanoramaHiTLDocument using:
    1) Individual Dental_008 tooth ROI crops via EfficientNet-B0 (when crop_model is provided), AND/OR
    2) Full-panorama YOLO-seg spatial restoration detections mapped by IoA/centroid.
    """
    eff_device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Preserve non-restoration lesions and clinician-corrected restorations; deduplicate AI entries
    preserved = [
        l for l in doc.lesions if l.category != "restoration" or l.clinician_corrected
    ]
    seen_keys = {
        (l.fdi, l.label, tuple(round(v, 1) for v in l.bbox_xyxy[:4]))
        for l in preserved
        if l.category == "restoration"
    }

    def _append_unique(lesion: LesionAnnotation) -> None:
        key = (lesion.fdi, lesion.label, tuple(round(v, 1) for v in lesion.bbox_xyxy[:4]))
        if key not in seen_keys:
            seen_keys.add(key)
            preserved.append(lesion)

    # 1. Crop-based classification per Dental_008 tooth
    if crop_model is not None:
        for tooth in doc.teeth:
            crop_bgr = extract_padded_tooth_crop(image_bgr, tooth.bbox_xyxy, margin_ratio=margin_ratio)
            pred = predict_crop_array(
                crop_model,
                eff_device,
                crop_bgr,
                class_names=DEFAULT_CROP_CLASS_NAMES,
                is_bgr=True,
            )
            ssot_cat = pred["clinical_category"]
            conf = float(pred["confidence"])
            if ssot_cat is not None and conf >= conf_threshold:
                if ssot_cat not in tooth.restoration_tags:
                    tooth.restoration_tags.append(ssot_cat)
                _append_unique(
                    LesionAnnotation(
                        category="restoration",
                        label=str(ssot_cat),
                        bbox_xyxy=list(tooth.bbox_xyxy),
                        fdi=tooth.fdi,
                        confidence=round(conf, 4),
                        rct_associated=(ssot_cat == "Endodontic (RCT)"),
                        raw_class=str(pred.get("raw_class") or ssot_cat),
                    )
                )

    # 2. Spatial detection attribution (31-class YOLO-seg fusion)
    if spatial_detections:
        teeth_refs = [
            {"instance_id": t.instance_id, "fdi": t.fdi, "bbox_xyxy": t.bbox_xyxy}
            for t in doc.teeth
        ]
        mapped_map = attribute_detections_to_teeth(spatial_detections, teeth_refs)
        id_to_tooth = {t.instance_id: t for t in doc.teeth}
        fdi_to_teeth: Dict[int, List[Any]] = {}
        for t in doc.teeth:
            fdi_to_teeth.setdefault(t.fdi, []).append(t)

        for fdi, items in mapped_map.items():
            candidates = fdi_to_teeth.get(fdi, [])
            for item in items:
                cat = item["clinical_category"]
                target_tooth = None
                if len(candidates) == 1:
                    target_tooth = candidates[0]
                elif len(candidates) > 1:
                    rx1, ry1, rx2, ry2 = item["bbox"]
                    rcx, rcy = 0.5 * (rx1 + rx2), 0.5 * (ry1 + ry2)
                    best_dist = float("inf")
                    for cand in candidates:
                        tx1, ty1, tx2, ty2 = cand.bbox_xyxy
                        d = (rcx - 0.5 * (tx1 + tx2)) ** 2 + (rcy - 0.5 * (ty1 + ty2)) ** 2
                        if d < best_dist:
                            best_dist = d
                            target_tooth = id_to_tooth.get(cand.instance_id, cand)
                if target_tooth is not None and cat not in target_tooth.restoration_tags:
                    target_tooth.restoration_tags.append(cat)
                _append_unique(
                    LesionAnnotation(
                        category="restoration",
                        label=cat,
                        bbox_xyxy=item["bbox"],
                        fdi=fdi,
                        confidence=round(float(item["confidence"]), 4),
                        rct_associated=(cat == "Endodontic (RCT)"),
                        raw_class=str(item.get("raw_class") or cat),
                    )
                )

    doc.lesions = preserved
    return doc
