# [ 261125 1530 랩탑 개발팀 ] (Dental_013#4 tooth_crop_bridge 정합화 및 export_onnx/README 수정 계획서)

## 1. 대상 이슈 및 위치

- 대상 이슈: [Dental_013#4](https://github.com/HyunchanAn/Dental_013/issues/4) (`[P1][브리지] tooth_crop_bridge.py 중복 적재/과잉치 키 덮어쓰기 방지, export_onnx.py 및 README 수정`)
- 상위 Umbrella: [Dental_000#18](https://github.com/HyunchanAn/Dental_000/issues/18) (`Dental_000/plans_replies_walkthrough/261125_1530_Dental_Series_Remediation_Umbrella_Plan.md` Track 2)
- 선행 권장: `Dental_Core#7` (`rct_associated`, `rawClass` 라운드트립 보존)

---

## 2. 즉시 실행 가능 작업 (`[READY]`)

### 2.1 `src/tooth_crop_bridge.py` 중복 적재·키 덮어쓰기·임포트 충돌 수정
- 대상 라인: `6`, `116-147`
- 작업 내용:
  - `doc.lesions`에 무조건 `append`하여 재실행 또는 HiTL 병합 시 동일 수복물이 중복 적재되는 버그 수정.
  - `fdi_to_tooth` 딕셔너리 키를 단일 `fdi`에서 `instance_id` 기반으로 보완하여 동일 FDI 과잉치 복수 존재 시 덮어쓰기 방지.
  - 범용 top-level `from predict import ...`로 인해 다중 리포지토리 통합 시 `sys.path` 충돌이 발생하는 구조 개선(상대 임포트 또는 명시적 패키지/모듈 경로 로드).

### 2.2 `export_onnx.py` 및 `README.md` 현행화
- `export_onnx.py:9-11`: 실제 모델(`models/best_restoration_model.pth`, EfficientNet-B0 7-class 분류기)과 불일치하는 `MODEL_TYPE = "yolo"` 및 `REPO_ID`를 실제 아키텍처와 네임스페이스에 맞게 수정.
- `README.md`:
  - L1 미존재 `UI-Streamlit` 배지 제거.
  - L35-36 디렉토리 트리에 `tooth_crop_bridge.py`, `test_tooth_crop_bridge.py` 추가.
  - L45 `reports_archive/` -> `reports/` 경로 정정.
  - L79 전체 테스트 실행 명령 반영 및 Python 배지/CI 버전 통일.

---

## 3. `[FREEZE]` 보류 항목

- 없음 (전체 항목 코드·테스트·문서 수준에서 즉시 해결 가능).

---

## 4. 검증 기준

- `tests/test_tooth_crop_bridge.py`에 중복 적재 방지 및 복수 과잉치 `instance_id` 보존 케이스 추가 후 `pytest tests -v` 전수 통과.
- 원격 GitHub Actions CI 통과 확인.
