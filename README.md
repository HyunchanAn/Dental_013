![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen "Status") ![Python](https://img.shields.io/badge/Python-3.12%2B-blue "Python") ![Backend](https://img.shields.io/badge/Backend-YOLOv8--seg%20%2F%20EfficientNet--B0-red "Backend") ![UI](https://img.shields.io/badge/UI-Streamlit-orange "UI") ![License](https://img.shields.io/badge/License-MIT-green "License") ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github "CI/CD Pipeline")

# Dental_013: Restoration & Prosthesis Classifier / Segmentation Module (치과 수복물 및 보철물 분류·분할)

## 개요

`Dental_013`은 파노라마 방사선 영상에서 치과 수복물 및 보철물(Crown, Implant, Filling, Root Canal Treatment 등)을 탐지·분류하는 모듈로, 단독 실행 환경과 통합 파이프라인(`Dental_Panoramic_Reader`) 환경에 맞추어 이중 아키텍처(Dual Architecture)를 제공합니다.

- [학습 및 추론 하드웨어 사양]:
  - Training Env (메인 워크스테이션): AMD Ryzen 9 9900X, NVIDIA GeForce RTX 5080 (16GB VRAM), 64GB RAM
  - Development / Inference Env (랩탑 워크스테이션): Intel Core i5-14450HX, NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), 16GB RAM

---

## 주요 기능 및 아키텍처

### 1. 단독 치아 ROI 분류기 (Standalone Tooth-ROI Torchvision Classifier)
- 구성 파일: `src/prepare_data.py`, `src/train.py`, `src/predict.py`
- 동작 방식: `Dental_008`(치아 분할 모듈) 또는 YOLO 바운딩 박스로 크롭(Crop)된 개별 치아 ROI 이미지를 입력받아 `torchvision`의 `EfficientNet-B0` 백본(`num_classes=7`: `Bridge`, `Crown`, `Filling`, `Implant`, `Natural`, `Other`, `RCT`)으로 상태를 분류하는 2-Stage 파이프라인입니다.

### 2. 통합 파이프라인 31-Class YOLOv8-seg ONNX 모델 (`Dental_Panoramic_Reader` 연동)
- 가중치 식별자: Hugging Face `chemahc94/Dental_013` (`models/best_restoration_model.onnx`, 입력 해상도 `640x640`, 31개 원본 치과 소견 클래스 내장)
- 통합 래퍼: `Dental_Panoramic_Reader/modules/restoration_predictor.py` (`RestorationPredictorWrapper`)
- 4대 임상 수복물 범주 매핑 (SSOT Clinical Mapping):
  - `Crown` (`Crown`, `metal band`, `plating`) -> `Crown/Bridge`
  - `Implant` (`Implant`, `TAD`, `abutment`, `gingival former`) -> `Implant`
  - `Filling` (`Filling`) -> `Inlay/Onlay`
  - `Root Canal Treatment` (`Root Canal Treatment`, `post - core`) -> `Endodontic (RCT)`
  - 교정 장치(`orthodontic brackets`, `permanent retainer`, `wire`)는 `Other/Orthodontic`으로 분리하며, 비수복성 병소 클래스(`Caries`, `Missing teeth`, `Periapical lesion` 등)는 타 전문 모듈(`Dental_002`, `Dental_008`, `Dental_012`)과의 중복을 막기 위해 필터링합니다.
  - 검출된 수복물 바운딩 박스는 `Dental_008`의 치아 인스턴스 바운딩 박스와 공간 교차 면적(IoU / 중심점 포함 여부)을 계산하여 개별 FDI 치아 번호에 자동 매핑됩니다.

### 3. 디렉토리 구조
- `data/raw/`: 원본 데이터셋 (Kaggle 다운로드 경로, Gitignore 처리)
- `data/processed/`: `prepare_data.py`를 거쳐 클래스별로 분할된 치아 ROI 분류 학습용 폴더
- `src/`: 단독 분류기 학습 및 추론 코드 (`download_data.py`, `prepare_data.py`, `train.py`, `predict.py`)
- `tests/`: 모델 생성 및 추론 인터페이스 검증 단위 테스트 (`test_predict.py`)
- `models/`: 학습된 가중치(`best_restoration_model.pth`, `best_restoration_model.onnx`) 저장 폴더

---

## 벤치마크 및 검증 성능

- 초기 `Dental_000` 문서에 기재되었던 `31종 YOLOv8-seg 88.5% / 84.2%` 수치는 프로토타입 단계의 내부 검증 요약치였으며, 과거 감사 문서의 `데이터/라벨 준비 전 보류` 표기는 본 레포지토리의 단독 `EfficientNet-B0` 크롭 분류기(`src/train.py`) 로컬 데이터셋 미배치 상태를 지칭한 것이었습니다.
- 현재 프로덕션 통합 파이프라인(`Dental_Panoramic_Reader`)에서는 `best_restoration_model.onnx` (31-class YOLOv8-seg) 가중치를 통해 전체 파노라마 영상에서 직접 수복물을 검출하고 4대 임상 범주로 매핑합니다.
- [10대 임상 케이스 정량 벤치마크 (`10_cases_quantitative_benchmark.json`, `IoU >= 0.50` 엄격 매칭 기준)]:
  - 수복물 전용 케이스 검출 성공률(Case-level Detection Rate): `100%` (전용 테스트 케이스 3건 및 복합 케이스 1건 모두에서 누락 없이 수복물 인스턴스 검출 성공)
  - 케이스별 인스턴스 매칭 성적 (`IoU >= 0.50`, 4대 수복물 범주):
    - `case_013_restoration_1.jpg` (`Gold=32`): `TP=11`, `FP=14`, `FN=21` (Precision `44.0%`, Recall `34.4%`)
    - `case_013_restoration_2.jpg` (`Gold=38`): `TP=16`, `FP=5`, `FN=22` (Precision `76.2%`, Recall `42.1%`)
    - `case_013_restoration_3.jpg` (`Gold=27`): `TP=13`, `FP=10`, `FN=14` (Precision `56.5%`, Recall `48.1%`)

---

## 데이터셋 출처 및 라이선스

- [원본 데이터셋]: Kaggle [Dental Disease Panoramic Detection Dataset](https://www.kaggle.com/datasets/lokisilvres/dental-disease-panoramic-detection-dataset) (`lokisilvres/dental-disease-panoramic-detection-dataset`)
- [데이터셋 이용 조건]:
  - 공개된 연구·교육 목적의 파노라마 치과 질환/수복물 어노테이션 데이터셋으로, 원본 데이터 파일은 용량 및 재배포 정책에 따라 본 Git 저장소(`data/` Gitignore)에 포함하지 않으며 Kaggle API(`src/download_data.py`)를 통해 개별 다운로드해야 합니다.
  - 상용 의료기기 인증 또는 임상 진단 목적으로 직접 사용할 수 없으며, 학술 연구 및 알고리즘 벤치마크 용도로만 활용합니다.
- [소스 코드 라이선스]: `MIT License` ([LICENSE](LICENSE), `Copyright (c) 2026 Hyunchan An`)

---

## 설치 및 실행 방법

1. 패키지 설치 및 데이터 다운로드:
   ```bash
   pip install -r requirements.txt
   python src/download_data.py
   ```
2. 데이터 전처리 및 단독 분류 모델 학습:
   ```bash
   python src/prepare_data.py
   python src/train.py
   ```
3. 추론 및 단위 테스트 실행:
   ```bash
   python src/predict.py <path_to_cropped_image>
   pytest tests/test_predict.py -v
   ```

---

## 참고 문헌 및 연구용 실행 주의사항

- Kaggle Dataset: `lokisilvres/dental-disease-panoramic-detection-dataset`
- [연구용 실행 주의사항]:
  - 기본 실행 주소: `127.0.0.1` (로컬 전용)
  - 외부 공개/상용 배포 금지 및 오픈 데이터셋 외 민감 의료데이터 사용 금지
  - API 키·토큰은 환경변수로만 관리하며, 외부 접근이 필요하면 `ALLOW_EXTERNAL_ACCESS=true`로만 명시적 허용
