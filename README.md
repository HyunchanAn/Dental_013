# Dental_013: Restoration & Prosthesis Classifier

![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12%2B-blue) ![Backend](https://img.shields.io/badge/Backend-YOLOv8-red) ![UI](https://img.shields.io/badge/UI-Streamlit-orange) ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github)

## 개요
> **[학습 환경 사양]** 실질적 모델 학습은 **RTX 5080 + 라이젠9-6 9900x** 환경에서 진행되었습니다.

파노라마 방사선 사진에서 분할(Segmentation)된 개별 치아 이미지를 입력받아, 해당 치아의 수복물 및 보철물 상태(Crown, Implant, Filling, RCT 등)를 판별하는 딥러닝 분류(Classification) 모듈입니다.

기존 `Dental_008` (치아 분할 및 FDI 번호 부여 모듈)과 연동되는 **2-Stage 아키텍처**로 작동합니다. `Dental_008`이 전체 파노라마에서 개별 치아를 분할 및 Crop하면, 이 모듈(`Dental_013`)이 해당 Crop된 이미지를 바탕으로 치아의 상태를 판별합니다.

## 디렉토리 구조
- `data/raw/`: 원본 데이터셋 (Kaggle)
- `data/processed/`: `prepare_data.py`를 거쳐 클래스별로 분할된 학습용 폴더
- `src/`: 학습 및 추론 코드 (`prepare_data.py`, `train.py`, `predict.py`)
- `scripts/`: 데이터 다운로드 스크립트 등
- `models/`: 학습된 가중치(.pth)가 저장되는 폴더

## 설치 및 실행 방법

### 1. 데이터 다운로드
Kaggle API가 설정된 워크스테이션에서 실행합니다.
```bash
python src/download_data.py
```

### 2. 데이터 전처리 (분류형 폴더 구축)
다운로드된 YOLO 포맷의 Bounding Box 라벨을 바탕으로 치아 이미지를 개별 Crop하여 `train`, `val` 폴더로 나눕니다.
```bash
python src/prepare_data.py
```

### 3. 학습 시작
PyTorch 및 EfficientNet-B0 기반 모델을 학습시킵니다.
```bash
python src/train.py
```

### 4. 추론 테스트
단일 치아 Crop 이미지를 입력하여 추론을 진행합니다.
```bash
python src/predict.py <path_to_cropped_image>
```

## References
* **Kaggle Dataset:** [Dental Disease Panoramic Detection Dataset](https://www.kaggle.com/datasets/lokisilvres/dental-disease-panoramic-detection-dataset)
* **Architecture:** 2-Stage Pipeline with Dental_008
