# Dental_013: EfficientNet-B0 Classification Benchmark Report

**Date**: 2026-07-14
**Model**: EfficientNet-B0 (PyTorch)
**Task**: Dental Image Classification (치과 방사선/파노라마 이미지 분류)

---

## 1. 훈련 환경 최적화 내역
본 훈련은 대상 워크스테이션(Ryzen 9900X + RTX 5080 16GB)의 스펙을 최대한 활용하도록 최적화되어 진행되었습니다.

- **Batch Size**: 128 (RTX 5080의 16GB VRAM 활용 극대화)
- **Num Workers**: 8 (Ryzen 9900X의 병렬 데이터 로딩 가속)
- **Pin Memory**: True (호스트-디바이스 간 메모리 전송 최적화)
- **Early Stopping**: Patience 15 에포크 (과적합 방지 로직 자체 구현 적용)

---

## 2. 훈련 개요 및 종료 상태

- **총 소요 시간**: 약 1시간 50분
- **종료 사유**: Early Stopping (조기 종료 발동)
  - 훈련 데이터세트의 크기(약 9만 장)가 워낙 거대하여 100 에포크를 목표로 세팅하였으나, Validation 성능 개선이 15 에포크 연속으로 이뤄지지 않자 **에포크 26 지점에서 훈련이 자동 종료**되었습니다.
- **최종 가중치 보존**: 과적합(Overfitting) 발생 전, 검증 셋에서 가장 똑똑한 성능을 냈던 황금기(Best)의 가중치가 보존되었습니다.
- **가중치 파일 경로**: `\\rtx4060laptop-hc\Users\chema\Github\Dental_013\models\best_restoration_model.pth`

---

## 3. 벤치마크 평가 지표 (Best Validation Metrics)

모델이 과적합 없이 도달한 최고의 검증(Validation) 성능 지표입니다.

| Metric | Score | 비고 |
| :--- | :--- | :--- |
| **Best Val Accuracy (최고 검증 정확도)** | **`85.76%`** | 9만 장이라는 거대한 데이터세트 대비 훌륭한 초기 정답률 |
| **Best Val Loss (최소 검증 오차)** | **`0.4041`** | 안정적으로 수렴한 최소 오차 지점 |

> [!WARNING]
> **과적합(Overfitting) 방어 분석**
> 훈련 막바지 로그 분석 결과, 훈련 세트(Train Set)에 대한 모델의 정확도는 무려 **94.17%**까지 치솟았으나, 정작 처음 보는 검증 세트(Val Set)에 대한 정확도는 85.76%에서 더 오르지 않고 정체되었습니다. 
> 이는 모델이 데이터의 일반적인 패턴(Generalization)을 넘어서서, 훈련 정답지 자체를 통째로 암기해 버리려는 '과적합(Overfitting)' 현상에 빠져들기 시작했다는 명백한 증거입니다. 다행히 훈련 시작 전 세팅해 둔 스마트 컷오프(Early Stopping) 알고리즘이 이를 정확히 포착하고 즉시 파이프라인을 절단하여, 멍청해지기 직전의 가장 깨끗한 가중치 상태(`85.76%`)를 온전히 보존하는 데 성공했습니다.

---

## 4. 추론 속도 및 향후 방안
- **초당 처리량 (Throughput)**: 훈련 구간에서 `128 Batch` 기준 초당 약 4.5~5.0 회전 (즉, **초당 약 570~640장**의 엄청난 이미지 연산량)을 달성했습니다.
- 추론(Inference) 시에는 역전파(Backpropagation) 연산이 없으므로 이보다 훨씬 더 빠르고 가벼운 속도로 실시간 분류가 가능합니다.
- 추가 성능 향상을 원할 경우, Augmentation 기법(MixUp, CutMix 등)을 다변화하거나 `EfficientNet-B3` 또는 `B4` 등 상위 체급의 모델로 아키텍처 스케일업을 고려해 볼 수 있습니다.
