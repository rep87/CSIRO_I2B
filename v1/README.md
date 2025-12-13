# v1: csiro simple Colab 재현 버전

- Kaggle 공개 노트북 **csiro simple**(Public Score 0.64 V21) 구조를 기반으로, Colab T4 환경에서 학습/추론/제출을 한 번에 실행할 수 있도록 재구성했습니다.
- 단일 EfficientNet-B2 백본으로 5-fold 교차 검증을 수행하고, fold별 best 가중치를 평균 앙상블하여 추론합니다.
- 학습 시 세 타깃(Dry_Green_g, Dry_Clover_g, Dry_Dead_g)을 직접 예측하며, 추론 단계에서 GDM_g와 Dry_Total_g를 산술적으로 계산합니다.
- 데이터는 long-format CSV를 wide-format으로 피벗해 한 이미지당 한 행으로 묶어 사용합니다. `sample_id_prefix`는 `sample_id`의 앞부분(`__` 이전)으로 추출합니다.
- 학습/검증 데이터 분할은 `KFold(n_splits=5, shuffle=False)`로 수행하여 원본 노트북의 단순 분할 방식을 재현합니다.
- 손실 함수는 `SmoothL1Loss`, 평가 지표는 5개 타깃에 대한 weighted R²이며, optimizer는 AdamW + ReduceLROnPlateau 조합입니다.
- `DEBUG` 토글로 소량 샘플/1 epoch 실행이 가능하고, `USE_OPTUNA` 토글로 간단한 하이퍼파라미터 탐색을 선택적으로 수행할 수 있습니다.
- 산출물은 `outputs/<run_name>/` 아래에 fold별 체크포인트, 로그, 제출 파일이 저장되며, `submission/submission.csv`는 Kaggle 제출 규격을 따릅니다.
