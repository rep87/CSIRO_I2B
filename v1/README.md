# v1: csiro simple Colab 재현 버전

- Kaggle 공개 노트북 **csiro simple**(Public Score 0.64 V21) 구조를 기반으로, Colab T4 환경에서 학습/추론/제출을 한 번에 실행할 수 있도록 재구성했습니다.
- 단일 EfficientNet-B2 백본으로 5-fold 교차 검증을 수행하고, fold별 best 가중치를 평균 앙상블하여 추론합니다.
- 학습 시 세 타깃(Dry_Green_g, Dry_Clover_g, Dry_Dead_g)을 직접 예측하며, 추론 단계에서 GDM_g와 Dry_Total_g를 산술적으로 계산합니다.
- 데이터는 long-format CSV를 wide-format으로 피벗해 한 이미지당 한 행으로 묶어 사용합니다. `sample_id_prefix`는 `sample_id`의 앞부분(`__` 이전)으로 추출합니다.
- 학습/검증 데이터 분할은 `KFold(n_splits=5, shuffle=False)`로 수행하여 원본 노트북의 단순 분할 방식을 재현합니다.
- 손실 함수는 `SmoothL1Loss`, 평가 지표는 5개 타깃에 대한 weighted R²이며, optimizer는 AdamW + ReduceLROnPlateau 조합입니다.
- `DEBUG` 토글로 소량 샘플/1 epoch 실행이 가능하고, `USE_OPTUNA` 토글로 간단한 하이퍼파라미터 탐색을 선택적으로 수행할 수 있습니다.
- 산출물은 `outputs/<run_name>/` 아래에 fold별 체크포인트, 로그, 제출 파일이 저장되며, `submission/submission.csv`는 Kaggle 제출 규격을 따릅니다.

## 모델 및 이미지 기법
- **모델**: EfficientNet-B2(`pretrained=False`) 기반 회귀 모델로 Dry_Green_g, Dry_Clover_g, Dry_Dead_g 세 타깃을 직접 예측한 뒤, 추론 단계에서 GDM_g와 Dry_Total_g를 계산합니다.
- **이미지 기법**: 이미지를 정사각형으로 리사이즈한 후 `RandomHorizontalFlip`, `RandomVerticalFlip`, `ColorJitter`를 학습 시에 적용합니다. 추가 메타데이터 없이 순수 이미지 입력만 사용합니다.
- **추론 파이프라인**: 학습된 가중치(fold별 `_best.pth`)를 불러와 테스트 이미지에 대해 예측하고, `sample_id_prefix`를 활용해 long-format 제출 파일을 생성합니다.

## Kaggle 제출용 실행 방법
- Kaggle Notebook에서 데이터셋 `csiro-biomass`를 추가한 뒤, 런타임에 `!python v1/kaggle_runner.py` 한 줄만 실행합니다.
- 출력은 `/kaggle/working/outputs/<run_name>/`와 `/kaggle/working/submission.csv`에 저장됩니다.
- `DEBUG=1`, `RUN_NAME=myrun` 같은 환경 변수를 전달해 빠른 점검이나 실행 이름을 지정할 수 있습니다.

## Flattened Kaggle submission (inference-only)
- `v1/kaggle_submit_flat.ipynb`는 **Kaggle 제출 전용(flattened, inference-only) 노트북**으로, v1/src의 추론 로직을 코드 셀 안에 그대로 인라인했습니다. 외부 `.py` 임포트나 `__file__` 없이 오프라인에서도 동작합니다.
- **모델 백본 및 이미지 설정**: EfficientNet-B2(`pretrained=False`), `image_size=456`, `batch_size=32`, `num_workers=2`를 사용합니다. 추론에서는 리사이즈 + 정규화만 적용하며 학습용 증강은 없습니다.
- **타깃 구성**: 세 개 1차 타깃(`Dry_Green_g`, `Dry_Clover_g`, `Dry_Dead_g`)을 직접 예측하고, 추론 중 `GDM_g = Dry_Green_g + Dry_Clover_g`, `Dry_Total_g = GDM_g + Dry_Dead_g`를 산출해 최종 다섯 개 타깃을 제출합니다.
- **가중치 로딩**: 노트북 상단의 `WEIGHTS_ROOT`를 Kaggle에 첨부한 가중치 데이터셋 경로(예: `/kaggle/input/<your-weights-dataset>/v1_weights`)로 교체한 뒤 실행하면 fold별 `*_best.pth` 다섯 개를 평균 앙상블하여 `submission.csv`를 생성합니다.
