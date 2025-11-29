# FEATURE_TOGGLES

아래 목록은 학습/추론 동작을 바꾸기 위해 사람이 자주 조정하는 주요 설정 모음입니다. 각 항목마다 설정 경로, 역할, 그리고 바로 적용할 수 있는 예시 스니펫을 제공합니다.

## 모델 관련
- **백본 선택**
  - 설정 경로: `configs/train_config.yaml` → `model.backbone`
  - 역할: timm 모델(예: ConvNeXt) 중 어떤 아키텍처를 사용할지 결정합니다.
  - 수정 예시:
    ```yaml
    model:
      backbone: convnext_base
    ```

- **패치 개수**
  - 설정 경로: `configs/train_config.yaml` → `model.patch_count`
  - 역할: 이미지를 몇 개의 패치로 분할할지 결정하며, 데이터셋 패치 생성과 모델의 피처 결합 차원을 모두 결정합니다.
  - 수정 예시:
    ```yaml
    model:
      patch_count: 4  # 2x2 그리드로 분할
    ```

- **사전학습 가중치 사용**
  - 설정 경로: `configs/train_config.yaml` → `model.pretrained`
  - 역할: 백본을 ImageNet 등 사전학습 가중치로 초기화할지 여부를 설정합니다.
  - 수정 예시:
    ```yaml
    model:
      pretrained: false
    ```

## 학습 하이퍼파라미터
- **학습 epoch 수**
  - 설정 경로: `configs/train_config.yaml` → `train.epochs`
  - 역할: 전체 데이터셋을 몇 번 반복해 학습할지 결정합니다.
  - 수정 예시:
    ```yaml
    train:
      epochs: 50
    ```

- **배치 크기**
  - 설정 경로: `configs/train_config.yaml` → `train.batch_size`
  - 역할: 한 번의 옵티마이저 업데이트마다 처리할 샘플 수를 지정합니다.
  - 수정 예시:
    ```yaml
    train:
      batch_size: 4
    ```

- **학습률**
  - 설정 경로: `configs/train_config.yaml` → `train.lr`
  - 역할: 옵티마이저의 스텝 크기를 결정하여 학습 안정성과 속도에 영향을 줍니다.
  - 수정 예시:
    ```yaml
    train:
      lr: 1e-4
    ```

- **옵티마이저 유형**
  - 설정 경로: `configs/train_config.yaml` → `train.optimizer`
  - 역할: 파라미터 업데이트 방식을 결정합니다 (`adamw`, `adam` 지원).
  - 수정 예시:
    ```yaml
    train:
      optimizer: adam
    ```

- **스케줄러**
  - 설정 경로: `configs/train_config.yaml` → `train.scheduler`
  - 역할: 학습률 변화를 정의합니다 (`cosine` 지원).
  - 수정 예시:
    ```yaml
    train:
      scheduler: cosine
    ```

- **입력 이미지 크기**
  - 설정 경로: `configs/train_config.yaml` → `train.image_size`
  - 역할: 입력 이미지를 리사이즈하는 크기이며, 패치의 기본 해상도도 함께 결정합니다.
  - 수정 예시:
    ```yaml
    train:
      image_size: 768
    ```

## 데이터/증강
- **데이터 루트 경로**
  - 설정 경로: `configs/train_config.yaml` → `data.data_root`
  - 역할: 이미지와 메타데이터가 위치한 루트 디렉토리를 지정합니다.
  - 수정 예시:
    ```yaml
    data:
      data_root: "/data/biomass"
    ```

- **메타데이터 날짜 스플릿**
  - 동작 순서: `date` 컬럼이 있으면 그대로 사용 → 없으면 `Sampling_Date`를 `date`로 변환 → 둘 다 없으면 80/20 랜덤 스플릿으로 폴백합니다.
  - Kaggle `train.csv` 스키마(필수): `sample_id`, `image_path`, `Sampling_Date`, `State`, `Species`, `Pre_GSHH_NDVI`, `Height_Ave_cm`, `target_name`, `target`
  - 직접 가공 시 `Sampling_Date`를 유지하거나 `date`로 만들어 두면 시간 기반 스플릿이 안정적으로 동작합니다.
  - 예시 코드:
    ```python
    import pandas as pd

    df = pd.read_csv("train.csv")
    df = df.rename(columns={"Sampling_Date": "date"})
    df.to_csv("train_with_date.csv", index=False)
    ```

- **색상 변조 증강**
  - 설정 경로: `configs/train_config.yaml` → `data.augment.color_jitter`
  - 역할: ColorJitter 증강 사용 여부를 설정합니다.
  - 수정 예시:
    ```yaml
    data:
      augment:
        color_jitter: false
    ```

- **수평 뒤집기 증강**
  - 설정 경로: `configs/train_config.yaml` → `data.augment.horizontal_flip`
  - 역할: 수평 플립 증강 적용 여부를 설정합니다.
  - 수정 예시:
    ```yaml
    data:
      augment:
        horizontal_flip: false
    ```

## 손실 및 실험 관리
- **손실 함수**
  - 설정 경로: `configs/train_config.yaml` → `loss.type`
  - 역할: 회귀 손실 종류를 선택합니다 (`smooth_l1`, `huber`, `mae`, `rmse` 지원).
  - 수정 예시:
    ```yaml
    loss:
      type: huber
    ```

- **실험 결과 저장 경로**
  - 설정 경로: `configs/train_config.yaml` → `experiment.save_dir`
  - 역할: 체크포인트와 로그를 저장할 디렉토리를 지정합니다. 실행 시 타임스탬프 서브폴더가 자동 생성됩니다.
  - 수정 예시:
    ```yaml
    experiment:
      save_dir: "./experiments/debug_runs"
    ```
