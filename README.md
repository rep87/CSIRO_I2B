# CSIRO Image2Biomass Baseline (Colab-ready)

이 레포는 Kaggle **CSIRO – Image2Biomass Prediction** 대회용 베이스라인을 Google Colab(T4 기준)에서 바로 학습/추론 재현 가능하도록 구성했습니다. `git clone` 후 경로 설정 블록만 수정하면 학습 → 추론 → 제출 파일 생성까지 한 번에 실행할 수 있습니다.

## 빠른 시작 (Colab)
1. Google Drive 마운트 후 대회 데이터를 Drive에 복사합니다(`train/`, `test/`, `train.csv`, `test.csv`).
2. 본 레포를 클론합니다.
3. `v1/colab_runner.py`의 **PATHS** 블록에서 `data_root`만 Drive 경로에 맞게 수정합니다.
4. **CONFIG** 블록에서 하이퍼파라미터(백본, 이미지 크기, 배치, 에폭, seed, AMP, grad accumulation 등)를 필요에 맞게 변경합니다. `DEBUG=True`로 두면 소량 샘플로 1 epoch만 실행해 빠른 점검이 가능합니다.
5. `python v1/colab_runner.py`를 실행하면 5-fold 학습 → 저장된 가중치로 추론 → `submission/submission.csv` 생성까지 진행합니다.
6. Optuna 하이퍼파라미터 탐색을 사용하려면 CONFIG 블록의 `use_optuna=True`로 바꾸고 `n_trials`, `timeout_minutes`를 조정합니다.

## 버전 운용 규칙
- 루트에는 `v1/`, `v2/` … 버전 디렉토리를 둡니다. 큰 변경(아키텍처, 데이터 파이프라인, 스플릿, 손실, 추론 로직 등)은 새 버전을 만들어 추가하며 기존 버전은 그대로 보존합니다.
- 사소한 변경(버그 수정, 로그/주석 개선)은 동일 버전 내에서 진행합니다.
- 항상 최신 작업은 가장 큰 버전 번호 디렉토리에서 수행합니다.

## 데이터 준비 가이드
- Google Drive에 아래 구조로 배치하는 것을 가정합니다(`data_root`가 가리키는 경로).
```
<data_root>/
├── train.csv
├── test.csv
├── train/  # 학습 이미지(JPEG)
└── test/   # 테스트 이미지(JPEG)
```
- Kaggle 전용 경로(`/kaggle/input/...`)는 사용하지 않으며, 필요 시 별도 분기만 추가하면 됩니다.

## 설정 셀/블록 안내
- **PATHS 블록**: `data_root`와 `output_root`, `run_name`을 한 곳에서 제어합니다. `data_root`만 바꾸면 CSV/이미지/출력 경로가 모두 자동으로 따라갑니다.
- **CONFIG 블록**: 모델 백본, 이미지 크기, 배치, 에폭, seed, optimizer/scheduler, AMP, grad accumulation, CV fold 수, `DEBUG` 토글, `USE_OPTUNA` 토글을 한 곳에서 관리합니다.

## Optuna 사용법
- `use_optuna=True`로 설정하면 CV weighted R²를 최대화하는 방향으로 간단한 파라미터 탐색을 수행합니다(`lr`, `weight_decay`, `image_size`, `batch_size`, `patience`).
- `n_trials`, `timeout_minutes`, `study_name`, `storage`를 CONFIG 블록에서 조정합니다.
- 탐색이 끝나면 best trial을 출력하며, 탐색 시에는 디버그 모드로 빠르게 실행합니다.

## 산출물 위치
- 실행 시 `outputs/<run_name>/` 아래에 결과가 저장됩니다.
  - `checkpoints/`: fold별 `*_best.pth`, `*_last.pth`
  - `preds/`: (확장용, 현재는 미사용)
  - `submission/submission.csv`: Kaggle 제출 형식 파일 (두 컬럼: `sample_id`, `target`)
  - `log.txt`: stdout 로그를 파일로도 저장

## 파일 트리 예시 (v1 기준)
```
README.md
requirements.txt
v1/
├── README.md
├── colab_runner.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── inference.py
    ├── metrics.py
    ├── model.py
    ├── optuna_search.py
    ├── train.py
    └── utils.py
```

## 버전별 README
각 버전 디렉토리(`v1/README.md` 등)는 해당 버전의 핵심 변경점과 실험 목적을 5~10줄로 요약해 기록합니다.

## 주의사항
- 분석/시각화 코드는 포함하지 않았습니다(사용자가 Colab에서 직접 수행).
- 단일 GPU(Colab T4) 환경을 기본으로 하며, 멀티 GPU/분산은 비활성화되어 있습니다.
- 제출 파일(`submission.csv`)은 테스트 이미지 수 × 5개 타깃 행으로 구성되며, 헤더는 `sample_id,target`만 포함합니다.
