# PIM Framework

Llama 계열 언어모델에 대해 `SmoothQuant`, linear fake quantization, custom attention, 그리고 PIM 하드웨어 효과를 함께 적용해 perplexity를 평가하는 실험용 프레임워크입니다.

## Overview

이 프로젝트는 다음 흐름을 중심으로 구성되어 있습니다.

1. calibration 데이터로 activation scale 수집
2. `smooth.py`에서 SmoothQuant 적용
3. `fake_quant.py`에서 linear layer를 양자화된 PIM-aware 모듈로 치환
4. `custom_attention.py`에서 attention matmul 경로를 커스텀
5. `hw_effect.py`에서 retention, noise, ADC 같은 device-level 효과 반영
6. `ppl_eval.py`에서 전체 설정을 묶어 perplexity 평가

핵심 실행 경로는 다음과 같습니다.

`ppl_eval.py -> fake_quant.py + custom_attention.py -> hw_effect.py`

## Key Features

- Llama 계열 모델 대상 SmoothQuant 적용
- per-channel weight quantization
- per-token activation quantization
- linear layer에 대한 PIM-aware fake quantization
- attention의 `QK^T`와 `softmax x V` 경로에 대한 custom hardware-aware simulation
- retention, conductance variation, ADC quantization 반영
- WikiText-2 기반 perplexity 평가

## Project Structure

```text
PIM_Framework/
+-- calibration.py
+-- custom_attention.py
+-- fake_quant.py
+-- hw_effect.py
+-- ppl_eval.py
+-- smooth.py
+-- LICENSE
+-- README.md
```

## File Guide

- [`ppl_eval.py`](./ppl_eval.py): 실험 진입점, 모델 로드, 옵션 파싱, perplexity 평가
- [`smooth.py`](./smooth.py): Llama decoder layer에 대한 SmoothQuant 적용
- [`fake_quant.py`](./fake_quant.py): linear layer 양자화 및 PIM-aware `W8A8Linear` 구현
- [`custom_attention.py`](./custom_attention.py): custom Llama attention forward 구현
- [`hw_effect.py`](./hw_effect.py): retention, device noise, ADC helper 함수
- [`calibration.py`](./calibration.py): activation scale 수집용 calibration 코드

## Requirements

실행에는 아래와 같은 패키지가 필요합니다.

- Python 3.10+
- PyTorch
- transformers
- datasets
- tqdm

필요한 환경은 예를 들어 아래와 같습니다.

```bash
pip install torch transformers datasets tqdm
```

## Usage

### 1. Activation Scale Calibration

먼저 calibration dataset으로 activation scale을 수집합니다.

`calibration.py`의 `get_act_scales()`를 사용해 scale tensor를 저장한 뒤, 그 경로를 `--act_scales_path`로 넘기면 됩니다.

### 2. Perplexity Evaluation

기본 실행 예시는 아래와 같습니다.

```bash
python ppl_eval.py ^
  --model_path /path/to/Llama-2-7b-hf ^
  --act_scales_path /path/to/act_scales.pt ^
  --smooth 1 ^
  --quantize 1 ^
  --custom_attn 1
```

Linux/macOS에서는 줄바꿈만 바꾸면 됩니다.

```bash
python ppl_eval.py \
  --model_path /path/to/Llama-2-7b-hf \
  --act_scales_path /path/to/act_scales.pt \
  --smooth 1 \
  --quantize 1 \
  --custom_attn 1
```

## Main Arguments

- `--model_path`: Hugging Face 모델 경로 또는 로컬 체크포인트 경로
- `--act_scales_path`: calibration으로 생성한 activation scale 파일 경로
- `--n_samples`: perplexity evaluation에 사용할 sample block 수
- `--smooth`: SmoothQuant 적용 여부
- `--quantize`: fake quantization 적용 여부
- `--custom_attn`: custom attention 적용 여부

### Linear Layer Hardware Arguments

- `--detect_L`
- `--cellBit_L`
- `--subArray_L`
- `--ADCprecision_L`
- `--ADCtype_L`
- `--ADCgamma_L`
- `--t_L`
- `--cycle_L`

### KV Cache / Attention Hardware Arguments

- `--detect_KV`
- `--cellBit_KV`
- `--subArray_KV`
- `--ADCprecision_KV`
- `--ADCtype_KV`
- `--ADCgamma_KV`
- `--t_KV`
- `--cycle_KV`

## Notes

- 현재 정리된 코드는 Llama 계열 모델 중심의 최소 실행 경로에 맞춰 정리되어 있습니다.
- 디버깅용 CSV 저장, plotting, profiling 관련 보조 코드는 제거된 상태입니다.
- `fake_quant.py`는 현재 `per_channel` / `per_token` 경로를 기준으로 유지되었습니다.
- `custom_attention.py`는 custom attention이 켜졌을 때만 사용됩니다.

## License

이 프로젝트는 [MIT License](./LICENSE)를 따릅니다.
