# PIM Framework

Llama 계열 대규모 언어모델에 대해 `SmoothQuant`, linear fake quantization, custom attention, 그리고 PIM 하드웨어 효과를 함께 적용해 perplexity를 평가하는 연구용 프레임워크입니다.

이 프로젝트의 핵심은 단순한 양자화 실험이 아니라, LLM 연산 경로 안에 실제 메모리 소자 특성과 PIM 연산 제약을 직접 반영합니다.

## Overview

프로젝트는 아래 흐름을 중심으로 동작합니다.

1. calibration 데이터로 activation scale 수집
2. `smooth.py`에서 SmoothQuant 적용
3. `fake_quant.py`에서 linear layer를 PIM-aware fake quant 모듈로 치환
4. `custom_attention.py`에서 attention 내부 matmul을 hardware-aware path로 대체
5. `hw_effect.py`에서 retention, noise, ADC 등의 device-level 특성 반영
6. `ppl_eval.py`에서 전체 설정을 묶어 perplexity 평가

핵심 실행 경로는 다음과 같습니다.

`ppl_eval.py -> fake_quant.py + custom_attention.py -> hw_effect.py`

## Key Features

- Llama 계열 모델 대상 SmoothQuant 적용
- per-channel weight quantization
- per-token activation quantization
- linear layer에 대한 PIM-aware fake quantization
- attention의 `QK^T`, `softmax x V` 경로에 대한 custom hardware-aware simulation
- retention, conductance variation, ADC quantization 반영
- WikiText-2 기반 perplexity 평가

## Hardware-Oriented Design

이 프로젝트는 소프트웨어에서 바로 `matmul`만 수행하는 구조가 아닙니다.  
양자화된 activation과 weight를 PIM 메모리 어레이에서 실제로 처리되는 방식에 가깝게 다시 분해하고 누적합니다.

구체적으로는 다음과 같은 연산 흐름을 따릅니다.

### 1. Quantization

- weight는 `per-channel` 방식으로 양자화됩니다.
- activation은 `per-token` 방식으로 양자화됩니다.
- 이 과정에서 scale과 zero-point를 함께 저장해 이후 복원에 사용합니다.

### 2. Bit-Serial / Cell-Level Decomposition

양자화된 값은 바로 행렬곱에 들어가지 않습니다.

- activation은 bit-plane 단위로 분해됩니다.
- weight는 cell precision (`cellBit`) 단위로 분해됩니다.

즉, 디지털 텐서를 한 번 더 쪼개어, 메모리 셀과 비트 직렬 누산 구조를 소프트웨어에서 모사합니다.

### 3. Subarray-Aware Computation

weight matrix 전체를 한 번에 처리하지 않고, `subArray` 크기를 기준으로 나누어 부분 연산을 수행합니다.

이 구조는 실제 PIM 아키텍처에서 하나의 어레이가 감당할 수 있는 column/row 범위를 반영하기 위한 것입니다.  
따라서 이 프로젝트는 단순 quantization이 아니라, subarray partitioning이 성능과 오차에 미치는 영향까지 포함합니다.

### 4. Analog Partial Sum and Reconstruction

각 bit-plane, 각 cell slice, 각 subarray에 대해 부분합을 계산한 뒤,

- 소자 특성이 반영된 conductance/current로 변환하고
- ADC를 거쳐 다시 디지털 값으로 양자화한 다음
- dummy-column 보정과 scale 복원을 통해
- 최종적으로 원래 linear output 또는 attention output에 대응되는 값으로 합성합니다.

즉, 양자화 -> 소자 매핑 -> partial sum 생성 -> ADC -> 보정 -> 복원까지를 한 경로 안에서 수행합니다.

## Device Characteristics Reflected in the Model

하드웨어 및 소자 특성은 [`hw_effect.py`](./hw_effect.py)에서 반영됩니다.

### 1. Retention Modeling

`Retention()` 함수는 시간 경과에 따른 conductance 열화를 반영합니다.

프로젝트에서 구분하는 소자 유형은 다음과 같습니다.

- `detect == 1`: RRAM (HZO 5nm)
- `detect == 2`: RRAM
- `detect == 3`: VRRAM (HZO 10nm)
- `detect == 4`: FTJ (HZO 5nm)

소자 유형에 따라 retention ratio가 다르게 정의되고, 이 ratio가 cell conductance에 직접 곱해집니다.  
즉, retention time이 길어질수록 저장된 전도도가 감소하는 효과를 모델에 포함합니다.

### 2. Conductance Mapping

`apply_noise()`에서는 디지털 cell value를 실제 read current 기반 값으로 매핑합니다.

이 단계에서 사용되는 핵심 파라미터는 아래와 같습니다.

- `Ion`: on-state current
- `Ioff`: off-state current
- `Vread`: read voltage
- `cellBit`: 한 메모리 셀이 표현하는 상태 수
- `cycle`: read / endurance condition

즉, 동일한 양자화값이라도 소자 종류와 read cycle에 따라 실제 current level이 달라집니다.

### 3. Device Noise / Variation

각 cell level에 대해 표준편차를 따로 두어 noise를 주입합니다.

- `cell == 0` 상태는 additive noise
- `cell != 0` 상태는 multiplicative noise

방식으로 처리하여, 완전히 이상적인 메모리 셀을 가정하지 않고 소자 간 편차와 read fluctuation을 반영합니다.

### 4. ADC Quantization

`ADC_compute_new()`는 아날로그 partial sum을 ADC resolution으로 다시 양자화하는 단계입니다.

이때 아래 요소가 오차에 직접 영향을 줍니다.

- `subArray`: 한 번에 누산되는 current의 범위
- `ADCprecision`: ADC bit precision

즉, subarray가 커질수록 누적 전류 범위가 커지고, ADC precision이 낮을수록 더 거친 step으로 양자화됩니다.

### 5. Dummy-Column Based Correction

`ADC_new_correction()`은 signal path와 dummy path를 함께 사용해 baseline을 보정하고, 이를 digital scale로 복원합니다.

이는 단순히 `ADC(signal)`만 쓰는 것이 아니라,

- signal column의 ADC output
- dummy column의 ADC output

을 함께 이용한다는 점에서 실제 PIM 보정 흐름을 반영한 구조입니다.

## Linear Path in `fake_quant.py`

[`fake_quant.py`](./fake_quant.py)의 `W8A8Linear`는 일반 `nn.Linear`를 직접 대체하는 모듈입니다.

이 경로에서 수행되는 연산은 아래와 같습니다.

1. activation을 `per-token` 양자화
2. weight를 `per-channel` 양자화
3. activation을 bit-plane으로 분해
4. weight를 `cellBit` 기준으로 분해
5. weight matrix를 `subArray` 기준으로 분할
6. conductance mapping + retention + noise 반영
7. ADC quantization과 dummy correction 수행
8. 모든 결과를 누적하여 output 복원

따라서 이 모듈은 "양자화된 linear layer"라기보다, PIM 메모리 어레이에서 linear 연산이 수행되는 과정을 software로 재현하는 모듈에 가깝습니다.

## Attention Path in `custom_attention.py`

[`custom_attention.py`](./custom_attention.py)는 Hugging Face `LlamaAttention.forward`를 대체하여 attention 내부 연산을 hardware-aware하게 처리합니다.

반영되는 연산은 다음 두 부분입니다.

- `QK^T`
- `softmax x V`

두 경로 모두 linear path와 유사하게,

- 정규화
- bit-plane decomposition
- cell-level decomposition
- subarray partitioning
- analog partial sum
- ADC quantization
- digital reconstruction

순서로 처리됩니다.

## Why This Project Matters

이 프로젝트는 다음 네 층위를 한 번에 연결합니다.

- **Model level**: Llama perplexity로 최종 품질 평가
- **Algorithm level**: SmoothQuant + fake quantization
- **Architecture level**: bit-serial accumulation, cell precision, subarray partitioning
- **Device level**: `Ion`, `Ioff`, `Vread`, retention, noise, ADC

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

필요한 기본 환경은 다음과 같습니다.

- Python 3.10+
- PyTorch
- transformers
- datasets
- tqdm

예시 설치:

```bash
pip install torch transformers datasets tqdm
```

## Usage

### 1. Activation Scale Calibration

먼저 calibration dataset으로 activation scale을 수집합니다.

`calibration.py`의 `get_act_scales()`로 생성한 scale tensor를 저장한 뒤, 그 경로를 `--act_scales_path`로 전달하면 됩니다.

### 2. Perplexity Evaluation

```bash
python ppl_eval.py ^
  --model_path /path/to/Llama-2-7b-hf ^
  --act_scales_path /path/to/act_scales.pt ^
  --smooth 1 ^
  --quantize 1 ^
  --custom_attn 1
```

Linux/macOS:

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