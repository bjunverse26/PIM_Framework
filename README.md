# PIM Framework Project

Llama 계열 언어모델에 대해 SmoothQuant, fake quantization, custom attention, 그리고 PIM 하드웨어 효과를 적용해 perplexity를 평가한 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 단순한 양자화 실험이 아니라, LLM 추론 경로 안에 실제 메모리 소자 특성과 PIM 연산 제약을 반영하는 것을 목표로 합니다.  
linear layer와 attention matmul 경로 모두에 대해 bit-serial accumulation, cell-level decomposition, subarray partitioning, retention, noise, ADC quantization을 반영하도록 구성했습니다.

## 한눈에 보기

| 항목 | 내용 |
| --- | --- |
| 프로젝트 유형 | LLM 시스템 연구 + PIM 하드웨어 시뮬레이션 |
| 대상 모델 | Llama 계열 |
| 핵심 기법 | SmoothQuant, Fake Quantization, Custom Attention |
| 하드웨어 반영 요소 | Retention, Conductance Mapping, Noise, ADC |
| 평가 지표 | Perplexity |
| 실행 흐름 | `ppl_eval.py -> fake_quant.py + custom_attention.py -> hw_effect.py` |

## 핵심 성과

### 1. Llama 추론 경로에 PIM-aware 연산 구조 반영
- `fake_quant.py`에서 linear layer를 `W8A8Linear`로 치환
- activation bit-plane 분해, weight cell-level 분해, subarray 기반 partial sum 누적 구조 구현

### 2. Attention matmul까지 hardware-aware path로 확장
- `custom_attention.py`에서 `QK^T`, `softmax x V` 경로를 custom forward로 대체
- attention 내부 matmul에도 retention, noise, ADC 기반 연산 흐름 반영

### 3. 소자 특성을 모델 수준 평가와 연결
- `hw_effect.py`에서 `Ion`, `Ioff`, `Vread`, retention, device variation, ADC quantization 반영
- 최종적으로 perplexity를 통해 하드웨어 제약이 모델 품질에 미치는 영향 평가

### 4. 정리된 최소 실행 구조 구축
- 디버깅/보조 코드 제거 후 핵심 경로 중심으로 구조 단순화
- `ppl_eval.py`, `fake_quant.py`, `custom_attention.py`, `hw_effect.py`, `smooth.py` 중심으로 실험 구조 정리

## Beomjun Kim 기여 사항

코드 주석 기준으로 직접 기여한 핵심은 아래와 같습니다.

- `fake_quant.py`
  물리적 noise 및 retention error 적용
  아날로그 선형 연산 수행
  ADC 변환 및 디지털 복원(correction) 경로 반영

- `custom_attention.py`
  PV matmul 하드웨어 시뮬레이션 적용
  attention 경로에서 retention 효과 반영

- `hw_effect.py`
  retention / ADC helper 보강

- `ppl_eval.py`
  KV cache / attention 하드웨어 시뮬레이션 파라미터 구성
  WikiText-2 기반 evaluation entry 정리

## 기능

- Llama 계열 모델 대상 SmoothQuant 적용
- per-channel weight quantization
- per-token activation quantization
- linear layer PIM-aware fake quantization
- attention `QK^T`, `softmax x V` hardware-aware simulation
- retention, conductance variation, ADC quantization 반영
- WikiText-2 기반 perplexity 평가

## 기술 스택

| 구분 | 내용 |
| --- | --- |
| 언어 | Python |
| 프레임워크 | PyTorch, Hugging Face Transformers |
| 데이터셋 | WikiText-2 |
| 핵심 기법 | SmoothQuant, Fake Quantization |
| 하드웨어 모델링 | Retention, Noise, ADC, Subarray-based Computation |

## 프로젝트 구조

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