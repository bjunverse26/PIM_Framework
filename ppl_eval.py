import argparse
import os
import warnings

import torch
import torch.nn as nn
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention

import custom_attention
from custom_attention import custom_forward
from fake_quant import quantize_model
from smooth import smooth_lm

# 환경 설정 및 경고 문구 무시
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore", category=FutureWarning)
torch.cuda.set_device(0)

# 실행 인자(Argument) 설정 - 2025.03.11, Minki Choi
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5) # SmoothQuant의 이동 계수
parser.add_argument("--model_path", type=str, default="/home/oem/Llama-2-7b-hf")
parser.add_argument("--act_scales_path", type=str, default="/home/oem/smoothquant/act_scales/Llama-2-7b-hf.pt")
parser.add_argument("--n_samples", type=int, default=1) # 평가에 사용할 샘플 수
parser.add_argument("--smooth", type=int, default=1) # SmoothQuant 적용 여부
parser.add_argument("--quantize", type=int, default=1) # 양자화 적용 여부
parser.add_argument("--custom_attn", type=int, default=0) # 커스텀 어텐션 로직 사용 여부
parser.add_argument("--proj_list", type=str, default=None)

# 양자화 비트 설정 (가중치, 활성화 값, 에러) - 2025.05.15, Minki Choi
parser.add_argument("--wl_weight", type=int, default=8)
parser.add_argument("--wl_activate", type=int, default=8)
parser.add_argument("--wl_error", type=int, default=8)
parser.add_argument("--inference", type=int, default=1)

# Linear 레이어 관련 하드웨어 시뮬레이션 파라미터 (Analog PIM) - 2025.05.16, Minki Choi
parser.add_argument("--detect_L", type=float, default=2)
parser.add_argument("--cellBit_L", type=int, default=4)
parser.add_argument("--subArray_L", type=int, default=24)
parser.add_argument("--ADCprecision_L", type=int, default=8)
parser.add_argument("--ADCtype_L", type=str, default="linear")
parser.add_argument("--ADCgamma_L", type=float, default=1.0)
parser.add_argument("--k_L", type=float, default=10.0)
parser.add_argument("--center_L", type=float, default=0.3)
parser.add_argument("--bf_L", type=float, default=5.0)
parser.add_argument("--t_L", type=float, default=0)
parser.add_argument("--cycle_L", type=float, default=0)
parser.add_argument("--vari_L", type=float, default=0)
parser.add_argument("--v_L", type=float, default=0)
parser.add_argument("--target_L", type=float, default=0)

# KV 캐시/어텐션 관련 하드웨어 시뮬레이션 파라미터 (Analog PIM) - 2025.06.12, Beomjun Kim
parser.add_argument("--detect_KV", type=float, default=4)
parser.add_argument("--cellBit_KV", type=int, default=3)
parser.add_argument("--subArray_KV", type=int, default=128)
parser.add_argument("--ADCprecision_KV", type=int, default=11)
parser.add_argument("--ADCtype_KV", type=str, default="linear")
parser.add_argument("--ADCgamma_KV", type=float, default=1.0)
parser.add_argument("--t_KV", type=float, default=0)
parser.add_argument("--cycle_KV", type=float, default=0)
parser.add_argument("--vari_KV", type=float, default=0)
parser.add_argument("--v_KV", type=float, default=0)
parser.add_argument("--target_KV", type=float, default=0)

args = parser.parse_args()


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):

        # 데이터셋 텍스트를 하나로 합친 뒤 토큰화하여 GPU로 전송 - 2025.03.23, Minki Choi
        self.dataset = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids.to(device)
        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []

        # 샘플 수가 지정되지 않으면 전체 데이터셋을 2048 길이로 나눠서 처리
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):

            # 2048 컨텍스트 윈도우 단위로 배치 추출
            batch = self.dataset[:, (i * 2048):((i + 1) * 2048)].to(model.device)
            lm_logits = model(batch).logits

            # Loss 계산을 위해 로짓과 라벨 정렬 (Shift 처리)
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048):((i + 1) * 2048)][:, 1:]
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(loss.float() * 2048)

        # 모든 샘플의 Negative Log-Likelihood 평균을 계산하여 exp(PPL) 반환
        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))

# 토크나이저 및 데이터셋 로드 (Wikitext-2 사용) - 2025.03.22, Beomjun Kim
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=args.n_samples)
 
# 커스텀 어텐션(하드웨어 제약 조건 반영) 활성화 시 전역 설정 업데이트 - 2025.05.13, Minki Choi
if args.custom_attn == 1:
    custom_attention.WL_ACTIVATE = args.wl_activate
    custom_attention.WL_ERROR = args.wl_error
    custom_attention.WL_WEIGHT = args.wl_weight
    custom_attention.CYCLE_KV = args.cycle_KV
    custom_attention.CELLBIT_KV = args.cellBit_KV
    custom_attention.SUBARRAY_KV = args.subArray_KV
    custom_attention.ADCPRECISION_KV = args.ADCprecision_KV
    custom_attention.ADCTYPE_KV = args.ADCtype_KV
    custom_attention.ADCGAMMA_KV = args.ADCgamma_KV
    custom_attention.VARI_KV = args.vari_KV
    custom_attention.T_KV = args.t_KV
    custom_attention.V_KV = args.v_KV
    custom_attention.DETECT_KV = args.detect_KV
    custom_attention.TARGET_KV = args.target_KV
    LlamaAttention.forward = custom_forward

# 사전 학습된 모델 로드 (BFloat16 정밀도) - 2025.03.24, Minki Choi
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
 
# SmoothQuant 적용: 활성화 값의 아웃라이어를 가중치로 전이 - 2025.03.31, Minki Choi
if args.smooth == 1:
    act_scales = torch.load(args.act_scales_path)
    smooth_lm(model, act_scales, args.alpha)

# Fake Quantization 적용: 하드웨어 특성(ADC, Vari, Cycle 등) 반영  - 2025.04.04, Minki Choi
if args.quantize == 1:
    model = quantize_model(
        model,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_bmm_input=True,
        wl_activate=args.wl_activate,
        wl_error=args.wl_error,
        wl_weight=args.wl_weight,
        inference=args.inference,
        cycle_L=args.cycle_L,
        cellBit_L=args.cellBit_L,
        subArray_L=args.subArray_L,
        ADCprecision_L=args.ADCprecision_L,
        ADCtype_L=args.ADCtype_L,
        ADCgamma_L=args.ADCgamma_L,
        k_L=args.k_L,
        center_L=args.center_L,
        bf_L=args.bf_L,
        vari_L=args.vari_L,
        t_L=args.t_L,
        v_L=args.v_L,
        detect_L=args.detect_L,
        target_L=args.target_L,
    )

# 최종 결과 출력 # 2025.04.11, Minki Choi
print(f"Perplexity: {evaluator.evaluate(model)}")