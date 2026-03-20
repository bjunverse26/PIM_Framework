import math

import torch
from torch import nn

import hw_effect # Analog PIM 하드웨어 효과 추가 - 2025.04.24, Chan-Gi Yook

# 연산 정밀도 설정 - 2025.04.22, Chan-Gi Yook
FP_PRECISION = 32

if FP_PRECISION == 16:
    TARGET_DTYPE = torch.float16

elif FP_PRECISION == 32:
    TARGET_DTYPE = torch.float32

elif FP_PRECISION == 64:
    TARGET_DTYPE = torch.float64

else:
    raise ValueError(f"Unsupported FP_PRECISION: {FP_PRECISION}")

# 하드웨어 시뮬레이션을 적용할 타겟 레이어 및 블록 정의 - 2025.05.07, Chan-Gi Yook
LAYER_HW_CONFIG = []
PROJ_NAME_LIST = ["q_proj", "k_proj", "v_proj", "o_proj"]
BLOCK_IDX_LIST = [0, 1, 2, 3, 4, 5, 6, 7]


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, wl_weight=8):
    """가중치 채널별 Absmax 양자화 및 스케일/제로포인트 계산"""
    # 2025.04.06, Minki Choi

    if TARGET_DTYPE is not None and w.dtype != TARGET_DTYPE:
        w = w.to(TARGET_DTYPE)

    w_min = w.min(dim=-1, keepdim=True)[0]
    w_max = w.max(dim=-1, keepdim=True)[0]
    q_max = 2**wl_weight - 1
    scales = (w_max - w_min).clamp_(min=1e-5) / q_max
    zero_points = (-w_min / scales).round().clamp_(0, q_max)
    w_norm = (w / scales + zero_points).round().clamp_(0, q_max)

    return (w_norm - zero_points) * scales, scales, zero_points


@torch.no_grad()
def quantize_activation_per_token_absmax(t, wl_activate=8):
    """활성화 값 토큰별 Absmax 양자화"""
    # 2025.04.06, Minki Choi

    if TARGET_DTYPE is not None and t.dtype != TARGET_DTYPE:
        t = t.to(TARGET_DTYPE)

    t_min = t.min(dim=-1, keepdim=True)[0]
    t_max = t.max(dim=-1, keepdim=True)[0]
    q_max = 2**wl_activate - 1
    scales = (t_max - t_min).clamp_(min=1e-5) / q_max
    zero_points = (-t_min / scales).round().clamp_(0, q_max)
    t_norm = (t / scales + zero_points).round().clamp_(0, q_max)

    return (t_norm - zero_points) * scales, scales, zero_points


class W8A8Linear(nn.Module):
    """
    8비트 양자화 및 하드웨어(Analog PIM) 에러 시뮬레이션을 포함한 Linear 레이어
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        proj_name=None,
        layer_idx=None,
        block_idx=None,
        wl_activate=8,
        wl_error=8,
        wl_weight=8,
        inference=1,
        cycle=10,
        cellBit=1,
        subArray=128,
        ADCprecision=5,
        ADCtype="linear",
        ADCgamma=1.0,
        k=10.0,
        center=0.3,
        bf=0.5,
        vari=0,
        t=0,
        v=0,
        detect=0,
        target=0,
        model="llama",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 가중치 및 파라미터 등록 - 2025.04.23, Chan-Gi Yook
        self.register_buffer("weight", torch.randn(out_features, in_features, dtype=torch.bfloat16, requires_grad=False))

        if bias:
            self.register_buffer("bias", torch.zeros((1, out_features), dtype=torch.bfloat16, requires_grad=False))

        else:
            self.register_buffer("bias", None)

        self.proj_name = proj_name
        self.layer_idx = layer_idx
        self.block_idx = block_idx
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_weight = wl_weight
        self.inference = inference
        self.cycle = cycle
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.ADCtype = ADCtype
        self.ADCgamma = ADCgamma
        self.k = k
        self.center = center
        self.bf = bf
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.model = model

        # 2025.04.03, Minki Choi
        if act_quant != "per_token":
            raise ValueError(f"Unsupported act_quant: {act_quant}")
        self.act_quant_name = "per_token"
        self.act_quant = self.custom_quantize_activation_per_token

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def custom_quantize_activation_per_token(self, t):
        """포워드 시 사용할 활성화 값 양자화 함수"""
        # 2025.04.03, Minki Choi

        if TARGET_DTYPE is not None and t.dtype != TARGET_DTYPE:
            t = t.to(TARGET_DTYPE)

        t, scales, zero_points = quantize_activation_per_token_absmax(t, self.wl_activate)
        self.act_scales = scales
        self.act_zero_points = zero_points

        return t

    @torch.no_grad()
    def forward(self, x):
        # Act/Weight/Cell Divide Linear 연산 - 2025.04.24, Minki Choi

        x = x.to(TARGET_DTYPE)
        weight = self.weight.to(TARGET_DTYPE)
        bias = self.bias.to(TARGET_DTYPE) if self.bias is not None else None

        # 1. 입력 및 가중치 정규화 (Integer 영역으로 변환)
        q_x = self.act_quant(x)
        sa = self.act_scales.to(TARGET_DTYPE).expand_as(q_x).contiguous()
        sw = self.weight_scales.to(TARGET_DTYPE).expand_as(weight).contiguous()
        za = self.act_zero_points.to(TARGET_DTYPE).expand_as(q_x).contiguous()
        zw = self.weight_zero_points.to(TARGET_DTYPE).expand_as(weight).contiguous()
        normalized_weight = (weight / sw + zw).round().clamp(0, 2**self.wl_weight - 1)
        normalized_x = (q_x / sa + za).round().clamp(0, 2**self.wl_activate - 1)

        # 2. 하드웨어 효과 시뮬레이션 (특정 레이어/블록인 경우)
        if self.proj_name in PROJ_NAME_LIST and self.block_idx in BLOCK_IDX_LIST:
            output_of_z_sum = torch.zeros_like(torch.nn.functional.linear(x, weight, bias))
            cell_range = 2**self.cellBit
            num_sub_array = math.ceil(weight.shape[1] / self.subArray)

            # 활성화 값 비트 단위 순회 (Bit-line 연산) - 2025.04.28, Minki Choi
            for z in range(int(self.wl_activate)):
                x_lsb = normalized_x % 2
                normalized_x = torch.div(normalized_x - x_lsb, 2, rounding_mode="floor")
                output_of_s_sum = torch.zeros_like(output_of_z_sum)

                # 서브 어레이별 분할 연산 - 2025.04.28, Minki Choi
                for s in range(num_sub_array):
                    mask = torch.zeros_like(weight)
                    start_idx = s * self.subArray
                    end_idx = min((s + 1) * self.subArray, weight.shape[1])
                    mask[:, start_idx:end_idx] = 1

                    weight_div = normalized_weight * mask
                    output_of_s_sum_p = torch.zeros_like(output_of_s_sum)

                    # 가중치 셀 비트 단위 순회 - 2025.04.28, Minki Choi
                    for k in range(math.ceil(self.wl_weight / self.cellBit)):
                        cell_div = weight_div % cell_range
                        weight_div = torch.div(weight_div - cell_div, cell_range, rounding_mode="floor")

                        # 물리적 노이즈 및 Retention 에러 적용 - 2025.04.30, Beomjun Kim
                        cell_div_v, vread, ioff, ion = hw_effect.apply_noise(
                            cell_div, self.detect, self.cellBit, cell_range, self.cycle
                        )
                        cell_div_v, retention_ratio = hw_effect.Retention(
                            cell_div_v, self.t, self.v, self.detect, self.target, 300
                        )

                        # 아날로그 선형 연산 수행 - 2025.04.30, Beomjun Kim
                        output_of_k_sum = torch.nn.functional.linear(x_lsb * vread, cell_div_v * mask, bias)
                        output_of_dummy_sum = torch.nn.functional.linear(x_lsb, mask * ioff, bias)

                        if self.ADCtype != "linear":
                            raise ValueError(f"Unsupported ADCtype: {self.ADCtype}")

                        # ADC 변환 및 디지털 복원(Correction) - 2025.05.02, Beomjun Kim
                        adc_output, _ = hw_effect.ADC_compute_new(output_of_k_sum, ion, ioff, self.subArray, self.ADCprecision)
                        adc_dummy, _ = hw_effect.ADC_compute_new(output_of_dummy_sum, ion, ioff, self.subArray, self.ADCprecision)
                        corrected = hw_effect.ADC_new_correction(
                            adc_output, adc_dummy, ion, ioff, self.cellBit, retention_ratio
                        )
                        output_of_s_sum_p.add_(corrected * (cell_range**k))

                    output_of_s_sum.add_(output_of_s_sum_p)

                output_of_z_sum.add_(output_of_s_sum * (2**z))

            sa_v = self.act_scales.to(TARGET_DTYPE)
            sw_v = self.weight_scales.to(TARGET_DTYPE)
            za_v = self.act_zero_points.to(TARGET_DTYPE)
            zw_v = self.weight_zero_points.to(TARGET_DTYPE)

            # 3. 양자화 수식에 따른 제로포인트 보정 (De-quantization 준비) - 2025.04.24, Minki Choi
            a_new = torch.matmul(q_x.sum(dim=2, keepdim=True) / sa_v, zw_v.transpose(0, 1))
            b_new = torch.matmul(za_v, (weight.sum(dim=1, keepdim=True) / sw_v).transpose(0, 1))
            width = 11008 if self.proj_name == "down_proj" else 4096
            c_new = torch.matmul(za_v, zw_v.transpose(0, 1)).mul_(width)

            subtract_a = output_of_z_sum - c_new - b_new - a_new
            sw_t = self.weight_scales.to(TARGET_DTYPE).transpose(0, 1).unsqueeze(0)
            subtract_a_scaled = subtract_a * sw_t * self.act_scales.to(TARGET_DTYPE)

        else:
            # 시뮬레이션 비대상 레이어는 일반 FP 연산 수행 - 2025.04.29, Minki Choi
            q_x_same_dtype = q_x.to(weight.dtype) if q_x.dtype != weight.dtype else q_x
            subtract_a_scaled = torch.nn.functional.linear(q_x_same_dtype, weight, bias)

        return self.output_quant(subtract_a_scaled).to(torch.bfloat16)

    @staticmethod
    def from_float(module, weight_quant="per_channel", act_quant="per_token", quantize_output=False, **kwargs):

        if weight_quant != "per_channel":
            raise ValueError(f"Unsupported weight_quant: {weight_quant}")
        if act_quant != "per_token":
            raise ValueError(f"Unsupported act_quant: {act_quant}")

        # 기존 nn.Linear 레이어를 W8A8Linear로 변환 - 2025.04.23, Chan-Gi Yook
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            **kwargs,
        )

        # 가중치 양자화 및 스케일 정보 저장 - 2025.04.03, Minki Choi
        weight = module.weight.to(TARGET_DTYPE) if module.weight.dtype != TARGET_DTYPE else module.weight
        w_dequant, new_module.weight_scales, new_module.weight_zero_points = quantize_weight_per_channel_absmax(
            weight, new_module.wl_weight
        )
        new_module.weight = w_dequant.to(torch.bfloat16)

        if module.bias is not None:
            new_module.bias = module.bias
        new_module.weight_quant_name = weight_quant

        return new_module


def quantize_llama_like(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, **kwargs):
    """Llama 모델의 모든 레이어를 순회하며 양자화 레이어로 교체"""

    for block_idx, block in enumerate(model.model.layers):
        specs = [
            ("q_proj", block.self_attn, True),
            ("k_proj", block.self_attn, True),
            ("v_proj", block.self_attn, True),
            ("o_proj", block.self_attn, True),
            ("gate_proj", block.mlp, False),
            ("up_proj", block.mlp, False),
            ("down_proj", block.mlp, False),
        ]
        for layer_idx, (proj_name, parent, quant_out) in enumerate(specs):
            # 레이어별 개별 HW 설정이 있으면 적용, 없으면 기본값 사용 - 2025.04.03, Minki Choi

            used_adc = kwargs["ADCprecision_L"]
            used_cell = kwargs["cellBit_L"]

            if block_idx < len(LAYER_HW_CONFIG) and layer_idx < len(LAYER_HW_CONFIG[block_idx]):
                cfg = LAYER_HW_CONFIG[block_idx][layer_idx]
                used_adc = cfg.get("ADCprecision", used_adc)
                used_cell = cfg.get("cellBit", used_cell)

            setattr(
                parent,
                proj_name,
                W8A8Linear.from_float(
                    getattr(parent, proj_name),
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quant_out and quantize_bmm_input,
                    proj_name=proj_name,
                    block_idx=block_idx,
                    layer_idx=layer_idx,
                    wl_activate=kwargs["wl_activate"],
                    wl_error=kwargs["wl_error"],
                    wl_weight=kwargs["wl_weight"],
                    inference=kwargs["inference"],
                    cycle=kwargs["cycle_L"],
                    cellBit=used_cell,
                    subArray=kwargs["subArray_L"],
                    ADCprecision=used_adc,
                    ADCtype=kwargs["ADCtype_L"],
                    ADCgamma=kwargs["ADCgamma_L"],
                    k=kwargs["k_L"],
                    center=kwargs["center_L"],
                    bf=kwargs["bf_L"],
                    vari=kwargs["vari_L"],
                    t=kwargs["t_L"],
                    v=kwargs["v_L"],
                    detect=kwargs["detect_L"],
                    target=kwargs["target_L"],
                ),
            )

    return model


def quantize_model(model, **kwargs):
    """모델 타입 체크 및 양자화 시작"""

    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel

    if not isinstance(model, LlamaPreTrainedModel):
        raise ValueError(f"Unsupported model type: {type(model)}")
    
    return quantize_llama_like(model, **kwargs)