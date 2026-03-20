import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

import hw_effect # Analog PIM 하드웨어 효과 추가 - 2025.05.16, Minki Choi
from fake_quant import TARGET_DTYPE

# 하드웨어 시뮬레이션 기본 파라미터 설정 (KV 캐시 및 어텐션용) - 2025.05.16, Minki Choi
WL_ACTIVATE = 8
WL_ERROR = 8
WL_WEIGHT = 8
CYCLE_KV = 0
CELLBIT_KV = 3
SUBARRAY_KV = 256
ADCPRECISION_KV = 16
ADCTYPE_KV = "linear"
ADCGAMMA_KV = 1.0
VARI_KV = 0
T_KV = 0
V_KV = 0
DETECT_KV = 0
TARGET_KV = 0
BLOCK_IDX_LIST = list(range(40))

# 기존 LlamaAttention의 forward 함수 백업
original_attention_forward = LlamaAttention.forward


def custom_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    """하드웨어 노이즈가 반영된 Llama Attention Forward 루틴"""

    if "padding_mask" in kwargs:
        warnings.warn("Passing `padding_mask` is deprecated; use `attention_mask` instead.")

    # 대상 블록이 아니면 기본 forward 수행 - 2024.06.26, Minki Choi
    if not hasattr(self, "layer_idx") or self.layer_idx not in BLOCK_IDX_LIST:
        return original_attention_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

    bsz, q_len, _ = hidden_states.size()

    # 1. Q, K, V Projection 수행 (병렬 처리 지원) - 2024.06.26, Minki Choi
    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
        query_states = torch.cat([F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
        key_states = torch.cat([F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
        value_states = torch.cat([F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    # 2. 로터리 임베딩(RoPE) 및 KV 캐시 업데이트 - 2024.05.15, Minki Choi
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError("Layer index is required for cached decoding.")
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos}
        )

    # GQA(Grouped Query Attention)를 위한 KV 복제 - 2024.05.15, Minki Choi
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # 3. QK Matmul (하드웨어 시뮬레이션 적용) - 2024.05.15, Minki Choi
    attn_weights = run_qk_matmul(query_states, key_states)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Softmax 및 Dropout - 2024.05.15, Minki Choi
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    # 4. PV Matmul (하드웨어 시뮬레이션 적용) 2024.05.17, Beomjun Kim
    attn_output = run_pv_matmul(attn_weights, value_states).to(torch.bfloat16)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)

    # 5. Output Projection (O_proj) 2024.05.15, Minki Choi
    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum(F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp))
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def run_qk_matmul(query_states, key_states):
    """Q와 K의 행렬곱 시 PIM 하드웨어 오차를 시뮬레이션"""

    query_states_norm, q_scale, q_zero = normalize_attention_states(query_states)
    key_states_norm, k_scale, k_zero = normalize_attention_states(key_states)
    cell_range = 2**CELLBIT_KV
    num_sub_array_k = math.ceil(key_states.shape[3] / SUBARRAY_KV)

    query_states = query_states.to(TARGET_DTYPE)
    key_states = key_states.to(TARGET_DTYPE)
    q_scale = q_scale.to(TARGET_DTYPE)
    k_scale = k_scale.to(TARGET_DTYPE)
    q_zero = q_zero.to(TARGET_DTYPE)
    k_zero = k_zero.to(TARGET_DTYPE)
    query_states_norm = query_states_norm.to(TARGET_DTYPE)
    key_states_norm = key_states_norm.to(TARGET_DTYPE)

    # 연산용 텐서 전처리 - 2024.05.15, Minki Choi
    batch_size0, num_heads0, seq_len0, _ = query_states_norm.shape
    q_scale = q_scale.reshape(batch_size0, num_heads0, seq_len0, 1)
    k_scale = k_scale.reshape(batch_size0, num_heads0, seq_len0, 1)
    q_zero = q_zero.reshape(batch_size0, num_heads0, seq_len0, 1)
    k_zero = k_zero.reshape(batch_size0, num_heads0, seq_len0, 1)
    sq = q_scale.expand_as(query_states_norm)
    sk = k_scale.expand_as(key_states_norm)
    zq = q_zero.expand_as(query_states_norm)
    zk = k_zero.expand_as(key_states_norm)

    output_of_z_sum = torch.zeros_like(torch.matmul(query_states_norm, key_states_norm.transpose(2, 3)))

    # 활성화 값 비트 슬라이싱 루프 - 2025.04.28, Minki Choi
    for z in range(WL_ACTIVATE):
        x_lsb = query_states_norm % 2
        query_states_norm = torch.div(query_states_norm - x_lsb, 2, rounding_mode="floor")
        output_of_s_sum = torch.zeros_like(output_of_z_sum)

        # 서브 어레이 분할 루프 - 2025.04.28, Minki Choi
        for s in range(num_sub_array_k):
            mask = torch.zeros_like(key_states.transpose(2, 3))
            start_idx = s * SUBARRAY_KV
            end_idx = min((s + 1) * SUBARRAY_KV, key_states.transpose(2, 3).shape[2])
            mask[:, :, start_idx:end_idx, :] = 1
            weight_div = key_states_norm.transpose(2, 3) * mask
            output_of_s_sum_p = torch.zeros_like(output_of_s_sum)

            # 가중치 셀 비트 슬라이싱 루프 - 2025.04.28, Minki Choi
            for k in range(math.ceil(WL_WEIGHT / CELLBIT_KV)):
                cell_div = weight_div % cell_range
                weight_div = torch.div(weight_div - cell_div, cell_range, rounding_mode="floor")

                # 물리 노이즈 및 Retention 반영 - 2025.04.28, Minki Choi
                cell_div_v, vread, ioff, ion = hw_effect.apply_noise(cell_div, DETECT_KV, CELLBIT_KV, cell_range, CYCLE_KV)

                t_kv_scalar = torch.as_tensor(T_KV, device=cell_div_v.device, dtype=cell_div_v.dtype)

                # 시간 경과에 따른 감쇠 모사 (Retention) - 2025.05.02, Beomjun Kim
                factors = torch.linspace(1.0, 1e-8, steps=2048, device=cell_div_v.device, dtype=cell_div_v.dtype)
                retention_t = torch.pow(factors * t_kv_scalar, -0.04487).view(1, 1, 1, 2048)
                cell_div_v = cell_div_v * retention_t.clamp(max=1).expand_as(cell_div_v)

                # 아날로그 도메인 행렬곱 및 ADC 변환 - 2025.04.28, Minki Choi
                output_of_k_sum = torch.matmul(x_lsb * vread, cell_div_v * mask)
                output_of_dummy_sum = torch.matmul(x_lsb.to(torch.float32), (mask * ioff).to(torch.float32))

                if ADCTYPE_KV != "linear":
                    raise ValueError(f"Unsupported ADCtype_KV: {ADCTYPE_KV}")

                adc_output, _ = hw_effect.ADC_compute_new(output_of_k_sum, ion, ioff, SUBARRAY_KV, ADCPRECISION_KV)
                adc_dummy, _ = hw_effect.ADC_compute_new(output_of_dummy_sum, ion, ioff, SUBARRAY_KV, ADCPRECISION_KV)

                # 디지털 복원 - 2025.05.29, Minki Choi
                corrected = (adc_output - adc_dummy) / ((ion - ioff) / (2**CELLBIT_KV - 1))
                output_of_s_sum_p.add_(corrected.round() * (cell_range**k))

            output_of_s_sum.add_(output_of_s_sum_p)

        output_of_z_sum.add_(output_of_s_sum * (2**z))

    # 제로포인트 보정 및 최종 스케일 복구 - 2025.05.15, Minki Choi
    a = torch.matmul(query_states / sq, zk.transpose(2, 3))
    b = torch.matmul(zq, (key_states / sk).transpose(2, 3))
    c = torch.matmul(zq, zk.transpose(2, 3))
    subtracted_weights = output_of_z_sum - a - b - c

    return (subtracted_weights * q_scale * k_scale.transpose(2, 3)) / math.sqrt(query_states.shape[-1])


def run_pv_matmul(attn_weights, value_states):
    """Attention Weights와 Value의 행렬곱 시 하드웨어 오차 시뮬레이션"""

    num_sub_array_v = math.ceil(value_states.shape[2] / SUBARRAY_KV)

    # 8비트 정규화 (Value는 컬럼 단위 정규화 적용) - 2025.05.20, Minki Choi
    value_states_norm, v_scale, v_zero = normalize_value_by_column(value_states)
    attn_weights_norm = (attn_weights * 255).round()

    attn_weights = attn_weights.to(TARGET_DTYPE)
    value_states_norm = value_states_norm.to(TARGET_DTYPE)
    v_scale = v_scale.to(TARGET_DTYPE)
    v_zero = v_zero.to(TARGET_DTYPE)
    attn_weights_norm = attn_weights_norm.to(TARGET_DTYPE)

    sv = v_scale.expand_as(value_states_norm)
    zv = v_zero.expand_as(value_states_norm)
    cell_range = 2**CELLBIT_KV
    output_of_z_sum = torch.zeros_like(torch.matmul(attn_weights_norm, value_states_norm))

    # Bit-slicing 루프 (QK 연산과 유사한 로직) - 2025.04.28, Minki Choi
    for z in range(WL_ACTIVATE):
        x_lsb = attn_weights_norm % 2
        attn_weights_norm = torch.div(attn_weights_norm - x_lsb, 2, rounding_mode="floor")
        output_of_s_sum = torch.zeros_like(output_of_z_sum)

        for s in range(num_sub_array_v):
            mask = torch.zeros_like(value_states_norm)
            start_idx = s * SUBARRAY_KV
            end_idx = min((s + 1) * SUBARRAY_KV, value_states_norm.shape[2])
            mask[:, :, start_idx:end_idx, :] = 1
            weight_div = value_states_norm * mask
            output_of_s_sum_p = torch.zeros_like(output_of_s_sum)

            for k in range(math.ceil(WL_WEIGHT / CELLBIT_KV)):
                cell_div = weight_div % cell_range
                weight_div = torch.div(weight_div - cell_div, cell_range, rounding_mode="floor")
                cell_div_v, vread, ioff, ion = hw_effect.apply_noise(cell_div, DETECT_KV, CELLBIT_KV, cell_range, CYCLE_KV)

                t_kv_scalar = torch.as_tensor(T_KV, device=cell_div_v.device, dtype=cell_div_v.dtype)

                # Retention 효과 (2048 시퀀스 길이 기준) - 2025.05.02, Beomjun Kim
                factors = torch.linspace(1.0, 1e-8, steps=2048, device=cell_div_v.device, dtype=cell_div_v.dtype)
                retention_t = torch.pow(factors * t_kv_scalar, -0.04487).view(1, 1, 2048, 1)
                cell_div_v = cell_div_v * retention_t.clamp(max=1).expand_as(cell_div_v)

                output_of_k_sum = torch.matmul(x_lsb * vread, cell_div_v * mask)
                output_of_dummy_sum = torch.matmul(x_lsb.to(torch.float32), (mask * ioff).to(torch.float32))

                if ADCTYPE_KV != "linear":
                    raise ValueError(f"Unsupported ADCtype_KV: {ADCTYPE_KV}")

                adc_output, _ = hw_effect.ADC_compute_new(output_of_k_sum, ion, ioff, SUBARRAY_KV, ADCPRECISION_KV)
                adc_dummy, _ = hw_effect.ADC_compute_new(output_of_dummy_sum, ion, ioff, SUBARRAY_KV, ADCPRECISION_KV)
                corrected = (adc_output - adc_dummy) / ((ion - ioff) / (2**CELLBIT_KV - 1))
                output_of_s_sum_p.add_(corrected.round() * (cell_range**k))

            output_of_s_sum.add_(output_of_s_sum_p)

        output_of_z_sum.add_(output_of_s_sum * (2**z))

    # 제로포인트 보정 및 복원 - 2025.04.29, Minki Choi
    d = torch.matmul((attn_weights * 255).round(), zv)

    return (output_of_z_sum - d) * sv / 255


def normalize_attention_states(states):
    # 2025.04.19, Minki Choi
    states_view = states.contiguous().view(-1, states.shape[-1])
    states_min = states_view.min(dim=-1, keepdim=True)[0]
    states_max = states_view.max(dim=-1, keepdim=True)[0]
    scale = (states_max - states_min).clamp_(min=1e-5) / 255
    zero_point = (-states_min / scale).round().clamp_(0, 255)
    normalized_states = (states / scale.view(*states.shape[:-1], 1) + zero_point.view(*states.shape[:-1], 1)).round().clamp_(0, 255)

    return normalized_states, scale, zero_point


def normalize_value_by_column(states):
    """Value 텐서를 컬럼(head_dim) 단위로 8비트 정규화"""
    # 2025.04.22, Minki Choi

    _, num_heads, _, head_dim = states.shape
    states_reshaped = states.contiguous().permute(1, 3, 0, 2).reshape(num_heads, head_dim, -1)
    states_min = states_reshaped.min(dim=2, keepdim=True)[0]
    states_max = states_reshaped.max(dim=2, keepdim=True)[0]
    scale = (states_max - states_min).clamp_(min=1e-5) / 255
    zero_point = (-states_min / scale).round().clamp_(0, 255)
    scale_reshaped = scale.reshape(1, num_heads, 1, head_dim)
    zero_point_reshaped = zero_point.reshape(1, num_heads, 1, head_dim)
    normalized_states = (states / scale_reshaped + zero_point_reshaped).round().clamp_(0, 255)

    return normalized_states, scale_reshaped, zero_point_reshaped