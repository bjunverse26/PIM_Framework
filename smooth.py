import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    """
    Normalization 레이어와 이후 연결된 Linear 레이어 간의 가중치를 조절하여 
    활성화 값(Activation)의 아웃라이어를 완화
    """
    if not isinstance(fcs, list):
        fcs = [fcs]
        
    # 입력 레이어 타입 및 파라미터 크기 검증
    assert isinstance(ln, LlamaRMSNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    
    # 각 Linear 레이어의 가중치 절대값 최대치 계산
    weight_scales = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    
    # SmoothQuant 공식에 따라 가중치와 활성화 값의 균형을 맞추는 스케일 계산
    scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5).to(device=device, dtype=dtype)

    # 노멀라이제이션 레이어 가중치에 스케일 적용
    ln.weight.div_(scales)
    # 선형 레이어 가중치에 스케일 반영
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    """
    Llama 구조 모델을 순회하며 디코더 레이어별로 Smoothing 적용
    """
    for name, module in model.named_modules():
        if not isinstance(module, LlamaDecoderLayer):
            continue

        # 1. Self-Attention 영역 스무딩 (Q, K, V Projections 대상)
        attn_ln = module.input_layernorm
        qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
        smooth_ln_fcs_llama_like(attn_ln, qkv, scales[name + ".self_attn.q_proj"], alpha)

        # 2. MLP 영역 스무딩 (Gate, Up Projections 대상)
        ffn_ln = module.post_attention_layernorm
        fcs = [module.mlp.gate_proj, module.mlp.up_proj]
        smooth_ln_fcs_llama_like(ffn_ln, fcs, scales[name + ".mlp.gate_proj"], alpha)