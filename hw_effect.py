import math
import torch

def Retention(cell_div, t, v, detect, target, temp):
    """소자 타입(detect)별 시간에 따른 데이터 감쇠(Retention Loss) 계산"""
    # 2025.05.28, Minki Choi

    if detect == 1: # RRAM(HZO 5nm)
        ratio = math.exp(t / -3.63e07)

    elif detect == 2 and t > 0: # RRAM
        ratio = t ** (-0.00229)

    elif detect == 3: # VRRAM(HZO 10nm)
        ratio = math.exp(t / -1.89e09)

    elif detect == 4 and t > 0: # VRRAM(HZO 10nm)
        ratio = t ** (-0.04487)

    else:
        ratio = 1
        
    return cell_div * ratio, ratio


def apply_noise(cell_div, detect, cellbit, cell_range, cycle):
    """소자 물리 특성(Ion, Ioff) 및 Write Cycle 기반 노이즈/전도도 적용"""
    # 2025.05.22, Minki Choi

    if detect == 1: # RRAM(HZO 5nm)
        sd_mapping = {0: 0.0260, 1: 0.0176, 2: 0.0143, 3: 0.0082, 4: 0.0077, 5: 0.0073, 6: 0.0046, 7: 0.0055}

    elif detect == 2: # RRAM
        sd_mapping = {i: 0.0 for i in range(cell_range)}

    elif detect == 3: # VRRAM(HZO 10nm)
        sd_mapping = {
            0: 0.0120, 1: 0.0020, 2: 0.0016, 3: 0.0018, 4: 0.0011, 5: 0.0006, 6: 0.0007, 7: 0.0009,
            8: 0.0009, 9: 0.0005, 10: 0.0020, 11: 0.0010, 12: 0.0010, 13: 0.0029, 14: 0.0006, 15: 0.0009,
        }

    elif detect == 4: # VRRAM(HZO 10nm)
        sd_mapping = {i: 0.0 for i in range(cell_range)}
        
    else:
        raise ValueError(f"Unsupported detect value: {detect}")

    # 가우시안 노이즈 생성
    stds = torch.tensor([sd_mapping[i] for i in range(cell_range)], dtype=torch.float32, device=cell_div.device)
    noise = torch.normal(mean=0.0, std=stds[cell_div.long()])

    # 소자/사이클 조합에 따른 물리 파라미터 로드
    device_params = {
        # RRAM(HZO 5nm)
        (1, 0): {"Ioff": 30.22e-6, "Ion": 561.85e-6, "Vread": 0.3},
        (1, 500): {"Ioff": 30.07e-6, "Ion": 559.34e-6, "Vread": 0.3},
        (1, 1000): {"Ioff": 29.757e-6, "Ion": 571.26e-6, "Vread": 0.3},
        (1, 1500): {"Ioff": 30.07e-6, "Ion": 562.16e-6, "Vread": 0.3},
        (1, 2000): {"Ioff": 30.07e-6, "Ion": 566.24e-6, "Vread": 0.3},
        (1, 2500): {"Ioff": 29.28e-6, "Ion": 567.18e-6, "Vread": 0.3},

        # RRAM
        (2, 0): {"Ioff": 1.3589e-06, "Ion": 23.334e-06, "Vread": 0.5},
        (2, 50): {"Ioff": 1.24e-06, "Ion": 23.147e-06, "Vread": 0.5},
        (2, 100): {"Ioff": 1.3712e-06, "Ion": 23.196e-06, "Vread": 0.5},
        (2, 150): {"Ioff": 0.807e-06, "Ion": 23.48e-06, "Vread": 0.5},
        (2, 180): {"Ioff": 0.791e-06, "Ion": 23.413e-06, "Vread": 0.5},

        # VRRAM(HZO 10nm)
        (3, 0): {"Ioff": 1.0087e-05, "Ion": 2.9567e-04, "Vread": 0.5},
        (3, 200): {"Ioff": 1.0086e-05, "Ion": 2.988e-04, "Vread": 0.5},
        (3, 400): {"Ioff": 1.0086e-05, "Ion": 3.0478e-04, "Vread": 0.5},
        (3, 600): {"Ioff": 1.0087e-05, "Ion": 3.0101e-04, "Vread": 0.5},
        (3, 800): {"Ioff": 1.0086e-05, "Ion": 3.0164e-04, "Vread": 0.5},
        (3, 1000): {"Ioff": 1.0090e-05, "Ion": 2.9441e-04, "Vread": 0.5},
        (3, 1200): {"Ioff": 1.0090e-05, "Ion": 2.9400e-04, "Vread": 0.5},

        # FTJ(HZO 5nm)
        (4, 0): {"Ioff": 11.97e-07, "Ion": 7.433e-05, "Vread": 1.5},
        (4, 200): {"Ioff": 5.539e-07, "Ion": 7.159e-05, "Vread": 1.5},
        (4, 400): {"Ioff": 11.03e-07, "Ion": 7.06e-05, "Vread": 1.5},
        (4, 600): {"Ioff": 7.315e-07, "Ion": 7.027e-05, "Vread": 1.5},
        (4, 800): {"Ioff": 6.083e-07, "Ion": 7.0384e-05, "Vread": 1.5},
        (4, 1000): {"Ioff": 10.35e-07, "Ion": 6.839e-05, "Vread": 1.5},
    }

    params = device_params.get((detect, cycle))
    if params is None:
        raise ValueError("Invalid detect value or cycle combination")

    ioff = params["Ioff"]
    ion = params["Ion"]
    vread = params["Vread"]

    # 디지털 레벨 -> 전도도(Conductance) 변환 및 노이즈 합산
    gmin = ioff / vread
    gmax = ion / vread
    cell_div_v = (((gmax - gmin) / (2**cellbit - 1)) * cell_div + gmin) * (1 + noise)
    return cell_div_v, vread, ioff, ion


def ADC_compute_new(input_tensor, ion, ioff, sub_array, adc_precision):
    """컬럼 출력 전류에 대한 ADC 양자화 수행"""
    # 2025.04.26, Beomjun Kim

    column_max = sub_array * ion

    adc_delta = column_max / (2**adc_precision - 1)
    output = torch.floor((input_tensor + adc_delta / 2) / adc_delta) * adc_delta
    output = torch.clamp(output, 0, column_max)

    return output, adc_delta


def ADC_new_correction(adc_out, dummy_out, ion, ioff, cellbit, retention_ratio):
    """물리적 오차 및 감쇠가 반영된 ADC 출력에서 디지털 값 복원"""
    # 2025.04.26, Beomjun Kim

    return (((adc_out - dummy_out) / ((ion - ioff) / (2**cellbit - 1))) / retention_ratio).round()