"""
export.py — Export ONNX et quantification INT8 pour déploiement embarqué.

Workflow cible :
    PyTorch (float32) → ONNX → ONNX Runtime (RPi, Jetson)
    PyTorch (float32) → Quantification INT8 → ONNX quantifié

Compatibilité :
    - Raspberry Pi 4 / 5         : onnxruntime (pip install onnxruntime)
    - Jetson Nano / Orin         : onnxruntime-gpu
    - STM32 / microcontrôleur    : STM32Cube.AI (importer le .onnx)
    - Android / iOS embarqué     : ONNX Runtime Mobile
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_onnx(
    model: nn.Module,
    onnx_path: str,
    window_size: int = 20,
    input_size: int = 7,
    opset_version: int = 13,
) -> None:
    """
    Exporte le modèle PyTorch vers ONNX.

    Le modèle est mis en eval() et les gradients sont désactivés.
    L'entrée dummy simule une séquence de longueur window_size.
    """
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    model.eval()
    dummy = torch.zeros(1, window_size, input_size, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            opset_version=opset_version,
            input_names=["input_sequence"],
            output_names=["residual_correction"],
            dynamic_axes={
                "input_sequence":      {0: "batch_size"},
                "residual_correction": {0: "batch_size"},
            },
            do_constant_folding=True,   # optimisation graphe
            verbose=False,
        )

    size_mb = os.path.getsize(onnx_path) / 1e6
    logger.info(f"ONNX exporté → {onnx_path} ({size_mb:.2f} MB)")

    # Vérification rapide du graphe
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX : graphe valide ✓")
    except ImportError:
        logger.warning("onnx non installé — skip vérification (pip install onnx)")
    except Exception as e:
        logger.warning(f"ONNX check warning : {e}")


def quantize_int8(
    model: nn.Module,
    int8_path: str,
    window_size: int = 20,
    input_size: int = 7,
) -> nn.Module:
    """
    Quantification dynamique INT8 post-training.

    Cible les couches Linear et GRU (ou Conv1d pour TCN).
    Réduit la taille du modèle d'environ 4× avec une dégradation
    de précision typiquement < 2% sur les résidus orbitaux.

    Pour les TCN : la quantification dynamique cible les Linear.
    Pour les GRU : cible également les GRU.
    """
    model.eval()
    model.cpu()

    # Détecter les types de couches à quantifier
    target_layers = {nn.Linear}
    for m in model.modules():
        if isinstance(m, nn.GRU):
            target_layers.add(nn.GRU)
            break

    model_int8 = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec=target_layers,
        dtype=torch.qint8,
    )

    # Sauvegarder via TorchScript (compatible avec quantification)
    dummy = torch.zeros(1, window_size, input_size, dtype=torch.float32)
    try:
        scripted = torch.jit.trace(model_int8, dummy)
        torch.jit.save(scripted, int8_path.replace(".onnx", "_ts.pt"))
        logger.info(f"TorchScript INT8 → {int8_path.replace('.onnx', '_ts.pt')}")
    except Exception as e:
        logger.warning(f"TorchScript trace échoué ({e}) — sauvegarde state_dict")
        torch.save(model_int8.state_dict(), int8_path.replace(".onnx", ".pth"))

    # Comparer les tailles
    _compare_sizes(model, model_int8, window_size, input_size)

    return model_int8


def _compare_sizes(
    model_fp32: nn.Module,
    model_int8: nn.Module,
    window_size: int,
    input_size: int,
) -> None:
    """Affiche la comparaison de taille et de latence entre FP32 et INT8."""
    import time

    dummy = torch.zeros(1, window_size, input_size, dtype=torch.float32)

    # Taille en paramètres (approximation INT8 = /4)
    n_params = sum(p.numel() for p in model_fp32.parameters())
    size_fp32_kb = n_params * 4 / 1024
    size_int8_kb = n_params * 1 / 1024   # approximation

    # Latence
    N = 100
    model_fp32.eval()
    model_int8.eval()

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(N):
            model_fp32(dummy)
        lat_fp32 = (time.perf_counter() - t0) / N * 1000

        t0 = time.perf_counter()
        for _ in range(N):
            model_int8(dummy)
        lat_int8 = (time.perf_counter() - t0) / N * 1000

    logger.info(
        f"\n{'':=<50}\n"
        f"  Modèle FP32 : ~{size_fp32_kb:.1f} KB | {lat_fp32:.2f} ms/inférence\n"
        f"  Modèle INT8 : ~{size_int8_kb:.1f} KB | {lat_int8:.2f} ms/inférence\n"
        f"  Gain taille : ~{size_fp32_kb/size_int8_kb:.1f}× | "
        f"Gain vitesse : ~{lat_fp32/lat_int8:.1f}×\n"
        f"{'':=<50}"
    )


def benchmark_onnx(onnx_path: str, window_size: int = 20, input_size: int = 7) -> float:
    """
    Mesure la latence d'inférence ONNX Runtime.
    Retourne la latence médiane en ms.
    """
    try:
        import onnxruntime as ort
        import time

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, window_size, input_size), dtype=np.float32)

        # Warmup
        for _ in range(10):
            sess.run(None, {"input_sequence": dummy})

        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            sess.run(None, {"input_sequence": dummy})
            latencies.append((time.perf_counter() - t0) * 1000)

        med = float(np.median(latencies))
        p99 = float(np.percentile(latencies, 99))
        logger.info(f"ONNX Runtime — médiane : {med:.2f} ms | P99 : {p99:.2f} ms")
        return med

    except ImportError:
        logger.warning("onnxruntime non installé (pip install onnxruntime)")
        return -1.0
