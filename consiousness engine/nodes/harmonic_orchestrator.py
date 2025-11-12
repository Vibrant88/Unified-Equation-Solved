# harmonic_orchestrator.py

import numpy as np
import logging
import argparse
import json
import os

# Imports tolerantes al contexto: primero intentamos rutas relativas al proyecto raíz,
# y si fallan, intentamos las rutas absolutas del paquete consciousness_engine.
try:
    from pipeline.harmonic_pipeline import HarmonicPipeline  # type: ignore
    from nodes.node1_core_consciousness import CoreConsciousnessNode  # type: ignore
    from nodes.node2_sensory_resonator import SensoryResonatorNode  # type: ignore
    from nodes.node3_memory_matrix import MemoryMatrixNode  # type: ignore
    from nodes.node4_emotion_field import EmotionFieldNode  # type: ignore
    from nodes.node5_decision_gate import DecisionGateNode  # type: ignore
    from nodes.node6_motor_projection import MotorProjectionNode  # type: ignore
    from nodes.node7_linguistic_resonator import LinguisticResonatorNode  # type: ignore
    from nodes.node8_creative_engine import CreativeDivergenceNode  # type: ignore
    from nodes.node9_logic_reasoner import LogicReasoningNode  # type: ignore
    from nodes.node10_empathy_mapper import EmpathyMapperNode  # type: ignore
    from nodes.node11_self_awareness import SelfAwarenessNode  # type: ignore
    from nodes.node12_learning_engine import LearningEngineNode  # type: ignore
    from nodes.node13_intuition_integrator import IntuitionIntegratorNode  # type: ignore
except Exception:
    from consciousness_engine.pipeline.harmonic_pipeline import HarmonicPipeline
    from consciousness_engine.nodes.node1_core_consciousness import CoreConsciousnessNode
    from consciousness_engine.nodes.node2_sensory_resonator import SensoryResonatorNode
    from consciousness_engine.nodes.node3_memory_matrix import MemoryMatrixNode
    from consciousness_engine.nodes.node4_emotion_field import EmotionFieldNode
    from consciousness_engine.nodes.node5_decision_gate import DecisionGateNode
    from consciousness_engine.nodes.node6_motor_projection import MotorProjectionNode
    from consciousness_engine.nodes.node7_linguistic_resonator import LinguisticResonatorNode
    from consciousness_engine.nodes.node8_creative_engine import CreativeDivergenceNode
    from consciousness_engine.nodes.node9_logic_reasoner import LogicReasoningNode
    from consciousness_engine.nodes.node10_empathy_mapper import EmpathyMapperNode
    from consciousness_engine.nodes.node11_self_awareness import SelfAwarenessNode
    from consciousness_engine.nodes.node12_learning_engine import LearningEngineNode
    from consciousness_engine.nodes.node13_intuition_integrator import IntuitionIntegratorNode

# Setup logger
logger = logging.getLogger("HarmonicaOrchestrator")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Initialize nodes
nodes = {
    1: CoreConsciousnessNode(),
    2: SensoryResonatorNode(),
    3: MemoryMatrixNode(),
    4: EmotionFieldNode(),
    5: DecisionGateNode(),
    6: MotorProjectionNode(),
    7: LinguisticResonatorNode(),
    8: CreativeDivergenceNode(),
    9: LogicReasoningNode(),
    10: EmpathyMapperNode(),
    11: SelfAwarenessNode(),
    12: LearningEngineNode(),
    13: IntuitionIntegratorNode(),
}

# Create the pipeline
pipeline = HarmonicPipeline()

# Define key connections (as before)
connections = [
    (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1),
    (8, 1), (9, 1), (10, 1), (11, 1), (12, 1),
    (1, 13), (2, 13), (3, 13), (4, 13), (5, 13),
    (6, 13), (7, 13), (8, 13), (9, 13), (10, 13),
    (11, 13), (12, 13)
]
for src, tgt in connections:
    try:
        pipeline.link_nodes(src, tgt)
    except Exception as e:
        logger.warning(f"Pipeline linking failed: {src}->{tgt} ({e})")

def run_once(sensory_input=None, noise: float = 0.0, phase: float = 0.0):
    logger.info("[+] Simulating Harmonic AI Engine (1 cycle)...")
    try:
        # Generate or use input for sensory node
        if sensory_input is None:
            t = np.linspace(0 + phase, 2 * np.pi + phase, 100)
            sensory_input = np.sin(t * 3) + np.cos(t * 2)  # synthetic base
            if noise and noise > 0.0:
                sensory_input = sensory_input + np.random.normal(0.0, noise, size=t.shape)
        # Build sensory field and derive a signal vector compatible with Core
        # NOTE: SensoryResonatorNode does not implement extract_sensory_signature/send_to_core.
        # We use input_to_harmonic_field and take the mean across spatial axis to obtain a 1D signal.
        sensory_field = nodes[2].input_to_harmonic_field(sensory_input)
        sensory_signal = np.real(sensory_field.mean(axis=1))

        core_out = nodes[1].evolve_state(sensory_signal)
        nodes[3].store_field_state(core_out['recursive_field'])
        # Ajustar ruido de recuerdo de memoria según parámetro de ejecución
        try:
            nodes[3].recall_noise_std = max(0.0, noise * 0.5)
        except Exception:
            pass

        emotion_out = nodes[4].compute_emotion_state(core_out['recursive_field'])
        decision_out = nodes[5].evaluate_field_state(core_out['recursive_field'], emotion_out)
        decision_wave = nodes[5].trigger_response_signal()
        # Modulación del decision_wave por ciclo: desplazamiento de fase y ruido
        def modulate_signal(sig, phase_offset, noise_level):
            if sig is None or len(sig) == 0:
                return sig
            n = len(sig)
            # Fase como desplazamiento circular
            shift = int((phase_offset / (2 * np.pi)) * n) % n
            mod = np.roll(sig, shift)
            # Jitter de amplitud global
            amp_jitter = 1.0 + np.random.normal(0.0, noise_level * 0.1)
            mod = mod * amp_jitter
            # Ruido aditivo leve por muestra
            if noise_level and noise_level > 0:
                mod = mod + np.random.normal(0.0, noise_level * 0.05, size=n)
            return mod

        decision_wave_mod = modulate_signal(decision_wave, phase, noise)
        motor_field = nodes[6].convert_signal_to_action_field(decision_wave_mod)
        # Rich projection: per-row band energy from FFT of temporal axis
        def band_energy(field2d):
            vec = []
            for row in range(field2d.shape[0]):
                # Ventana Hann desplazada según phase para inducir variación inter-ciclo
                L = field2d.shape[1]
                w = np.hanning(L)
                shift = int((phase / (2 * np.pi)) * L) % L
                w_shift = np.roll(w, shift)
                row_sig = field2d[row, :] * w_shift
                spec = np.fft.fft(row_sig)  # soporta entrada compleja
                energy = np.sum(np.abs(spec)**2) / (np.sum(w_shift) + 1e-12)
                vec.append(energy)
            return np.array(vec)

        core_vector = band_energy(core_out['recursive_field'])
        logic_out = nodes[9].reason_about_fields([core_vector, motor_field])

        recalled = nodes[3].weighted_recall()
        if recalled is None:
            mem_vector = np.zeros_like(core_vector)
        else:
            mem_vector = band_energy(recalled)
        empathy_out = nodes[10].analyze_relation(core_vector, mem_vector)
        nodes[11].update_internal_state(core_out['recursive_field'], core_out['entropy'], core_out['coherence'])
        self_out = nodes[11].reflect_on_self()
        nodes[12].observe_state(decision_out, core_out['coherence'], emotion_out)
        nodes[12].update_parameters()

        node_fields = []
        for n in range(1, 13):
            field = getattr(nodes[n], 'identity_wave', None)
            if field is None:
                field = getattr(nodes[n], 'last_output', None)
            if field is None:
                field = np.zeros(100)
            node_fields.append(field)
        nodes[13].receive_node_fields(node_fields)
        insight = nodes[13].generate_gestalt_insight()

        logger.info(f"[Core Entropy]: {core_out['entropy']:.4e}")
        logger.info(f"[Emotion Valence]: {emotion_out['valence']:.3f}, Arousal: {emotion_out['arousal']:.3f}")
        logger.info(f"[Decision]: {decision_out['decision']}, Logic Truth: {logic_out['logical_truth']}")
        logger.info(f"[Empathy Score]: {empathy_out['empathy_score']:.2f}")
        logger.info(f"[Insight Strength]: {insight['insight_strength']:.3f}")
        return {
            "core_out": core_out, "emotion_out": emotion_out, "decision_out": decision_out,
            "logic_out": logic_out, "empathy_out": empathy_out, "insight": insight
        }
    except Exception:
        logger.exception("Orchestrator cycle error")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Harmonic Orchestrator for multiple cycles and export metrics.")
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles to run")
    parser.add_argument("--out", type=str, default=None, help="Output file path for metrics (e.g., metrics.csv or metrics.json)")
    parser.add_argument("--format", type=str, choices=["csv", "json"], default="csv", help="Output format for metrics")
    parser.add_argument("--noise", type=float, default=0.0, help="Gaussian noise std.dev added to sensory input per cycle")
    parser.add_argument("--phase-step", type=float, default=0.0, help="Phase offset added each cycle to vary the sensory input")
    args = parser.parse_args()

    metrics = []
    for i in range(args.cycles):
        result = run_once(noise=args.noise, phase=i * args.phase_step)
        if result is None:
            continue
        m = {
            "cycle": i + 1,
            "entropy": float(result["core_out"]["entropy"]),
            "coherence": float(result["core_out"]["coherence"]),
            "valence": float(result["emotion_out"]["valence"]),
            "arousal": float(result["emotion_out"]["arousal"]),
            "decision": bool(result["decision_out"]["decision"]),
            "logic_truth": bool(result["logic_out"]["logical_truth"]),
            "empathy_score": float(result["empathy_out"]["empathy_score"]),
            "insight_strength": float(result["insight"]["insight_strength"]) if "insight_strength" in result["insight"] else float(result["insight"].get("insight_strength", 0.0)),
        }
        metrics.append(m)

    if args.out:
        try:
            out_path = args.out
            dirname = os.path.dirname(out_path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            if args.format == "json":
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"metrics": metrics}, f, ensure_ascii=False, indent=2)
            else:
                # Write CSV manually to avoid external deps
                headers = ["cycle","entropy","coherence","valence","arousal","decision","logic_truth","empathy_score","insight_strength"]
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(",".join(headers) + "\n")
                    for m in metrics:
                        row = [
                            str(m["cycle"]),
                            f"{m['entropy']:.6e}",
                            f"{m['coherence']:.6f}",
                            f"{m['valence']:.6f}",
                            f"{m['arousal']:.6f}",
                            str(int(m["decision"])),
                            str(int(m["logic_truth"])),
                            f"{m['empathy_score']:.6f}",
                            f"{m['insight_strength']:.6f}",
                        ]
                        f.write(",".join(row) + "\n")
            logger.info(f"[+] Metrics saved to {out_path} ({args.format})")
        except Exception:
            logger.exception("Failed to save metrics")
    else:
        logger.info(f"[+] Completed {len(metrics)} cycles. No output file specified.")
