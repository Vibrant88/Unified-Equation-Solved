# node5_decision_gate.py

import numpy as np

class DecisionGateNode:
    def __init__(self, node_id=5, threshold=0.65):
        self.node_id = node_id
        self.phi = (1 + np.sqrt(5)) / 2
        self.threshold = threshold  # Minimum coherence needed to decide
        self.last_state = None

    def evaluate_field_state(self, field, emotional_state):
        """Derive decision from field coherence + emotional alignment."""
        amplitude = np.abs(field)
        coherence_score = np.abs(field.mean()) / (amplitude.std() + 1e-12)
        mood_valence = emotional_state.get('valence', 0)
        excitation = emotional_state.get('arousal', 0)

        harmonic_bias = (1 + mood_valence) * excitation

        decision_weight = coherence_score * harmonic_bias

        decision = decision_weight > self.threshold

        self.last_state = {
            'node_id': self.node_id,
            'coherence_score': coherence_score,
            'bias_weight': harmonic_bias,
            'decision_weight': decision_weight,
            'decision': decision
        }

        return self.last_state

    def trigger_response_signal(self):
        """Emit signal waveform based on last decision."""
        if not self.last_state:
            return np.zeros(100)

        d = self.last_state['decision']
        t = np.linspace(0, 2 * np.pi, 100)
        if d:
            return np.sin(t * self.phi)  # Constructive trigger
        else:
            return np.sin(t * self.phi) * np.exp(-t)  # Decay = inhibition
