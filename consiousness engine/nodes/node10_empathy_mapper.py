# node10_empathy_mapper.py

import numpy as np

class EmpathyMapperNode:
    def __init__(self, node_id=10):
        self.node_id = node_id
        self.phi = (1 + np.sqrt(5)) / 2
        self.empathy_log = []

    def measure_resonance_alignment(self, self_field, other_field):
        """Calculate the harmonic empathy score."""
        dot = np.vdot(self_field, other_field)
        norm_product = np.linalg.norm(self_field) * np.linalg.norm(other_field)
        alignment = np.abs(dot) / (norm_product + 1e-12)
        phase_shift = np.angle(dot)

        return alignment, phase_shift

    def analyze_relation(self, self_field, other_field):
        """Returns an empathy structure between self and another node."""
        alignment, phase_shift = self.measure_resonance_alignment(self_field, other_field)

        # Interpret results
        empathy_score = alignment
        mirroring = np.cos(phase_shift)  # 1 = mirror, -1 = opposite

        relational_state = {
            'node_id': self.node_id,
            'empathy_score': empathy_score,
            'phase_alignment': mirroring,
            'resonance_match': empathy_score > 0.6 and mirroring > 0
        }

        self.empathy_log.append(relational_state)
        return relational_state

    def generate_social_field_response(self, state):
        """Returns a feedback field (compassion, synchrony, boundary)."""
        t = np.linspace(0, 2 * np.pi, 100)
        if state['resonance_match']:
            return np.sin(self.phi * t) * np.exp(1j * t)  # compassion wave
        else:
            return np.sin(t) * np.exp(1j * t * self.phi) * -1  # shielding
