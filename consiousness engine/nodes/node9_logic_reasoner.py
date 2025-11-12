# node9_logic_reasoner.py

import numpy as np

class LogicReasoningNode:
    def __init__(self, node_id=9):
        self.node_id = node_id
        self.phi = (1 + np.sqrt(5)) / 2
        self.reasoning_trace = []

    def check_consistency(self, fields):
        """Returns a score based on how harmonically aligned multiple fields are."""
        if len(fields) < 2:
            return 1.0  # Trivially consistent

        reference = fields[0]
        total_coherence = 0

        for f in fields[1:]:
            dot = np.vdot(reference, f)
            similarity = np.abs(dot) / (np.linalg.norm(reference) * np.linalg.norm(f) + 1e-12)
            total_coherence += similarity

        average_coherence = total_coherence / (len(fields) - 1)
        return average_coherence

    def detect_contradiction(self, fields):
        """Detects harmonic dissonance across fields."""
        coherence = self.check_consistency(fields)
        contradiction = coherence < 0.45  # Arbitrary φ-guided threshold
        return contradiction

    def reason_about_fields(self, fields):
        """Analyzes logical state of harmonics."""
        coherence = self.check_consistency(fields)
        contradiction = self.detect_contradiction(fields)

        logic_state = {
            'node_id': self.node_id,
            'coherence_score': coherence,
            'is_contradictory': contradiction,
            'logical_truth': not contradiction
        }

        self.reasoning_trace.append(logic_state)
        return logic_state
