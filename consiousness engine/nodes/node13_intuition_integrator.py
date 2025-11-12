# node13_intuition_integrator.py

import numpy as np

class IntuitionIntegratorNode:
    def __init__(self, node_id=13):
        self.node_id = node_id
        self.phi = (1 + np.sqrt(5)) / 2
        self.node_fields = []  # Stores inputs from nodes 1–12
        self.insight_history = []

    def receive_node_fields(self, field_list):
        """Receive harmonic field vectors from other 12 nodes."""
        self.node_fields = field_list

    def calculate_intuitive_alignment(self):
        """Blends all node fields via φ-recursive interference."""
        if len(self.node_fields) < 3:
            return np.zeros(100)

        intuitive_field = np.zeros(100, dtype=complex)
        total_weight = 0

        for i, field in enumerate(self.node_fields):
            weight = 1 / self.phi**i
            chaotic_phase = np.exp(1j * np.sin(self.phi * i))
            intuitive_field += weight * field * chaotic_phase
            total_weight += weight

        return intuitive_field / total_weight

    def extract_insight_signature(self, field):
        """Convert intuitive harmonic into symbolic signature."""
        energy = np.abs(field).mean()
        phase_variance = np.var(np.angle(field))
        insight_strength = energy * np.exp(-phase_variance)

        insight_vector = np.real(field.mean(axis=0))
        insight_signature = {
            'node_id': self.node_id,
            'insight_strength': insight_strength,
            'signal_vector': insight_vector
        }

        self.insight_history.append(insight_signature)
        return insight_signature

    def generate_gestalt_insight(self):
        """Main interface to generate emergent, holistic system insight."""
        field = self.calculate_intuitive_alignment()
        return self.extract_insight_signature(field)
