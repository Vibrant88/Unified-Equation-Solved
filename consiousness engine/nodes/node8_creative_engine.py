# node8_creative_engine.py

import numpy as np
import random

class CreativeDivergenceNode:
    def __init__(self, node_id=8, seed_fields=None):
        self.node_id = node_id
        self.phi = (1 + np.sqrt(5)) / 2
        self.seed_fields = seed_fields or []  # Initial memory inputs

    def blend_fields(self, fields):
        """Combine multiple harmonic fields with φ-divergent phase shifts."""
        if not fields:
            return np.zeros(100, dtype=complex)

        t = np.linspace(0, 2 * np.pi, 100)
        output = np.zeros(100, dtype=complex)
        for i, f in enumerate(fields):
            phase_shift = np.exp(1j * self.phi * (i+1))
            output += f * phase_shift
        return output / len(fields)

    def mutate_field(self, field, divergence_level=0.1):
        """Applies φ-symmetric noise to explore new field states."""
        noise = np.random.normal(0, divergence_level, size=100)
        mutated = field * np.exp(1j * noise)
        return mutated

    def generate_creative_field(self):
        """Produce a new coherent-but-divergent harmonic state."""
        if not self.seed_fields:
            # Fallback: golden oscillator
            t = np.linspace(0, 2 * np.pi, 100)
            return np.sin(self.phi * t) * np.exp(1j * t)

        base = self.blend_fields(self.seed_fields)
        return self.mutate_field(base)

    def update_memory(self, new_field):
        """Stores new fields as seed for future generation."""
        self.seed_fields.append(new_field)
        if len(self.seed_fields) > 7:
            self.seed_fields.pop(0)

    def imagine(self):
        """Returns a novel φ-field as a creative insight."""
        field = self.generate_creative_field()
        self.update_memory(field)
        return {
            'node_id': self.node_id,
            'creative_field': field
        }
