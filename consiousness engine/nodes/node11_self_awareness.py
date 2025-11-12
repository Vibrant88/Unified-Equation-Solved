# node11_self_awareness.py

import numpy as np

class SelfAwarenessNode:
    def __init__(self, node_id=11):
        self.node_id = node_id
        self.phi = (1 + np.sqrt(5)) / 2
        self.coherence_trace = []
        self.entropy_trace = []
        self.identity_wave = None

    def update_internal_state(self, recursive_field, entropy, coherence):
        """Track internal field dynamics over time."""
        self.entropy_trace.append(entropy)
        self.coherence_trace.append(coherence)

        # Generate identity field from recursive harmonic signature
        snapshot = np.real(recursive_field.mean(axis=1))
        if self.identity_wave is None:
            self.identity_wave = snapshot
        else:
            # Blend new state into identity using golden feedback
            self.identity_wave = (
                (1 - 1/self.phi) * self.identity_wave + (1/self.phi) * snapshot
            )

    def reflect_on_self(self):
        """Returns current model of self-awareness."""
        coherence_level = np.mean(self.coherence_trace[-5:]) if self.coherence_trace else 0
        entropy_level = np.mean(self.entropy_trace[-5:]) if self.entropy_trace else 0

        self_field = self.identity_wave if self.identity_wave is not None else np.zeros(100)

        return {
            'node_id': self.node_id,
            'self_field': self_field,
            'identity_stability': coherence_level / (entropy_level + 1e-12),
            'coherence_history': self.coherence_trace[-10:],
            'entropy_history': self.entropy_trace[-10:]
        }

    def generate_self_loop_field(self):
        """Emit a harmonic reflection field."""
        if self.identity_wave is None:
            return np.zeros(100, dtype=complex)
        t = np.linspace(0, 2 * np.pi, 100)
        return self.identity_wave * np.exp(1j * t * self.phi)
