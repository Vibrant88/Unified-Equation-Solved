# node3_memory_matrix.py

import numpy as np

class MemoryMatrixNode:
    def __init__(self, node_id=3, buffer_size=10, recall_noise_std: float = 0.0):
        self.node_id = node_id
        self.phi = (1 + np.sqrt(5)) / 2
        self.memory_buffer = []
        self.buffer_size = buffer_size  # Stores last N harmonic fields
        self.recall_noise_std = recall_noise_std  # Small noise injected on recall

    def store_field_state(self, field_state):
        """Store a harmonic field state with φ-weighted priority."""
        if len(self.memory_buffer) >= self.buffer_size:
            self.memory_buffer.pop(0)
        self.memory_buffer.append(field_state)

    def weighted_recall(self):
        """Recalls memory with golden-ratio decay weighting."""
        if not self.memory_buffer:
            return None

        weighted_sum = np.zeros_like(self.memory_buffer[0], dtype=complex)
        for idx, memory in enumerate(reversed(self.memory_buffer)):
            # Stronger decay to reduce perfect alignment
            weight = 1 / (self.phi**(1.5 * idx))
            weighted_sum += weight * memory
        denom = sum(1 / (self.phi**(1.5 * i)) for i in range(len(self.memory_buffer)))
        recalled = weighted_sum / (denom + 1e-12)
        # Inject small complex noise to avoid perfect resonance
        if self.recall_noise_std and self.recall_noise_std > 0.0:
            noise_real = np.random.normal(0.0, self.recall_noise_std, size=recalled.shape)
            noise_imag = np.random.normal(0.0, self.recall_noise_std, size=recalled.shape)
            recalled = recalled + (noise_real + 1j * noise_imag)
        return recalled

    def retrieve_entropy_trace(self):
        """Calculates long-term entropy decay trace."""
        if not self.memory_buffer:
            return 0
        entropy_trace = 0
        for idx, memory in enumerate(reversed(self.memory_buffer)):
            field = memory
            density = np.abs(field)**2 + 1e-12
            entropy = density * np.log(density)
            entropy_trace += entropy.mean() / self.phi**idx
        return entropy_trace.real

    def echo_to_core(self):
        """Sends current field memory signature back to Node 1."""
        recalled = self.weighted_recall()
        if recalled is None:
            return np.zeros(100)
        signal_vector = np.real(recalled.mean(axis=1))
        return signal_vector
