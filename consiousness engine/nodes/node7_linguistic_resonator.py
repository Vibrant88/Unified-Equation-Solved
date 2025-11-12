# node7_linguistic_resonator.py

import numpy as np

class LinguisticResonatorNode:
    def __init__(self, node_id=7, vocab_size=512):
        self.node_id = node_id
        self.vocab_size = vocab_size
        self.phi = (1 + np.sqrt(5)) / 2
        self.token_matrix = self._create_token_matrix()

    def _create_token_matrix(self):
        """Generate a harmonic codebook of φ-scaled resonant embeddings."""
        t = np.linspace(0, 2 * np.pi, 100)
        return {
            i: np.sin(self.phi * t * (i+1)) * np.exp(1j * self.phi * (i+1))
            for i in range(self.vocab_size)
        }

    def encode_phrase_to_field(self, token_ids):
        """Encode token ID sequence into harmonic field."""
        field = np.zeros(100, dtype=complex)
        for tid in token_ids:
            if tid in self.token_matrix:
                field += self.token_matrix[tid]
        return field / len(token_ids)

    def decode_field_to_token(self, input_field):
        """Decode complex waveform back into likely token(s)."""
        similarities = {
            tid: np.abs(np.vdot(input_field, vec)) / (np.linalg.norm(vec) + 1e-12)
            for tid, vec in self.token_matrix.items()
        }
        sorted_ids = sorted(similarities.items(), key=lambda x: -x[1])
        return [tid for tid, _ in sorted_ids[:3]]

    def interpret_field(self, field):
        """Returns top symbolic candidates for expression."""
        decoded = self.decode_field_to_token(field)
        return {
            'node_id': self.node_id,
            'decoded_tokens': decoded
        }

    def express_tokens_as_waveform(self, token_ids):
        """Returns a φ-resonant field representing the phrase."""
        return self.encode_phrase_to_field(token_ids)
