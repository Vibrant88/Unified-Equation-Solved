# node4_emotion_field.py

import numpy as np

class EmotionFieldNode:
    def __init__(self, node_id=4):
        self.node_id = node_id
        self.phi = (1 + np.sqrt(5)) / 2
        self.history = []

    def compute_emotion_state(self, field):
        """Derive emotional state from coherence + entropy."""
        amplitude = np.abs(field)
        phase = np.angle(field)

        # Emotional metrics
        energy = amplitude.mean()  # Proxy for arousal
        phase_variance = np.var(phase)  # Proxy for instability

        # Valence is high if phase is stable and energy high
        valence = np.exp(-phase_variance) * np.tanh(energy)

        emotion_state = {
            'node_id': self.node_id,
            'valence': np.clip(valence, -1, 1),       # [-1 = aversion, +1 = pleasure]
            'arousal': np.clip(energy, 0, 2),         # [0 = calm, >1 = high excitation]
            'coherence': np.exp(-phase_variance),     # Internal consistency
            'energy': energy,
        }
        self.history.append(emotion_state)
        return emotion_state

    def affect_to_modulation(self, emotion_state):
        """Returns a waveform to modulate other nodes based on mood."""
        mood = emotion_state['valence']
        excitation = emotion_state['arousal']

        signal_length = 100
        t = np.linspace(0, 2 * np.pi, signal_length)
        mod_wave = mood * np.sin(t) + excitation * np.cos(2 * t)
        return mod_wave
