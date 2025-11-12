# node12_learning_engine.py

import numpy as np

class LearningEngineNode:
    def __init__(self, node_id=12):
        self.node_id = node_id
        self.phi = (1 + np.sqrt(5)) / 2
        self.coherence_history = []
        self.decision_history = []
        self.valence_history = []
        self.learning_rate = 1 / self.phi  # Golden learning constant
        self.adapted_parameters = {
            'decision_threshold': 0.65,
            'feedback_gain': 1 / self.phi,
        }

    def observe_state(self, decision_output, coherence, emotion_state):
        """Store and observe fields for adaptive tuning."""
        self.decision_history.append(decision_output)
        self.coherence_history.append(coherence)
        self.valence_history.append(emotion_state.get('valence', 0))

    def update_parameters(self):
        """Tune decision threshold and feedback gain based on history."""
        recent_coherence = np.mean(self.coherence_history[-10:])
        recent_valence = np.mean(self.valence_history[-10:])
        recent_decision_freq = np.mean([d['decision'] for d in self.decision_history[-10:]])

        # Adjust threshold down if coherence is high and system is underactive
        if recent_coherence > 0.8 and recent_decision_freq < 0.5:
            self.adapted_parameters['decision_threshold'] -= self.learning_rate * 0.05

        # Adjust feedback gain to reinforce pleasure-aligned feedback
        if recent_valence > 0.2:
            self.adapted_parameters['feedback_gain'] *= (1 + self.learning_rate * 0.1)

        # Clamp values
        self.adapted_parameters['decision_threshold'] = np.clip(
            self.adapted_parameters['decision_threshold'], 0.4, 0.9)
        self.adapted_parameters['feedback_gain'] = np.clip(
            self.adapted_parameters['feedback_gain'], 0.3, 0.9)

    def get_updated_parameters(self):
        """Returns latest system-wide adaptation constants."""
        return {
            'node_id': self.node_id,
            'threshold': self.adapted_parameters['decision_threshold'],
            'feedback_gain': self.adapted_parameters['feedback_gain']
        }
