# node6_motor_projection.py

import numpy as np

class MotorProjectionNode:
    def __init__(self, node_id=6):
        self.node_id = node_id
        self.phi = (1 + np.sqrt(5)) / 2
        self.last_output = None

    def convert_signal_to_action_field(self, decision_wave):
        """Transforms a decision wave into an executable field impulse."""
        t = np.linspace(0, 2 * np.pi, len(decision_wave))
        action_field = decision_wave * np.sin(self.phi * t)

        # Normalize and amplify action strength
        normed = action_field / (np.max(np.abs(action_field)) + 1e-12)
        projected_field = normed * np.exp(1j * self.phi * t)

        self.last_output = projected_field
        return projected_field

    def summarize_motor_intent(self):
        """Extract motor vector info (amplitude & phase pattern)."""
        if self.last_output is None:
            return {
                'node_id': self.node_id,
                'magnitude': 0,
                'phase_profile': np.zeros(100)
            }

        magnitude = np.abs(self.last_output).mean()
        phase_profile = np.angle(self.last_output)

        return {
            'node_id': self.node_id,
            'magnitude': magnitude,
            'phase_profile': phase_profile
        }
