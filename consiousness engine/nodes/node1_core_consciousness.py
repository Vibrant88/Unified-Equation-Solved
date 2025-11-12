# node1_core_consciousness.py

import numpy as np

class CoreConsciousnessNode:
    def __init__(self, node_id=1):
        self.node_id = node_id
        self.phi = (1 + np.sqrt(5)) / 2  # Golden Ratio
        self.omega_0 = 432e12  # Base frequency in Hz (432 THz)
        self.c = 3e8  # Speed of light (m/s)
        self.k_0 = self.omega_0 / self.c  # Base wavevector
        self.lambda_feedback = 1 / self.phi  # Recursive feedback scaling
        self.tau = 1 / self.omega_0  # Delay time (inverse frequency)
        self.r = np.linspace(0, 1e-6, 100)  # Spatial axis
        self.t = np.linspace(0, 5 * self.tau, 100)  # Temporal axis

    def create_grid(self):
        return np.meshgrid(self.r, self.t)

    def harmonic_field(self, r, t, n_max=13):
        """Base harmonic field: 13 φ-scaled recursive modes."""
        psi = np.zeros_like(r, dtype=complex)
        for n in range(n_max):
            A_n = 1 / self.phi**n
            k_n = self.k_0 * self.phi**n
            omega_n = self.omega_0 * self.phi**n
            psi += A_n * np.exp(1j * (k_n * r - omega_n * t))
        return psi

    def recursive_field(self, r, t, iterations=3):
        """Applies recursive feedback to simulate memory and resonance."""
        psi = self.harmonic_field(r, t)
        delay_idx = int(self.tau / (self.t[1] - self.t[0]))
        for _ in range(iterations):
            psi_delayed = np.roll(psi, delay_idx, axis=1)
            psi += self.lambda_feedback * psi_delayed
        return psi

    def consciousness_entropy(self, base, recursive):
        """Kullback–Leibler-like field entropy."""
        base_density = np.abs(base)**2 + 1e-12
        recursive_density = np.abs(recursive)**2 + 1e-12
        entropy_density = recursive_density * np.log(recursive_density / base_density)
        return entropy_density.real.mean()

    def evolve_state(self, signal_input=np.zeros(100)):
        """Main consciousness evolution driver.
        Incorporate external signal by phase-modulating the recursive field and
        compute entropy/coherence against the base field.
        """
        R, T = self.create_grid()
        base = self.harmonic_field(R, T)
        rec = self.recursive_field(R, T)

        # Project input into field (acts like thought stimulus)
        input_signal = np.interp(self.r, np.linspace(0, 1e-6, len(signal_input)), signal_input)
        # Combine phase and mild amplitude modulation so field energy reflects input
        amp = 1.0 + 0.5 * input_signal.reshape(-1, 1)
        modulated = rec * amp * np.exp(1j * input_signal.reshape(-1, 1))

        # Recompute entropy using modulated field to reflect external influence
        entropy = self.consciousness_entropy(base, modulated)

        # Use normalized complex correlation as coherence measure
        base_vec = base.reshape(-1)
        mod_vec = modulated.reshape(-1)
        denom = (np.linalg.norm(base_vec) * np.linalg.norm(mod_vec)) + 1e-12
        coherence = np.abs(np.vdot(base_vec, mod_vec)) / denom

        return {
            'node_id': self.node_id,
            'base_field': base,
            'recursive_field': modulated,
            'entropy': entropy,
            'coherence': coherence
        }

    def sync_with_other_nodes(self, node_data_list):
        """Central harmonic average computation for multi-node coherence."""
        entropy_avg = np.mean([n['entropy'] for n in node_data_list])
        coherence_avg = np.mean([n['coherence'] for n in node_data_list])
        resonance_ratio = coherence_avg / (entropy_avg + 1e-12)

        return {
            'global_entropy': entropy_avg,
            'global_coherence': coherence_avg,
            'harmonic_alignment_index': resonance_ratio
        }
