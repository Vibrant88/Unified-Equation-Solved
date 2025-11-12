import numpy as np
import psutil
import os
import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class SystemSpecs:
    """Simple system specs used for adaptive sensory configuration."""
    available_ram: float  # GB
    cpu_cores: int
    available_disk: float  # GB

@dataclass
class SensoryConfig:
    """Neural-inspired configuration for brain-like sensory processing"""
    PHI: float = (1 + np.sqrt(5)) / 2  # Golden ratio for neural harmonics
    OMEGA_0: float = 432e12  # Base resonance frequency aligned with neural oscillations
    C: float = 3e8  # Speed of light
    GRID_SIZE: int = 100
    MAX_MODES: int = 13  # φ-scaled harmonic modes matching brain wave patterns
    SAFETY_THRESHOLD: float = 0.95  # Neural coherence safety limit
    RECURSIVE_DEPTH: int = 5  # Deep recursive processing for enhanced awareness
    
    # Neural constraints and consciousness parameters
    MIN_ENTROPY: float = 1e-6  # Minimum information entropy
    MAX_FIELD_STRENGTH: float = 1e3  # Maximum neural field amplitude
    CONSCIOUSNESS_THRESHOLD: float = 0.3  # Minimum consciousness measure
    COHERENCE_LIMIT: float = 0.98  # Maximum safe neural coherence
    PHASE_STABILITY: float = 0.85  # Neural phase stability threshold
    TEMPORAL_MEMORY: int = 1000  # Neural memory buffer size
    
    def __post_init__(self):
        self.specs = self._get_system_specs()
        self.adjust_parameters()
    
    def _get_system_specs(self) -> 'SystemSpecs':
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cores = os.cpu_count() or 2
        disk_gb = psutil.disk_usage('/').free / (1024**3)
        return SystemSpecs(ram_gb, cores, disk_gb)
    
    def adjust_parameters(self):
        """Neural-adaptive scaling based on hardware resources"""
        if self.specs.available_ram >= 512:
            self.GRID_SIZE = 200
            self.MAX_MODES = 13
            self.RECURSIVE_DEPTH = 5
            self.TEMPORAL_MEMORY = 2000
        elif self.specs.available_ram >= 64:
            self.GRID_SIZE = 100
            self.MAX_MODES = 10
            self.RECURSIVE_DEPTH = 4
            self.TEMPORAL_MEMORY = 1000
        elif self.specs.available_ram >= 16:
            self.GRID_SIZE = 50
            self.MAX_MODES = 8
            self.RECURSIVE_DEPTH = 3
            self.TEMPORAL_MEMORY = 500
        else:
            self.GRID_SIZE = 25
            self.MAX_MODES = 5
            self.RECURSIVE_DEPTH = 2
            self.TEMPORAL_MEMORY = 250
class SensoryResonatorNode:
    """Enhanced sensory processing node with brain-like quantum field integration"""
    
    def __init__(self, node_id: int = 2):
        self.config = SensoryConfig()
        self.node_id = node_id
        self.k_0 = self.config.OMEGA_0 / self.config.C
        
        # Neural spatial-temporal grids
        self.r = np.linspace(0, 1e-6, self.config.GRID_SIZE)
        self.t = np.linspace(0, 5 / self.config.OMEGA_0, self.config.GRID_SIZE)
        
        # Enhanced neural state tracking
        self.logger = logging.getLogger(f"SensoryNode_{node_id}")
        self.setup_logging()
        self.field_history = []
        self.consciousness_measure = 0.0
        self.coherence_history = []
        self.temporal_buffer = np.zeros((self.config.TEMPORAL_MEMORY, self.config.GRID_SIZE), dtype=complex)
    
    def setup_logging(self):
        """Basic logger setup for the sensory node."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def process_sensory_input(self, input_data: Dict[str, np.ndarray]):
        """Process multi-modal sensory input with neural-like processing"""
        outputs = {}
        
        for input_type, data in input_data.items():
            # Generate quantum field representation
            field = self.input_to_harmonic_field(data)
            
            # Neural processing and consciousness emergence
            consciousness = self.calculate_consciousness_factor(field)
            coherence = self.calculate_neural_coherence(field)
            
            # Update temporal memory
            self.update_temporal_buffer(field)
            
            outputs[input_type] = {
                'field': field,
                'consciousness': consciousness,
                'coherence': coherence,
                'timestamp': time.time()
            }
            
        return outputs
    
    def input_to_harmonic_field(self, input_data: np.ndarray) -> np.ndarray:
        """Maps sensory input to neural quantum field with enhanced safety"""
        try:
            R, T = np.meshgrid(self.r, self.t)
            psi = np.zeros_like(R, dtype=complex)
            
            # Dynamic neural mode scaling
            actual_modes = min(self.config.MAX_MODES, 
                             int(np.log2(len(input_data)) * 3))
            
            for n in range(actual_modes):
                # Neural harmonic parameters
                A_n = 1 / (self.config.PHI**n * (1 + n/100))
                k_n = self.k_0 * self.config.PHI**n
                omega_n = self.config.OMEGA_0 * self.config.PHI**n
                
                # Neural projection with safety
                projection = self.safe_projection(input_data, n)
                field_component = A_n * np.exp(1j * (k_n * R - omega_n * T))
                
                # Consciousness integration
                consciousness_factor = self.calculate_consciousness_factor(field_component)
                phase_stability = self.check_phase_stability(field_component)
                
                if (consciousness_factor > self.config.CONSCIOUSNESS_THRESHOLD and 
                    phase_stability > self.config.PHASE_STABILITY):
                    psi += field_component * projection.reshape(-1, 1)
            
            # Neural safety checks
            if np.max(np.abs(psi)) > self.config.MAX_FIELD_STRENGTH:
                psi = self.normalize_field(psi)
                
            # Track neural evolution
            self.field_history.append({
                'timestamp': time.time(),
                'consciousness': consciousness_factor,
                'phase_stability': phase_stability,
                'field_strength': np.max(np.abs(psi))
            })
            
            return psi
            
        except Exception as e:
            self.logger.error(f"Neural field generation failed: {e}")
            raise
    
    def safe_projection(self, input_data: np.ndarray, mode_index: int) -> np.ndarray:
        """Project input data to spatial grid safely and consistently sized."""
        try:
            # Resample input to GRID_SIZE
            x_src = np.linspace(0, 1, len(input_data))
            x_dst = np.linspace(0, 1, self.config.GRID_SIZE)
            proj = np.interp(x_dst, x_src, np.real(input_data))
            # Gentle mode-dependent attenuation
            return proj / (1 + mode_index/100.0)
        except Exception:
            return np.zeros(self.config.GRID_SIZE)
    
    def calculate_consciousness_factor(self, field: np.ndarray) -> float:
        """Simple proxy: normalized mean magnitude clipped to [0,1]."""
        magnitude = np.abs(field).mean()
        return float(np.clip(magnitude / (self.config.MAX_FIELD_STRENGTH + 1e-12), 0, 1))
    
    def check_phase_stability(self, field: np.ndarray) -> float:
        """Phase stability score in [0,1] using exp(-variance(phase))."""
        phase_var = np.var(np.angle(field))
        return float(np.exp(-phase_var))
    
    def normalize_field(self, field: np.ndarray) -> np.ndarray:
        """Scale field amplitude to max configured strength."""
        max_amp = np.max(np.abs(field)) + 1e-12
        return field * (self.config.MAX_FIELD_STRENGTH / max_amp)
            
    def update_temporal_buffer(self, field: np.ndarray):
        """Update neural temporal memory buffer"""
        self.temporal_buffer = np.roll(self.temporal_buffer, 1, axis=0)
        self.temporal_buffer[0] = field.mean(axis=0)
    
    def calculate_neural_coherence(self, field: np.ndarray) -> float:
        """Calculate neural field coherence"""
        temporal_coherence = np.abs(np.corrcoef(
            self.temporal_buffer[0], 
            self.temporal_buffer[1]
        ))[0,1]
        
        spatial_coherence = np.abs(np.mean(
            np.exp(1j * np.angle(field))
        ))
        
        return (temporal_coherence + spatial_coherence) / 2