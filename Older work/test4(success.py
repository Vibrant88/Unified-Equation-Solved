#!/usr/bin/env python3
"""
STABLE UNIFIED FIELD VALIDATOR
Fixed numerical stability and calculation errors
"""

import numpy as np
import scipy.constants as const
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
from enum import Enum
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StableFieldValidator")

class ValidationStatus(Enum):
    SUCCESS = "SUCCESS"
    WARNING = "WARNING" 
    FAILURE = "FAILURE"
    ERROR = "ERROR"

@dataclass
class StableFieldConfig:
    """Stable configuration with proper numerical bounds"""
    
    # Core constants with safe numerical ranges
    c: float = const.c
    h: float = const.h
    hbar: float = const.hbar
    G: float = const.G
    e: float = const.e
    m_e: float = const.m_e
    m_p: float = const.m_p
    alpha: float = const.alpha
    
    # Safe frequency ranges (Hz)
    safe_frequencies: Dict[str, float] = field(default_factory=lambda: {
        'solfeggio_417': 417e12,
        'geometric': 3.555e12,
        'molecular': 1.000e15,
        'base': 3.172e17,
        'multi_scale': 1.006e20
    })
    
    # Safe spatial scales (meters)
    safe_scales: Dict[str, float] = field(default_factory=lambda: {
        'planck': 1.616e-35,
        'nuclear': 1e-15,
        'atomic': 1e-10,
        'cellular': 1e-6,
        'human': 1e0,
        'planetary': 1e7
    })

class StableFieldValidator:
    """
    Stable validator with numerical safety and bounds checking
    """
    
    def __init__(self, config: StableFieldConfig):
        self.config = config
        self.results = {}
        
    def run_stable_validation(self) -> Dict[str, Any]:
        """Run validation with numerical stability checks"""
        logger.info("Running STABLE Field Validation")
        
        try:
            # 1. STABLE MASS-ENERGY VALIDATION
            logger.info("1. Stable Mass-Energy Validation")
            mass_results = self.validate_mass_energy_stable()
            self.results['mass_energy'] = mass_results
            
            # 2. STABLE SCALE VALIDATION
            logger.info("2. Stable Scale Validation")
            scale_results = self.validate_scales_stable()
            self.results['scales'] = scale_results
            
            # 3. STABLE FREQUENCY VALIDATION
            logger.info("3. Stable Frequency Validation")
            freq_results = self.validate_frequencies_stable()
            self.results['frequencies'] = freq_results
            
            # 4. STABLE GOLDEN RATIO VALIDATION
            logger.info("4. Stable Golden Ratio Validation")
            golden_results = self.validate_golden_ratio_stable()
            self.results['golden_ratio'] = golden_results
            
            # Generate stable assessment
            assessment = self.generate_stable_assessment()
            self.results['assessment'] = assessment
            
            logger.info("Stable validation completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Stable validation failed: {e}")
            self.results['error'] = str(e)
            return self.results
    
    def validate_mass_energy_stable(self) -> Dict[str, Any]:
        """STABLE mass-energy validation with bounds checking"""
        results = {'status': ValidationStatus.SUCCESS}
        
        try:
            # Test basic E=mc² relationship with safe numerical ranges
            test_masses = {
                'electron': self.config.m_e,
                'proton': self.config.m_p
            }
            
            particle_analysis = {}
            
            for name, mass in test_masses.items():
                # SAFE calculation: use logarithms to avoid large numbers
                if mass > 0 and self.config.c > 0:
                    log_energy = np.log(mass) + 2 * np.log(self.config.c)
                    equivalent_frequency = np.exp(log_energy - np.log(self.config.h))
                    
                    # Safe Compton wavelength
                    compton_wavelength = self.config.h / (mass * self.config.c)
                    
                    # Safe golden ratio analysis
                    if compton_wavelength > 0:
                        planck_length = np.sqrt(self.config.hbar * self.config.G / self.config.c**3)
                        if planck_length > 0:
                            log_ratio = np.log(compton_wavelength) - np.log(planck_length)
                            golden_ratio = 1.618033988749895
                            
                            # Find closest power of golden ratio
                            golden_power = log_ratio / np.log(golden_ratio)
                            nearest_power = round(golden_power)
                            golden_prediction = np.exp(nearest_power * np.log(golden_ratio))
                            
                            # Safe error calculation
                            actual_ratio = np.exp(log_ratio)
                            if actual_ratio > 0 and golden_prediction > 0:
                                error = abs(actual_ratio - golden_prediction) / actual_ratio
                                error = min(error, 1.0)  # Cap at 100% error
                            else:
                                error = 1.0
                        else:
                            error = 1.0
                    else:
                        error = 1.0
                else:
                    equivalent_frequency = 0
                    error = 1.0
                
                particle_analysis[name] = {
                    'equivalent_frequency': equivalent_frequency,
                    'golden_error': error,
                    'status': ValidationStatus.SUCCESS if error < 0.5 else ValidationStatus.WARNING
                }
            
            results['particle_analysis'] = particle_analysis
            
            # Safe 417 THz analysis
            f_417 = self.config.safe_frequencies['solfeggio_417']
            energy_417 = self.config.h * f_417
            mass_417 = energy_417 / self.config.c**2
            electron_ratio = mass_417 / self.config.m_e if self.config.m_e > 0 else 0
            
            results['417THz_analysis'] = {
                'electron_mass_ratio': electron_ratio,
                'status': ValidationStatus.SUCCESS
            }
            
            # Calculate average error safely
            errors = [data['golden_error'] for data in particle_analysis.values()]
            results['average_golden_error'] = np.mean(errors) if errors else 0.0
            
            logger.info(f"Mass-energy validation completed: avg_error={results['average_golden_error']:.3f}")
            
        except Exception as e:
            logger.error(f"Mass-energy validation failed: {e}")
            results['error'] = str(e)
            results['status'] = ValidationStatus.ERROR
            
        return results
    
    def validate_scales_stable(self) -> Dict[str, Any]:
        """STABLE scale validation with bounds checking"""
        results = {'status': ValidationStatus.SUCCESS}
        
        try:
            scale_analysis = {}
            errors = []
            
            golden_ratio = 1.618033988749895
            planck_scale = self.config.safe_scales['planck']
            
            for scale_name, scale_value in self.config.safe_scales.items():
                if scale_value <= 0 or planck_scale <= 0:
                    error = 1.0
                    golden_scale = scale_value
                else:
                    # Safe ratio calculation using logarithms
                    log_ratio = np.log(scale_value) - np.log(planck_scale)
                    golden_power = log_ratio / np.log(golden_ratio)
                    nearest_power = round(golden_power)
                    golden_scale = planck_scale * np.exp(nearest_power * np.log(golden_ratio))
                    
                    # Safe error calculation
                    if scale_value > 0 and golden_scale > 0:
                        error = abs(scale_value - golden_scale) / scale_value
                        error = min(error, 1.0)  # Cap at 100%
                    else:
                        error = 1.0
                
                scale_analysis[scale_name] = {
                    'scale': scale_value,
                    'golden_scale': golden_scale,
                    'error': error,
                    'status': ValidationStatus.SUCCESS if error < 0.2 else ValidationStatus.WARNING
                }
                errors.append(error)
            
            results['scale_analysis'] = scale_analysis
            results['average_scale_error'] = np.mean(errors) if errors else 0.0
            results['scale_quality'] = self._assess_quality(results['average_scale_error'])
            
            logger.info(f"Scale validation completed: avg_error={results['average_scale_error']:.3f}")
            
        except Exception as e:
            logger.error(f"Scale validation failed: {e}")
            results['error'] = str(e)
            results['status'] = ValidationStatus.ERROR
            
        return results
    
    def validate_frequencies_stable(self) -> Dict[str, Any]:
        """STABLE frequency validation with safe resonance calculations"""
        results = {'status': ValidationStatus.SUCCESS}
        
        try:
            frequency_analysis = {}
            resonances = []
            
            base_frequency = self.config.safe_frequencies['solfeggio_417']
            golden_ratio = 1.618033988749895
            
            # Test harmonic relationships safely
            harmonics = [0.618, 1.0, 1.618, 2.618, 4.236]  # Golden ratio series
            
            for i, harmonic in enumerate(harmonics):
                freq = base_frequency * harmonic
                
                # Safe resonance calculation
                resonance = self._calculate_safe_resonance(freq)
                resonances.append(resonance)
                
                frequency_analysis[f'harmonic_{i}'] = {
                    'frequency': freq,
                    'resonance': resonance,
                    'status': ValidationStatus.SUCCESS if resonance > 0.3 else ValidationStatus.WARNING
                }
            
            results['harmonic_analysis'] = frequency_analysis
            results['average_resonance'] = np.mean(resonances) if resonances else 0.0
            results['strongest_resonance'] = max(resonances) if resonances else 0.0
            
            logger.info(f"Frequency validation completed: avg_resonance={results['average_resonance']:.3f}")
            
        except Exception as e:
            logger.error(f"Frequency validation failed: {e}")
            results['error'] = str(e)
            results['status'] = ValidationStatus.ERROR
            
        return results
    
    def validate_golden_ratio_stable(self) -> Dict[str, Any]:
        """STABLE golden ratio validation with safe calculations"""
        results = {'status': ValidationStatus.SUCCESS}
        
        try:
            # Test golden ratio in safe physical relationships
            test_ratios = {
                'electron_proton_mass': self.config.m_e / self.config.m_p,
                'fine_structure_approx': 1/137.035999084,
                'gravitational_constant_ratio': self.config.G / 6.67430e-11
            }
            
            ratio_analysis = {}
            errors = []
            golden_ratio = 1.618033988749895
            
            for name, ratio in test_ratios.items():
                if ratio > 0:
                    # Find closest golden ratio relationship safely
                    log_ratio = np.log(ratio)
                    golden_power = log_ratio / np.log(golden_ratio)
                    nearest_power = round(golden_power)
                    golden_prediction = np.exp(nearest_power * np.log(golden_ratio))
                    
                    error = abs(ratio - golden_prediction) / ratio
                    error = min(error, 1.0)  # Cap at 100%
                else:
                    error = 1.0
                    golden_prediction = ratio
                
                ratio_analysis[name] = {
                    'actual_ratio': ratio,
                    'golden_prediction': golden_prediction,
                    'error': error,
                    'status': ValidationStatus.SUCCESS if error < 0.3 else ValidationStatus.WARNING
                }
                errors.append(error)
            
            results['ratio_analysis'] = ratio_analysis
            results['average_golden_error'] = np.mean(errors) if errors else 0.0
            results['golden_quality'] = self._assess_quality(results['average_golden_error'])
            
            logger.info(f"Golden ratio validation completed: avg_error={results['average_golden_error']:.3f}")
            
        except Exception as e:
            logger.error(f"Golden ratio validation failed: {e}")
            results['error'] = str(e)
            results['status'] = ValidationStatus.ERROR
            
        return results
    
    def _calculate_safe_resonance(self, frequency: float) -> float:
        """Calculate resonance strength with numerical safety"""
        try:
            if frequency <= 0:
                return 0.0
                
            # Safe Planck frequency calculation
            planck_frequency = 1.0 / np.sqrt(self.config.hbar * self.config.G / self.config.c**5)
            if planck_frequency <= 0:
                return 0.0
                
            # Safe ratio calculation using logarithms
            log_ratio = np.log(frequency) - np.log(planck_frequency)
            golden_ratio = 1.618033988749895
            
            # Multiple resonance conditions safely
            conditions = []
            
            # Golden ratio resonance
            golden_power = log_ratio / np.log(golden_ratio)
            golden_distance = abs(golden_power - round(golden_power))
            conditions.append(np.exp(-golden_distance**2 / 0.1))
            
            # Pi resonance
            pi_power = log_ratio / np.log(np.pi)
            pi_distance = abs(pi_power - round(pi_power))
            conditions.append(0.5 * np.exp(-pi_distance**2 / 0.2))
            
            # Euler's number resonance
            e_power = log_ratio / np.log(np.e)
            e_distance = abs(e_power - round(e_power))
            conditions.append(0.3 * np.exp(-e_distance**2 / 0.3))
            
            return float(np.mean(conditions))
            
        except Exception:
            return 0.0
    
    def _assess_quality(self, error: float) -> str:
        """Assess quality based on error"""
        if error < 0.05:
            return "EXCELLENT"
        elif error < 0.1:
            return "GOOD"
        elif error < 0.2:
            return "FAIR"
        elif error < 0.5:
            return "POOR"
        else:
            return "VERY_POOR"
    
    def generate_stable_assessment(self) -> Dict[str, Any]:
        """Generate stable assessment with safe scoring"""
        assessment = {'status': ValidationStatus.SUCCESS}
        
        try:
            scores = []
            weights = []
            findings = []
            
            # Calculate scores safely for each category
            if 'mass_energy' in self.results:
                mass_data = self.results['mass_energy']
                score = 1.0 - min(mass_data.get('average_golden_error', 1.0), 1.0)
                weight = 1.0
                scores.append(score)
                weights.append(weight)
                findings.append(f"Mass-energy: {self._assess_quality(1-score)}")
            
            if 'scales' in self.results:
                scale_data = self.results['scales']
                score = 1.0 - min(scale_data.get('average_scale_error', 1.0), 1.0)
                weight = 0.9
                scores.append(score)
                weights.append(weight)
                findings.append(f"Scale invariance: {scale_data.get('scale_quality', 'UNKNOWN')}")
            
            if 'frequencies' in self.results:
                freq_data = self.results['frequencies']
                score = min(freq_data.get('average_resonance', 0.0), 1.0)
                weight = 0.8
                scores.append(score)
                weights.append(weight)
                resonance = freq_data.get('average_resonance', 0)
                if resonance > 0.7:
                    findings.append("Strong frequency resonances")
                elif resonance > 0.5:
                    findings.append("Moderate frequency resonances")
                else:
                    findings.append("Weak frequency resonances")
            
            if 'golden_ratio' in self.results:
                golden_data = self.results['golden_ratio']
                score = 1.0 - min(golden_data.get('average_golden_error', 1.0), 1.0)
                weight = 0.7
                scores.append(score)
                weights.append(weight)
                findings.append(f"Golden ratio: {golden_data.get('golden_quality', 'UNKNOWN')}")
            
            # Calculate overall score safely
            if scores and weights:
                overall_score = np.average(scores, weights=weights)
                overall_score = max(0.0, min(overall_score, 1.0))  # Bound between 0 and 1
            else:
                overall_score = 0.0
            
            # Determine overall status
            if overall_score > 0.8:
                status = ValidationStatus.SUCCESS
                quality = "EXCELLENT"
            elif overall_score > 0.6:
                status = ValidationStatus.SUCCESS
                quality = "GOOD"
            elif overall_score > 0.4:
                status = ValidationStatus.WARNING
                quality = "FAIR"
            elif overall_score > 0.2:
                status = ValidationStatus.WARNING
                quality = "POOR"
            else:
                status = ValidationStatus.FAILURE
                quality = "VERY_POOR"
            
            assessment['overall_score'] = overall_score
            assessment['overall_quality'] = quality
            assessment['status'] = status
            assessment['key_findings'] = findings
            assessment['categories_tested'] = len(scores)
            
            logger.info(f"STABLE ASSESSMENT: Score = {overall_score:.3f}, Quality = {quality}")
            
        except Exception as e:
            logger.error(f"Assessment generation failed: {e}")
            assessment['error'] = str(e)
            assessment['status'] = ValidationStatus.ERROR
            assessment['overall_score'] = 0.0
            assessment['overall_quality'] = "ERROR"
        
        return assessment
    
    def save_stable_results(self, filename: str = "stable_validation_results.json"):
        """Save results with stable JSON serialization"""
        try:
            # Convert to safe JSON-serializable format
            safe_results = {}
            for key, value in self.results.items():
                if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                    safe_results[key] = value
                elif hasattr(value, 'value'):  # Handle Enum
                    safe_results[key] = value.value
                else:
                    safe_results[key] = str(value)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(safe_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def create_stable_visualization(self):
        """Create stable visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('STABLE FIELD VALIDATION RESULTS', fontsize=16, fontweight='bold')
            
            # 1. Category scores
            if 'assessment' in self.results:
                assessment = self.results['assessment']
                score = assessment.get('overall_score', 0)
                quality = assessment.get('overall_quality', 'UNKNOWN')
                
                ax1 = axes[0, 0]
                colors = {'EXCELLENT': 'green', 'GOOD': 'blue', 'FAIR': 'orange', 
                         'POOR': 'red', 'VERY_POOR': 'darkred', 'ERROR': 'gray'}
                ax1.bar(['Overall'], [score], color=colors.get(quality, 'gray'))
                ax1.set_ylim(0, 1)
                ax1.set_title(f'Overall: {quality}')
                ax1.set_ylabel('Score')
                ax1.grid(True, alpha=0.3)
            
            # 2. Error analysis
            errors = []
            labels = []
            
            if 'mass_energy' in self.results:
                errors.append(self.results['mass_energy'].get('average_golden_error', 0))
                labels.append('Mass')
            
            if 'scales' in self.results:
                errors.append(self.results['scales'].get('average_scale_error', 0))
                labels.append('Scale')
            
            if 'golden_ratio' in self.results:
                errors.append(self.results['golden_ratio'].get('average_golden_error', 0))
                labels.append('Golden')
            
            if errors:
                ax2 = axes[0, 1]
                ax2.bar(labels, errors, color=['red' if e > 0.2 else 'orange' for e in errors])
                ax2.set_title('Average Errors by Category')
                ax2.set_ylabel('Error')
                ax2.grid(True, alpha=0.3)
            
            # 3. Resonance strengths
            if 'frequencies' in self.results:
                freq_data = self.results['frequencies']
                if 'harmonic_analysis' in freq_data:
                    harmonics = list(freq_data['harmonic_analysis'].keys())
                    resonances = [data['resonance'] for data in freq_data['harmonic_analysis'].values()]
                    
                    ax3 = axes[1, 0]
                    ax3.plot(range(len(harmonics)), resonances, 'go-', linewidth=2)
                    ax3.set_title('Frequency Resonances')
                    ax3.set_ylabel('Resonance Strength')
                    ax3.set_xticks(range(len(harmonics)))
                    ax3.set_xticklabels([f'H{i}' for i in range(len(harmonics))])
                    ax3.grid(True, alpha=0.3)
                    ax3.set_ylim(0, 1)
            
            # 4. Quality summary
            if 'assessment' in self.results:
                assessment = self.results['assessment']
                tested = assessment.get('categories_tested', 0)
                
                qualities = []
                for finding in assessment.get('key_findings', []):
                    if 'EXCELLENT' in finding:
                        qualities.append(4)
                    elif 'GOOD' in finding:
                        qualities.append(3)
                    elif 'FAIR' in finding:
                        qualities.append(2)
                    elif 'POOR' in finding:
                        qualities.append(1)
                    else:
                        qualities.append(0)
                
                if qualities:
                    ax4 = axes[1, 1]
                    ax4.bar(['Quality'], [np.mean(qualities)], color='purple', alpha=0.7)
                    ax4.set_ylim(0, 4)
                    ax4.set_title('Average Quality Score')
                    ax4.set_ylabel('Quality (0-4)')
                    ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('stable_validation_results.png', dpi=150, bbox_inches='tight')
            logger.info("Visualization saved: stable_validation_results.png")
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")

def run_stable_validation():
    """Run the stable validation"""
    logger.info("Starting STABLE Field Validation")
    
    try:
        config = StableFieldConfig()
        validator = StableFieldValidator(config)
        
        results = validator.run_stable_validation()
        validator.create_stable_visualization()
        validator.save_stable_results()
        
        # Print stable summary
        assessment = results.get('assessment', {})
        print("\n" + "="*70)
        print("STABLE UNIFIED FIELD VALIDATION - RELIABLE SUMMARY")
        print("="*70)
        print(f"Overall Score: {assessment.get('overall_score', 0):.3f}")
        print(f"Overall Quality: {assessment.get('overall_quality', 'UNKNOWN')}")
        print(f"Validation Status: {assessment.get('status', 'UNKNOWN')}")
        print(f"Categories Tested: {assessment.get('categories_tested', 0)}")
        
        if 'key_findings' in assessment:
            print("\nKEY FINDINGS:")
            for finding in assessment['key_findings']:
                print(f"  • {finding}")
        
        # Print specific metrics safely
        if 'mass_energy' in results:
            mass_data = results['mass_energy']
            print(f"\nMASS-ENERGY ANALYSIS:")
            print(f"  Average Error: {mass_data.get('average_golden_error', 0):.3f}")
            if '417THz_analysis' in mass_data:
                hyp = mass_data['417THz_analysis']
                print(f"  417THz Status: {hyp.get('status', 'UNKNOWN')}")
        
        if 'scales' in results:
            scale_data = results['scales']
            print(f"\nSCALE INVARIANCE:")
            print(f"  Average Error: {scale_data.get('average_scale_error', 0):.3f}")
            print(f"  Quality: {scale_data.get('scale_quality', 'UNKNOWN')}")
        
        if 'frequencies' in results:
            freq_data = results['frequencies']
            print(f"\nFREQUENCY HARMONICS:")
            print(f"  Average Resonance: {freq_data.get('average_resonance', 0):.3f}")
            print(f"  Strongest Resonance: {freq_data.get('strongest_resonance', 0):.3f}")
        
        print("\n" + "="*70)
        
        return validator, results
        
    except Exception as e:
        logger.error(f"Stable validation failed: {e}")
        raise

if __name__ == "__main__":
    validator, results = run_stable_validation()