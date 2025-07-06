#!/usr/bin/env python3
"""
محاكاة الفتائل المتقدمة - النموذج الشامل المحدث
يدمج الانهيار الفجائي والبناء اللاحتمي والانتروبيا مع تحليل متقدم

تطوير: باسل يحيى عبدالله
تحليل وبرمجة: مساعد ذكي متخصص
التاريخ: 7 يناير 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy import stats, optimize
from scipy.fft import fft, fftfreq
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# تعيين الخط العربي
plt.rcParams['font.family'] = ['DejaVu Sans']

class AdvancedFilamentSimulator:
    """
    محاكي الفتائل المتقدم - النموذج الشامل المحدث
    يدمج جميع الآليات الجديدة مع تحليل متقدم
    """
    
    def __init__(self, config=None):
        """تهيئة المحاكي مع الإعدادات المتقدمة"""
        
        # الإعدادات الافتراضية المحدثة
        self.default_config = {
            # معاملات البناء اللاحتمي المحدثة
            'stochastic_building': {
                'lambda_base': 1.5,
                'alpha_weibull': 2.2,
                'beta_weibull': 1.1,
                'feedback_strength': 0.7,
                'memory_decay': 0.15,
                'noise_level': 0.05
            },
            
            # معاملات الانهيار الفجائي المحدثة
            'sudden_collapse': {
                'phi_critical': 12.0,
                'collapse_rate': 2000.0,
                'collapse_threshold': 0.95,
                'energy_release_factor': 0.8,
                'collapse_probability_power': 2.0
            },
            
            # معاملات الانتروبيا المحدثة
            'entropy': {
                'k_boltzmann': 1.38e-23,  # ثابت بولتزمان الحقيقي (مقياس)
                'entropy_coupling': 0.25,
                'max_entropy': 150.0,
                'temperature': 1.5,
                'thermal_fluctuation': 0.1
            },
            
            # معاملات البيئة والتفاعل
            'environment': {
                'external_field': 0.0,
                'damping_coefficient': 0.02,
                'coupling_strength': 0.1,
                'boundary_effects': True
            },
            
            # معاملات المحاكاة
            'simulation': {
                'dt': 0.005,
                'total_time': 100.0,
                'save_interval': 10,
                'analysis_window': 50
            }
        }
        
        # دمج الإعدادات المخصصة
        self.config = self.default_config.copy()
        if config:
            self._update_config(self.config, config)
        
        # متغيرات الحالة المحدثة
        self.reset_simulation()
        
        # إعداد التحليل المتقدم
        self.setup_advanced_analysis()
    
    def _update_config(self, base_config, update_config):
        """تحديث الإعدادات بشكل متداخل"""
        for key, value in update_config.items():
            if isinstance(value, dict) and key in base_config:
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def reset_simulation(self):
        """إعادة تعيين حالة المحاكاة"""
        self.time = 0.0
        self.phi = 0.0
        self.entropy = 0.0
        self.velocity = 0.0  # سرعة تغير الفتيلة
        self.acceleration = 0.0  # تسارع تغير الفتيلة
        
        # تاريخ مفصل
        self.history = {
            'time': [],
            'phi': [],
            'entropy': [],
            'velocity': [],
            'acceleration': [],
            'jumps': [],
            'collapses': [],
            'entropy_production': [],
            'energy_dissipation': []
        }
        
        # إحصائيات متقدمة
        self.statistics = {
            'total_jumps': 0,
            'total_collapses': 0,
            'total_energy_released': 0.0,
            'total_entropy_produced': 0.0,
            'last_jump_time': 0.0,
            'last_collapse_time': 0.0,
            'max_phi_reached': 0.0,
            'max_entropy_reached': 0.0
        }
        
        # حالة النظام
        self.system_state = 'building'  # building, critical, collapsed
        self.cycle_count = 0
    
    def setup_advanced_analysis(self):
        """إعداد أدوات التحليل المتقدم"""
        self.analysis_tools = {
            'fourier_analyzer': FourierAnalyzer(),
            'entropy_analyzer': EntropyAnalyzer(),
            'collapse_predictor': CollapsePredictor(),
            'pattern_detector': PatternDetector()
        }
    
    def calculate_advanced_entropy(self, phi, dphi_dt, d2phi_dt2):
        """
        حساب الانتروبيا المتقدم مع تأثيرات إضافية
        
        Args:
            phi: قيمة الفتيلة
            dphi_dt: السرعة
            d2phi_dt2: التسارع
        
        Returns:
            entropy: الانتروبيا المحسوبة
        """
        if phi <= 0:
            return 0.0
        
        config = self.config['entropy']
        
        # الانتروبيا الأساسية (لوغاريتمية)
        base_entropy = config['k_boltzmann'] * np.log(1 + phi**2)
        
        # انتروبيا الحركة (تعتمد على السرعة والتسارع)
        kinetic_entropy = 0.5 * config['entropy_coupling'] * (dphi_dt**2 + 0.1 * d2phi_dt2**2)
        
        # الانتروبيا الحرارية (تقلبات حرارية)
        thermal_entropy = config['temperature'] * np.sqrt(phi) * (1 + config['thermal_fluctuation'] * np.random.normal(0, 1))
        
        # انتروبيا التفاعل (تعتمد على التاريخ)
        interaction_entropy = 0.0
        if len(self.history['phi']) > 10:
            recent_phi = np.array(self.history['phi'][-10:])
            interaction_entropy = 0.1 * np.std(recent_phi)
        
        # الانتروبيا الكلية
        total_entropy = base_entropy + kinetic_entropy + thermal_entropy + interaction_entropy
        
        # تطبيق الحدود والقيود
        total_entropy = max(0, min(total_entropy, config['max_entropy']))
        
        return total_entropy
    
    def calculate_jump_intensity_advanced(self, phi, entropy, velocity, time):
        """
        حساب شدة القفزات المتقدم مع عوامل إضافية
        
        Args:
            phi: قيمة الفتيلة
            entropy: الانتروبيا
            velocity: السرعة
            time: الزمن
        
        Returns:
            intensity: شدة القفزات
        """
        config = self.config['stochastic_building']
        
        # الشدة الأساسية
        base_intensity = config['lambda_base']
        
        # تأثير الحالة الحالية (غير خطي)
        phi_critical = self.config['sudden_collapse']['phi_critical']
        state_factor = 1 + config['feedback_strength'] * np.tanh(phi / phi_critical)
        
        # تأثير الانتروبيا (تحفيز أو تثبيط)
        max_entropy = self.config['entropy']['max_entropy']
        entropy_factor = 1 + 0.2 * np.sin(np.pi * entropy / max_entropy)
        
        # تأثير السرعة (تغذية راجعة ديناميكية)
        velocity_factor = 1 + 0.1 * np.tanh(velocity)
        
        # تأثير الذاكرة (متقدم)
        time_since_jump = time - self.statistics['last_jump_time']
        memory_factor = 1 + config['memory_decay'] * np.exp(-time_since_jump / 5.0)
        
        # تأثير الدورات (تعلم النظام)
        cycle_factor = 1 + 0.05 * self.cycle_count
        
        # الضوضاء البيئية المتقدمة
        noise_factor = 1 + config['noise_level'] * (np.random.normal(0, 1) + 0.1 * np.sin(time))
        
        # الشدة الكلية
        total_intensity = (base_intensity * state_factor * entropy_factor * 
                          velocity_factor * memory_factor * cycle_factor * noise_factor)
        
        return max(0, total_intensity)
    
    def generate_advanced_jump(self, phi, entropy, velocity):
        """
        توليد قفزة متقدمة مع خصائص معقدة
        
        Args:
            phi: قيمة الفتيلة
            entropy: الانتروبيا
            velocity: السرعة
        
        Returns:
            jump_size: حجم القفزة
            jump_direction: اتجاه القفزة
        """
        config = self.config['stochastic_building']
        
        # الحجم الأساسي من توزيع ويبل معدل
        alpha = config['alpha_weibull'] * (1 + 0.1 * entropy / self.config['entropy']['max_entropy'])
        beta = config['beta_weibull'] * (1 + 0.2 * phi / self.config['sudden_collapse']['phi_critical'])
        
        base_size = np.random.weibull(alpha) * beta
        
        # تعديل الحجم بناءً على الحالة
        phi_critical = self.config['sudden_collapse']['phi_critical']
        state_modifier = 1 + 0.8 * (phi / phi_critical)**2
        
        # تأثير السرعة على الحجم
        velocity_modifier = 1 + 0.3 * np.tanh(abs(velocity))
        
        # تأثير الانتروبيا
        entropy_modifier = 1 + 0.4 * (entropy / self.config['entropy']['max_entropy'])
        
        # الحجم النهائي
        jump_size = base_size * state_modifier * velocity_modifier * entropy_modifier
        
        # تحديد الاتجاه (معظم القفزات إيجابية، لكن يمكن أن تكون سالبة أحياناً)
        direction_probability = 0.9 - 0.2 * (phi / phi_critical)
        jump_direction = 1 if np.random.random() < direction_probability else -1
        
        # تطبيق الاتجاه
        final_jump = jump_size * jump_direction
        
        # التأكد من عدم النزول تحت الصفر
        if phi + final_jump < 0:
            final_jump = -phi * 0.5
        
        return final_jump, jump_direction
    
    def check_advanced_collapse_condition(self, phi, entropy, velocity, acceleration):
        """
        فحص شرط الانهيار المتقدم مع عوامل متعددة
        
        Args:
            phi: قيمة الفتيلة
            entropy: الانتروبيا
            velocity: السرعة
            acceleration: التسارع
        
        Returns:
            should_collapse: هل يجب الانهيار
            collapse_probability: احتمالية الانهيار
        """
        config = self.config['sudden_collapse']
        
        # الشرط الأساسي: الوصول للنقطة الحرجة
        phi_critical = config['phi_critical']
        critical_reached = phi >= phi_critical * config['collapse_threshold']
        
        if not critical_reached:
            return False, 0.0
        
        # احتمالية الانهيار المتقدمة
        excess = phi - phi_critical * config['collapse_threshold']
        range_width = phi_critical * (1 - config['collapse_threshold'])
        
        # العامل الأساسي
        base_probability = (excess / range_width) ** config['collapse_probability_power']
        
        # تأثير الانتروبيا (انتروبيا عالية تزيد الاحتمالية)
        entropy_factor = 1 + 0.5 * (entropy / self.config['entropy']['max_entropy'])
        
        # تأثير السرعة (سرعة عالية تزيد الاحتمالية)
        velocity_factor = 1 + 0.3 * np.tanh(abs(velocity))
        
        # تأثير التسارع (تسارع عالي يزيد الاحتمالية)
        acceleration_factor = 1 + 0.2 * np.tanh(abs(acceleration))
        
        # تأثير التاريخ (كثرة الانهيارات السابقة تقلل الاحتمالية)
        history_factor = 1 / (1 + 0.1 * self.statistics['total_collapses'])
        
        # الاحتمالية الكلية
        total_probability = (base_probability * entropy_factor * velocity_factor * 
                           acceleration_factor * history_factor)
        
        # تطبيق الحدود
        total_probability = min(1.0, max(0.0, total_probability))
        
        # القرار النهائي
        should_collapse = np.random.random() < total_probability
        
        return should_collapse, total_probability
    
    def execute_advanced_collapse(self, phi, entropy, velocity):
        """
        تنفيذ الانهيار المتقدم مع تأثيرات معقدة
        
        Args:
            phi: قيمة الفتيلة قبل الانهيار
            entropy: الانتروبيا قبل الانهيار
            velocity: السرعة قبل الانهيار
        
        Returns:
            new_phi: قيمة الفتيلة بعد الانهيار
            entropy_released: الانتروبيا المحررة
            energy_released: الطاقة المحررة
        """
        config = self.config['sudden_collapse']
        
        # الطاقة المحررة (معادلة متقدمة)
        kinetic_energy = 0.5 * velocity**2
        potential_energy = 0.5 * phi**2
        entropy_energy = 0.1 * entropy
        
        total_energy = (potential_energy + kinetic_energy + entropy_energy) * config['energy_release_factor']
        
        # الانتروبيا المحررة للبيئة
        entropy_released = entropy + total_energy / self.config['entropy']['temperature']
        
        # تحديث الإحصائيات
        self.statistics['total_collapses'] += 1
        self.statistics['total_energy_released'] += total_energy
        self.statistics['last_collapse_time'] = self.time
        self.cycle_count += 1
        
        # تسجيل الانهيار
        collapse_data = {
            'time': self.time,
            'phi_before': phi,
            'entropy_before': entropy,
            'velocity_before': velocity,
            'energy_released': total_energy,
            'entropy_released': entropy_released,
            'cycle_number': self.cycle_count
        }
        
        self.history['collapses'].append(collapse_data)
        
        # تحديث حالة النظام
        self.system_state = 'collapsed'
        
        return 0.0, entropy_released, total_energy
    
    def simulate_step_advanced(self, dt):
        """
        محاكاة خطوة زمنية متقدمة
        
        Args:
            dt: الخطوة الزمنية
        
        Returns:
            state: حالة النظام الجديدة
        """
        # حفظ القيم السابقة
        old_phi = self.phi
        old_velocity = self.velocity
        
        # حساب شدة القفزات
        jump_intensity = self.calculate_jump_intensity_advanced(
            self.phi, self.entropy, self.velocity, self.time
        )
        
        # احتمالية حدوث قفزة
        jump_probability = 1 - np.exp(-jump_intensity * dt)
        
        # متغيرات التغيير
        dphi_dt = 0.0
        d2phi_dt2 = 0.0
        
        # فحص حدوث قفزة
        if np.random.random() < jump_probability and self.system_state != 'collapsed':
            # توليد قفزة متقدمة
            jump_size, jump_direction = self.generate_advanced_jump(
                self.phi, self.entropy, self.velocity
            )
            
            # تطبيق القفزة
            self.phi += jump_size
            dphi_dt = jump_size / dt
            
            # تحديث الإحصائيات
            self.statistics['total_jumps'] += 1
            self.statistics['last_jump_time'] = self.time
            self.statistics['max_phi_reached'] = max(self.statistics['max_phi_reached'], self.phi)
            
            # تسجيل القفزة
            jump_data = {
                'time': self.time,
                'size': jump_size,
                'direction': jump_direction,
                'phi_after': self.phi,
                'intensity': jump_intensity
            }
            
            self.history['jumps'].append(jump_data)
            
            # تحديث حالة النظام
            if self.phi > 0:
                self.system_state = 'building'
        
        # حساب السرعة والتسارع
        self.velocity = dphi_dt
        self.acceleration = (self.velocity - old_velocity) / dt
        d2phi_dt2 = self.acceleration
        
        # فحص شرط الانهيار المتقدم
        should_collapse, collapse_prob = self.check_advanced_collapse_condition(
            self.phi, self.entropy, self.velocity, self.acceleration
        )
        
        if should_collapse and self.system_state == 'building':
            # تنفيذ الانهيار المتقدم
            old_phi = self.phi
            self.phi, entropy_released, energy_released = self.execute_advanced_collapse(
                self.phi, self.entropy, self.velocity
            )
            
            # تحديث المتغيرات
            dphi_dt = -(old_phi) / dt  # معدل انهيار سالب كبير
            self.velocity = 0.0  # السرعة تصبح صفر بعد الانهيار
            self.acceleration = 0.0
        
        # حساب الانتروبيا المتقدمة
        self.entropy = self.calculate_advanced_entropy(self.phi, dphi_dt, d2phi_dt2)
        self.statistics['max_entropy_reached'] = max(self.statistics['max_entropy_reached'], self.entropy)
        
        # حساب إنتاج الانتروبيا
        entropy_production = max(0, self.entropy - (self.history['entropy'][-1] if self.history['entropy'] else 0))
        self.statistics['total_entropy_produced'] += entropy_production
        
        # تحديث الزمن
        self.time += dt
        
        # تسجيل التاريخ
        self.history['time'].append(self.time)
        self.history['phi'].append(self.phi)
        self.history['entropy'].append(self.entropy)
        self.history['velocity'].append(self.velocity)
        self.history['acceleration'].append(self.acceleration)
        self.history['entropy_production'].append(entropy_production)
        
        return {
            'phi': self.phi,
            'entropy': self.entropy,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'system_state': self.system_state,
            'jump_probability': jump_probability,
            'collapse_probability': collapse_prob if 'collapse_prob' in locals() else 0.0
        }
    
    def run_advanced_simulation(self):
        """تشغيل المحاكاة المتقدمة الكاملة"""
        
        # إعدادات المحاكاة
        dt = self.config['simulation']['dt']
        total_time = self.config['simulation']['total_time']
        save_interval = self.config['simulation']['save_interval']
        
        steps = int(total_time / dt)
        save_every = max(1, int(save_interval / dt))
        
        print(f"بدء المحاكاة المتقدمة:")
        print(f"  - عدد الخطوات: {steps}")
        print(f"  - الزمن الكلي: {total_time}")
        print(f"  - الخطوة الزمنية: {dt}")
        print(f"  - حفظ كل: {save_every} خطوة")
        print()
        
        # إعادة تعيين الحالة
        self.reset_simulation()
        
        # تشغيل المحاكاة
        for i in range(steps):
            # تنفيذ خطوة
            state = self.simulate_step_advanced(dt)
            
            # طباعة التقدم
            if i % (steps // 20) == 0:
                progress = (i / steps) * 100
                print(f"التقدم: {progress:5.1f}% | "
                      f"الفتيلة: {state['phi']:6.3f} | "
                      f"الانتروبيا: {state['entropy']:6.3f} | "
                      f"السرعة: {state['velocity']:6.3f} | "
                      f"الحالة: {state['system_state']}")
        
        print(f"\nانتهت المحاكاة:")
        print(f"  - القفزات الكلية: {self.statistics['total_jumps']}")
        print(f"  - الانهيارات الكلية: {self.statistics['total_collapses']}")
        print(f"  - الطاقة المحررة: {self.statistics['total_energy_released']:.3f}")
        print(f"  - الانتروبيا المنتجة: {self.statistics['total_entropy_produced']:.3f}")
        print(f"  - أقصى فتيلة: {self.statistics['max_phi_reached']:.3f}")
        print(f"  - أقصى انتروبيا: {self.statistics['max_entropy_reached']:.3f}")
        
        return self.get_results()
    
    def get_results(self):
        """الحصول على نتائج المحاكاة المنظمة"""
        return {
            'config': self.config,
            'statistics': self.statistics,
            'history': self.history,
            'analysis': self.perform_advanced_analysis(),
            'metadata': {
                'simulation_time': self.time,
                'total_steps': len(self.history['time']),
                'generated_at': datetime.now().isoformat(),
                'model_version': '2.0_advanced'
            }
        }
    
    def perform_advanced_analysis(self):
        """تحليل متقدم للنتائج"""
        if len(self.history['time']) < 10:
            return {}
        
        # تحويل إلى arrays
        times = np.array(self.history['time'])
        phi_values = np.array(self.history['phi'])
        entropy_values = np.array(self.history['entropy'])
        velocity_values = np.array(self.history['velocity'])
        
        analysis = {}
        
        # تحليل فورييه
        if len(phi_values) > 50:
            analysis['fourier'] = self.analysis_tools['fourier_analyzer'].analyze(phi_values, times)
        
        # تحليل الانتروبيا
        analysis['entropy_analysis'] = self.analysis_tools['entropy_analyzer'].analyze(
            entropy_values, phi_values, times
        )
        
        # تنبؤ الانهيار
        analysis['collapse_prediction'] = self.analysis_tools['collapse_predictor'].analyze(
            phi_values, entropy_values, velocity_values, self.history['collapses']
        )
        
        # كشف الأنماط
        analysis['patterns'] = self.analysis_tools['pattern_detector'].analyze(
            phi_values, entropy_values, self.history['jumps'], self.history['collapses']
        )
        
        return analysis

# أدوات التحليل المتقدم
class FourierAnalyzer:
    """محلل فورييه للإشارات"""
    
    def analyze(self, signal, times):
        """تحليل فورييه للإشارة"""
        if len(signal) < 10:
            return {}
        
        # تحليل فورييه
        fft_values = fft(signal)
        freqs = fftfreq(len(signal), times[1] - times[0])
        
        # القوة الطيفية
        power_spectrum = np.abs(fft_values)**2
        
        # الترددات المهيمنة
        dominant_freqs = freqs[np.argsort(power_spectrum)[-5:]]
        
        return {
            'dominant_frequencies': dominant_freqs.tolist(),
            'total_power': np.sum(power_spectrum),
            'peak_frequency': freqs[np.argmax(power_spectrum)],
            'spectral_centroid': np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        }

class EntropyAnalyzer:
    """محلل الانتروبيا المتقدم"""
    
    def analyze(self, entropy_values, phi_values, times):
        """تحليل سلوك الانتروبيا"""
        if len(entropy_values) < 10:
            return {}
        
        # معدل إنتاج الانتروبيا
        entropy_production_rate = np.gradient(entropy_values, times)
        
        # الارتباط مع الفتيلة
        correlation = np.corrcoef(entropy_values, phi_values)[0, 1] if len(entropy_values) == len(phi_values) else 0
        
        # نقاط التحول
        turning_points = []
        for i in range(1, len(entropy_production_rate) - 1):
            if ((entropy_production_rate[i-1] < entropy_production_rate[i] > entropy_production_rate[i+1]) or
                (entropy_production_rate[i-1] > entropy_production_rate[i] < entropy_production_rate[i+1])):
                turning_points.append(i)
        
        return {
            'mean_entropy': np.mean(entropy_values),
            'entropy_variance': np.var(entropy_values),
            'mean_production_rate': np.mean(entropy_production_rate),
            'phi_correlation': correlation,
            'turning_points_count': len(turning_points),
            'max_production_rate': np.max(entropy_production_rate),
            'min_production_rate': np.min(entropy_production_rate)
        }

class CollapsePredictor:
    """متنبئ الانهيار المتقدم"""
    
    def analyze(self, phi_values, entropy_values, velocity_values, collapses):
        """تحليل أنماط الانهيار والتنبؤ"""
        if len(collapses) < 2:
            return {}
        
        # أزمنة الانهيارات
        collapse_times = [c['time'] for c in collapses]
        
        # الفترات بين الانهيارات
        intervals = []
        for i in range(1, len(collapse_times)):
            intervals.append(collapse_times[i] - collapse_times[i-1])
        
        # خصائص ما قبل الانهيار
        pre_collapse_phi = [c['phi_before'] for c in collapses]
        pre_collapse_entropy = [c['entropy_before'] for c in collapses]
        
        return {
            'mean_interval': np.mean(intervals) if intervals else 0,
            'interval_variance': np.var(intervals) if intervals else 0,
            'mean_pre_collapse_phi': np.mean(pre_collapse_phi),
            'mean_pre_collapse_entropy': np.mean(pre_collapse_entropy),
            'collapse_frequency': len(collapses) / (collapse_times[-1] - collapse_times[0]) if len(collapse_times) > 1 else 0,
            'predictability_score': 1 / (1 + np.var(intervals)) if intervals else 0
        }

class PatternDetector:
    """كاشف الأنماط المتقدم"""
    
    def analyze(self, phi_values, entropy_values, jumps, collapses):
        """كشف الأنماط في السلوك"""
        patterns = {}
        
        # أنماط القفزات
        if jumps:
            jump_sizes = [j['size'] for j in jumps]
            jump_times = [j['time'] for j in jumps]
            
            patterns['jump_patterns'] = {
                'size_trend': self._detect_trend(jump_sizes),
                'temporal_clustering': self._detect_clustering(jump_times),
                'size_distribution': self._analyze_distribution(jump_sizes)
            }
        
        # أنماط الدورات
        if collapses and len(collapses) > 2:
            collapse_times = [c['time'] for c in collapses]
            patterns['cycle_patterns'] = {
                'regularity': self._measure_regularity(collapse_times),
                'trend': self._detect_trend(collapse_times)
            }
        
        # أنماط الفتيلة والانتروبيا
        patterns['correlation_patterns'] = {
            'phi_entropy_correlation': np.corrcoef(phi_values, entropy_values)[0, 1] if len(phi_values) == len(entropy_values) else 0,
            'phase_relationship': self._analyze_phase_relationship(phi_values, entropy_values)
        }
        
        return patterns
    
    def _detect_trend(self, values):
        """كشف الاتجاه في القيم"""
        if len(values) < 3:
            return 'insufficient_data'
        
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)
        
        if abs(r_value) < 0.3:
            return 'no_trend'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _detect_clustering(self, times):
        """كشف التجمع الزمني"""
        if len(times) < 3:
            return 'insufficient_data'
        
        intervals = np.diff(times)
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        
        if cv < 0.5:
            return 'regular'
        elif cv > 1.5:
            return 'clustered'
        else:
            return 'random'
    
    def _analyze_distribution(self, values):
        """تحليل التوزيع"""
        if len(values) < 5:
            return 'insufficient_data'
        
        # اختبار التوزيع الطبيعي
        _, p_normal = stats.normaltest(values)
        
        # اختبار التوزيع الأسي
        _, p_exp = stats.kstest(values, 'expon')
        
        if p_normal > 0.05:
            return 'normal'
        elif p_exp > 0.05:
            return 'exponential'
        else:
            return 'other'
    
    def _measure_regularity(self, times):
        """قياس انتظام الأحداث"""
        if len(times) < 3:
            return 0
        
        intervals = np.diff(times)
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        
        return 1 / (1 + cv)  # قيمة بين 0 و 1
    
    def _analyze_phase_relationship(self, signal1, signal2):
        """تحليل العلاقة الطورية بين إشارتين"""
        if len(signal1) != len(signal2) or len(signal1) < 10:
            return 'insufficient_data'
        
        # حساب الارتباط المتقاطع
        correlation = np.correlate(signal1, signal2, mode='full')
        max_corr_index = np.argmax(correlation)
        phase_shift = max_corr_index - len(signal2) + 1
        
        if abs(phase_shift) < len(signal1) * 0.1:
            return 'in_phase'
        elif abs(phase_shift) > len(signal1) * 0.4:
            return 'out_of_phase'
        else:
            return 'phase_shifted'

def run_comprehensive_advanced_simulation():
    """تشغيل محاكاة شاملة متقدمة"""
    
    print("=" * 60)
    print("محاكاة الفتائل المتقدمة - النموذج الشامل المحدث")
    print("تطوير: باسل يحيى عبدالله")
    print("التاريخ: 7 يناير 2025")
    print("=" * 60)
    print()
    
    # سيناريوهات الاختبار المتقدمة
    scenarios = [
        {
            'name': 'النموذج الأساسي المحدث',
            'config': {}
        },
        {
            'name': 'بناء سريع مع انتروبيا عالية',
            'config': {
                'stochastic_building': {
                    'lambda_base': 2.5,
                    'feedback_strength': 1.0
                },
                'entropy': {
                    'entropy_coupling': 0.4,
                    'temperature': 2.0
                }
            }
        },
        {
            'name': 'انهيار حساس مع ذاكرة قوية',
            'config': {
                'sudden_collapse': {
                    'collapse_threshold': 0.85,
                    'collapse_probability_power': 3.0
                },
                'stochastic_building': {
                    'memory_decay': 0.3,
                    'feedback_strength': 0.9
                }
            }
        }
    ]
    
    all_results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*20} {scenario['name']} {'='*20}")
        
        # إنشاء المحاكي
        simulator = AdvancedFilamentSimulator(scenario['config'])
        
        # تشغيل المحاكاة
        results = simulator.run_advanced_simulation()
        
        # حفظ النتائج
        all_results[scenario['name']] = results
        
        # طباعة الملخص
        stats = results['statistics']
        print(f"\nملخص النتائج:")
        print(f"  القفزات: {stats['total_jumps']}")
        print(f"  الانهيارات: {stats['total_collapses']}")
        print(f"  الطاقة المحررة: {stats['total_energy_released']:.3f}")
        print(f"  الانتروبيا المنتجة: {stats['total_entropy_produced']:.3f}")
        print(f"  أقصى فتيلة: {stats['max_phi_reached']:.3f}")
        print(f"  أقصى انتروبيا: {stats['max_entropy_reached']:.3f}")
        
        # حفظ النتائج
        filename = f'/home/ubuntu/advanced_results_{scenario["name"].replace(" ", "_")}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            # تحويل numpy arrays إلى lists
            serializable_results = convert_to_serializable(results)
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"  تم حفظ النتائج في: {filename}")
    
    print(f"\n{'='*60}")
    print("انتهت جميع المحاكاات المتقدمة بنجاح!")
    print("="*60)
    
    return all_results

def convert_to_serializable(obj):
    """تحويل الكائنات إلى قابلة للتسلسل"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

if __name__ == "__main__":
    # تشغيل المحاكاة الشاملة المتقدمة
    results = run_comprehensive_advanced_simulation()

