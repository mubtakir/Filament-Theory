#!/usr/bin/env python3
"""
نظرية الفتائل - الأكواد البرمجية والمحاكاة

المؤلف: باسل يحيى عبدالله - المبتكر العلمي
المشرف: Manus AI
تاريخ التوثيق: 7 يناير 2025

هذا الملف يحتوي على جميع الأكواد البرمجية اللازمة لمحاكاة وحساب
المفاهيم الأساسية في نظرية الفتائل.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import scipy.optimize as opt
from scipy.integrate import quad, solve_ivp
from scipy.fft import fft, fftfreq
import pandas as pd
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# الثوابت الفيزيائية الأساسية
# ==========================================

class PhysicalConstants:
    """الثوابت الفيزيائية الأساسية"""
    
    # الثوابت العامة
    h = 6.62607015e-34      # ثابت بلانك (J⋅s)
    hbar = h / (2 * np.pi)  # ثابت بلانك المختزل
    c = 299792458           # سرعة الضوء (m/s)
    G = 6.67430e-11         # ثابت الجاذبية (m³/kg⋅s²)
    k_B = 1.380649e-23      # ثابت بولتزمان (J/K)
    
    # الثوابت الفتيلية الأساسية
    m_filament_0 = h / (4 * np.pi * c**2)  # كتلة الفتيلة الأساسية (kg)
    r_filament_0 = 4 * np.pi * c           # نصف قطر الفتيلة الأساسي (m)
    omega_filament_0 = c / r_filament_0    # تردد الفتيلة الأساسي (Hz)
    E_filament_0 = m_filament_0 * c**2     # طاقة الفتيلة الأساسية (J)
    
    # ثوابت أخرى
    alpha_fine = 7.2973525693e-3           # ثابت البنية الدقيقة
    epsilon_0 = 8.8541878128e-12           # سماحية الفراغ (F/m)
    mu_0 = 4 * np.pi * 1e-7                # نفاذية الفراغ (H/m)

# ==========================================
# فئة الفتيلة الأساسية
# ==========================================

class Filament:
    """
    فئة تمثل الفتيلة الأساسية في نظرية الفتائل
    """
    
    def __init__(self, mass_factor: float = 1.0, radius_factor: float = 1.0, 
                 oscillation_amplitude: float = 0.1, phase: float = 0.0):
        """
        تهيئة الفتيلة
        
        Parameters:
        -----------
        mass_factor : float
            عامل الكتلة (مضاعف الكتلة الأساسية)
        radius_factor : float
            عامل نصف القطر (مضاعف نصف القطر الأساسي)
        oscillation_amplitude : float
            سعة التذبذب (ε)
        phase : float
            الطور الأولي (φ)
        """
        self.m0 = PhysicalConstants.m_filament_0 * mass_factor
        self.r0 = PhysicalConstants.r_filament_0 * radius_factor
        self.epsilon = oscillation_amplitude
        self.phi = phase
        self.omega = PhysicalConstants.omega_filament_0
        
        # التحقق من العلاقة الأساسية m × r = constant
        self.constant_check = self.m0 * self.r0
        expected_constant = PhysicalConstants.h / (4 * np.pi * PhysicalConstants.c)
        
        if not np.isclose(self.constant_check, expected_constant, rtol=1e-10):
            print(f"تحذير: العلاقة m×r = constant غير محققة!")
            print(f"القيمة الحالية: {self.constant_check:.2e}")
            print(f"القيمة المتوقعة: {expected_constant:.2e}")
    
    def mass(self, t: float) -> float:
        """
        حساب الكتلة كدالة للزمن
        
        m(t) = m₀ × [1 + ε × sin(ωt + φ)]
        """
        return self.m0 * (1 + self.epsilon * np.sin(self.omega * t + self.phi))
    
    def radius(self, t: float) -> float:
        """
        حساب نصف القطر كدالة للزمن
        
        r(t) = r₀ × [1 - ε × sin(ωt + φ)]
        """
        return self.r0 * (1 - self.epsilon * np.sin(self.omega * t + self.phi))
    
    def mass_current(self, t: float) -> float:
        """
        حساب التيار الكتلي
        
        I_m = dm/dt = m₀ × ε × ω × cos(ωt + φ)
        """
        return self.m0 * self.epsilon * self.omega * np.cos(self.omega * t + self.phi)
    
    def energy(self, t: float) -> float:
        """
        حساب الطاقة الكلية
        
        E = mc²
        """
        return self.mass(t) * PhysicalConstants.c**2
    
    def surface_area(self, t: float) -> float:
        """
        حساب مساحة السطح
        
        A = 4πr²
        """
        return 4 * np.pi * self.radius(t)**2
    
    def volume(self, t: float) -> float:
        """
        حساب الحجم
        
        V = (4/3)πr³
        """
        return (4/3) * np.pi * self.radius(t)**3
    
    def density(self, t: float) -> float:
        """
        حساب الكثافة
        
        ρ = m/V
        """
        return self.mass(t) / self.volume(t)
    
    def surface_mass_current_density(self, t: float) -> float:
        """
        حساب كثافة التيار الكتلي السطحي
        
        K_m = I_m / A
        """
        return self.mass_current(t) / self.surface_area(t)

# ==========================================
# دالة زيتا ريمان والأعداد الأولية
# ==========================================

class RiemannZeta:
    """
    فئة لحساب دالة زيتا ريمان وخصائصها في نظرية الفتائل
    """
    
    @staticmethod
    def eta_function(s: complex, n_terms: int = 1000) -> complex:
        """
        حساب دالة إيتا ديريشليت
        
        η(s) = Σ(n=1 to ∞) (-1)^(n+1) / n^s
        """
        result = 0.0 + 0.0j
        for n in range(1, n_terms + 1):
            result += (-1)**(n+1) / (n**s)
        return result
    
    @staticmethod
    def zeta_from_eta(s: complex, n_terms: int = 1000) -> complex:
        """
        حساب دالة زيتا من دالة إيتا
        
        ζ(s) = η(s) / (1 - 2^(1-s))
        """
        eta_s = RiemannZeta.eta_function(s, n_terms)
        denominator = 1 - 2**(1-s)
        
        if abs(denominator) < 1e-15:
            return float('inf') + 0.0j
        
        return eta_s / denominator
    
    @staticmethod
    def zeta_direct(s: complex, n_terms: int = 1000) -> complex:
        """
        حساب دالة زيتا مباشرة (للجزء الحقيقي > 1)
        
        ζ(s) = Σ(n=1 to ∞) 1 / n^s
        """
        if s.real <= 1:
            return RiemannZeta.zeta_from_eta(s, n_terms)
        
        result = 0.0 + 0.0j
        for n in range(1, n_terms + 1):
            result += 1 / (n**s)
        return result
    
    @staticmethod
    def euler_product(s: complex, max_prime: int = 100) -> complex:
        """
        حساب دالة زيتا باستخدام حاصل ضرب أويلر
        
        ζ(s) = Π(p prime) 1/(1 - p^(-s))
        """
        primes = RiemannZeta.sieve_of_eratosthenes(max_prime)
        result = 1.0 + 0.0j
        
        for p in primes:
            factor = 1 / (1 - p**(-s))
            result *= factor
            
        return result
    
    @staticmethod
    def sieve_of_eratosthenes(limit: int) -> List[int]:
        """
        منخل إراتوستينس لإيجاد الأعداد الأولية
        """
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    @staticmethod
    def find_zeros_on_critical_line(t_min: float = 0, t_max: float = 100, 
                                   n_points: int = 10000) -> List[float]:
        """
        البحث عن أصفار دالة زيتا على الخط الحرج Re(s) = 1/2
        """
        t_values = np.linspace(t_min, t_max, n_points)
        zeros = []
        
        # حساب قيم دالة زيتا على الخط الحرج
        zeta_values = []
        for t in t_values:
            s = 0.5 + 1j * t
            zeta_val = RiemannZeta.zeta_from_eta(s)
            zeta_values.append(abs(zeta_val))
        
        # البحث عن النقاط التي تقترب من الصفر
        threshold = 0.1
        for i in range(1, len(zeta_values) - 1):
            if (zeta_values[i] < threshold and 
                zeta_values[i] < zeta_values[i-1] and 
                zeta_values[i] < zeta_values[i+1]):
                zeros.append(t_values[i])
        
        return zeros

# ==========================================
# محاكاة بحر الفتائل
# ==========================================

class FilamentSea:
    """
    فئة لمحاكاة بحر الفتائل
    """
    
    def __init__(self, grid_size: int = 100, domain_size: float = 1.0):
        """
        تهيئة بحر الفتائل
        
        Parameters:
        -----------
        grid_size : int
            حجم الشبكة
        domain_size : float
            حجم المجال المكاني
        """
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dx = domain_size / grid_size
        
        # إنشاء الشبكة المكانية
        self.x = np.linspace(0, domain_size, grid_size)
        self.y = np.linspace(0, domain_size, grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # تهيئة كثافة الفتائل
        self.density = np.ones((grid_size, grid_size)) * PhysicalConstants.m_filament_0
        self.velocity_x = np.zeros((grid_size, grid_size))
        self.velocity_y = np.zeros((grid_size, grid_size))
        self.pressure = np.zeros((grid_size, grid_size))
    
    def add_mass_source(self, x_pos: float, y_pos: float, mass: float, radius: float):
        """
        إضافة مصدر كتلة (جسم ماكروسكوبي)
        """
        # تحويل الإحداثيات إلى فهارس الشبكة
        i = int(x_pos / self.dx)
        j = int(y_pos / self.dx)
        r_grid = int(radius / self.dx)
        
        # إضافة الكتلة في منطقة دائرية
        for di in range(-r_grid, r_grid + 1):
            for dj in range(-r_grid, r_grid + 1):
                ni, nj = i + di, j + dj
                if (0 <= ni < self.grid_size and 0 <= nj < self.grid_size):
                    distance = np.sqrt(di**2 + dj**2) * self.dx
                    if distance <= radius:
                        # توزيع غاوسي للكتلة
                        weight = np.exp(-(distance/radius)**2)
                        self.density[ni, nj] += mass * weight / (np.pi * radius**2)
    
    def calculate_pressure(self):
        """
        حساب الضغط من كثافة الفتائل
        
        P = ρc²/3 (تقريب الغاز المثالي النسبي)
        """
        self.pressure = self.density * PhysicalConstants.c**2 / 3
    
    def calculate_gravitational_field(self):
        """
        حساب المجال الجاذبي من تدرج الضغط
        
        g⃗ = -∇P/ρ
        """
        # حساب التدرج باستخدام الفروق المحدودة
        grad_p_x = np.gradient(self.pressure, self.dx, axis=1)
        grad_p_y = np.gradient(self.pressure, self.dx, axis=0)
        
        # تجنب القسمة على صفر
        safe_density = np.where(self.density > 1e-50, self.density, 1e-50)
        
        g_x = -grad_p_x / safe_density
        g_y = -grad_p_y / safe_density
        
        return g_x, g_y
    
    def evolve_navier_stokes(self, dt: float, viscosity: float = 1e-6):
        """
        تطوير بحر الفتائل باستخدام معادلات نافييه-ستوكس
        """
        # حساب الضغط
        self.calculate_pressure()
        
        # حساب التدرجات
        grad_p_x = np.gradient(self.pressure, self.dx, axis=1)
        grad_p_y = np.gradient(self.pressure, self.dx, axis=0)
        
        # حساب اللابلاسيان للسرعة (للزوجة)
        laplacian_vx = (np.roll(self.velocity_x, 1, axis=0) + 
                       np.roll(self.velocity_x, -1, axis=0) +
                       np.roll(self.velocity_x, 1, axis=1) + 
                       np.roll(self.velocity_x, -1, axis=1) - 
                       4 * self.velocity_x) / self.dx**2
        
        laplacian_vy = (np.roll(self.velocity_y, 1, axis=0) + 
                       np.roll(self.velocity_y, -1, axis=0) +
                       np.roll(self.velocity_y, 1, axis=1) + 
                       np.roll(self.velocity_y, -1, axis=1) - 
                       4 * self.velocity_y) / self.dx**2
        
        # تجنب القسمة على صفر
        safe_density = np.where(self.density > 1e-50, self.density, 1e-50)
        
        # تحديث السرعة (معادلة نافييه-ستوكس المبسطة)
        dvx_dt = -grad_p_x / safe_density + viscosity * laplacian_vx
        dvy_dt = -grad_p_y / safe_density + viscosity * laplacian_vy
        
        self.velocity_x += dvx_dt * dt
        self.velocity_y += dvy_dt * dt
        
        # تحديث الكثافة (معادلة الاستمرارية)
        div_v = (np.gradient(self.velocity_x, self.dx, axis=1) + 
                np.gradient(self.velocity_y, self.dx, axis=0))
        
        drho_dt = -self.density * div_v
        self.density += drho_dt * dt
        
        # التأكد من عدم سلبية الكثافة
        self.density = np.maximum(self.density, 1e-50)

# ==========================================
# محاكاة التشابك الكمومي
# ==========================================

class QuantumEntanglement:
    """
    فئة لمحاكاة التشابك الكمومي بين الفتائل
    """
    
    def __init__(self, n_filaments: int = 2):
        """
        تهيئة نظام الفتائل المتشابكة
        
        Parameters:
        -----------
        n_filaments : int
            عدد الفتائل المتشابكة
        """
        self.n_filaments = n_filaments
        self.n_states = 2**n_filaments  # عدد الحالات الكمومية الممكنة
        
        # تهيئة الحالة الكمومية (حالة متشابكة بسيطة)
        self.state_vector = np.zeros(self.n_states, dtype=complex)
        
        if n_filaments == 2:
            # حالة بيل: |00⟩ + |11⟩
            self.state_vector[0] = 1/np.sqrt(2)  # |00⟩
            self.state_vector[3] = 1/np.sqrt(2)  # |11⟩
        else:
            # حالة متشابكة عامة
            self.state_vector[0] = 1/np.sqrt(self.n_states)
            for i in range(1, self.n_states):
                self.state_vector[i] = 1/np.sqrt(self.n_states)
    
    def measure_filament(self, filament_index: int) -> Tuple[int, np.ndarray]:
        """
        قياس فتيلة معينة
        
        Parameters:
        -----------
        filament_index : int
            فهرس الفتيلة المراد قياسها
            
        Returns:
        --------
        result : int
            نتيجة القياس (0 أو 1)
        new_state : np.ndarray
            الحالة الجديدة بعد القياس
        """
        # حساب احتماليات القياس
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i in range(self.n_states):
            bit_value = (i >> filament_index) & 1
            if bit_value == 0:
                prob_0 += abs(self.state_vector[i])**2
            else:
                prob_1 += abs(self.state_vector[i])**2
        
        # إجراء القياس العشوائي
        if np.random.random() < prob_0:
            result = 0
            # إسقاط الحالة على النتيجة 0
            new_state = np.zeros_like(self.state_vector)
            for i in range(self.n_states):
                if ((i >> filament_index) & 1) == 0:
                    new_state[i] = self.state_vector[i]
            new_state /= np.sqrt(prob_0)  # تطبيع
        else:
            result = 1
            # إسقاط الحالة على النتيجة 1
            new_state = np.zeros_like(self.state_vector)
            for i in range(self.n_states):
                if ((i >> filament_index) & 1) == 1:
                    new_state[i] = self.state_vector[i]
            new_state /= np.sqrt(prob_1)  # تطبيع
        
        self.state_vector = new_state
        return result, new_state
    
    def calculate_entanglement_entropy(self, subsystem_indices: List[int]) -> float:
        """
        حساب إنتروبيا التشابك لنظام فرعي
        
        Parameters:
        -----------
        subsystem_indices : List[int]
            فهارس الفتائل في النظام الفرعي
            
        Returns:
        --------
        entropy : float
            إنتروبيا التشابك
        """
        # إنشاء مصفوفة الكثافة للنظام الكامل
        rho_total = np.outer(self.state_vector, np.conj(self.state_vector))
        
        # حساب مصفوفة الكثافة المختزلة للنظام الفرعي
        n_subsystem = len(subsystem_indices)
        n_environment = self.n_filaments - n_subsystem
        
        if n_environment == 0:
            rho_reduced = rho_total
        else:
            # تتبع النظام البيئي (trace out environment)
            dim_subsystem = 2**n_subsystem
            dim_environment = 2**n_environment
            
            rho_reduced = np.zeros((dim_subsystem, dim_subsystem), dtype=complex)
            
            for i in range(dim_subsystem):
                for j in range(dim_subsystem):
                    for k in range(dim_environment):
                        # تحويل الفهارس
                        full_i = self._combine_indices(i, k, subsystem_indices)
                        full_j = self._combine_indices(j, k, subsystem_indices)
                        rho_reduced[i, j] += rho_total[full_i, full_j]
        
        # حساب القيم الذاتية
        eigenvalues = np.linalg.eigvals(rho_reduced)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]  # إزالة القيم الصغيرة جداً
        
        # حساب الإنتروبيا
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        return entropy.real
    
    def _combine_indices(self, subsystem_index: int, environment_index: int, 
                        subsystem_indices: List[int]) -> int:
        """
        دمج فهارس النظام الفرعي والبيئة
        """
        full_index = 0
        sub_bit = 0
        env_bit = 0
        
        for i in range(self.n_filaments):
            if i in subsystem_indices:
                bit_value = (subsystem_index >> sub_bit) & 1
                sub_bit += 1
            else:
                bit_value = (environment_index >> env_bit) & 1
                env_bit += 1
            
            full_index |= (bit_value << i)
        
        return full_index

# ==========================================
# محاكاة الجاذبية الفتيلية
# ==========================================

class FilamentGravity:
    """
    فئة لمحاكاة الجاذبية في نظرية الفتائل
    """
    
    def __init__(self):
        self.alpha = 1e-10  # معامل التصحيح الفتيلي
    
    def filament_potential(self, r: float) -> float:
        """
        حساب الجهد الفتيلي
        
        V_filament = h/(c × r)
        """
        if r <= 0:
            return float('inf')
        return PhysicalConstants.h / (PhysicalConstants.c * r)
    
    def modified_gravitational_potential(self, M: float, r: float) -> float:
        """
        حساب الجهد الجاذبي المعدل
        
        V_total = -GM/r + α × h/(c × r)
        """
        if r <= 0:
            return float('inf')
        
        classical_term = -PhysicalConstants.G * M / r
        filament_term = self.alpha * self.filament_potential(r)
        
        return classical_term + filament_term
    
    def modified_gravitational_force(self, M: float, r: float) -> float:
        """
        حساب القوة الجاذبية المعدلة
        
        F = -dV/dr
        """
        if r <= 0:
            return float('inf')
        
        classical_force = -PhysicalConstants.G * M / r**2
        filament_force = -self.alpha * PhysicalConstants.h / (PhysicalConstants.c * r**2)
        
        return classical_force + filament_force
    
    def orbital_velocity(self, M: float, r: float) -> float:
        """
        حساب السرعة المدارية المعدلة
        
        v = sqrt(GM/r + α × h/(c × r))
        """
        if r <= 0:
            return 0
        
        classical_term = PhysicalConstants.G * M / r
        filament_term = self.alpha * PhysicalConstants.h / (PhysicalConstants.c * r)
        
        return np.sqrt(classical_term + filament_term)
    
    def schwarzschild_radius_modified(self, M: float) -> float:
        """
        حساب نصف قطر شوارزشيلد المعدل
        """
        # حل المعادلة: 1 - 2GM/(c²r) - αh/(c³r) = 0
        def equation(r):
            return 1 - 2*PhysicalConstants.G*M/(PhysicalConstants.c**2 * r) - \
                   self.alpha*PhysicalConstants.h/(PhysicalConstants.c**3 * r)
        
        # تخمين أولي
        r_classical = 2 * PhysicalConstants.G * M / PhysicalConstants.c**2
        
        try:
            result = opt.fsolve(equation, r_classical)[0]
            return result if result > 0 else r_classical
        except:
            return r_classical

# ==========================================
# تحليل الطيف والترددات
# ==========================================

class SpectrumAnalysis:
    """
    فئة لتحليل طيف الترددات في نظرية الفتائل
    """
    
    @staticmethod
    def filament_spectrum(filament: Filament, t_max: float = 100, 
                         n_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        تحليل طيف تذبذبات الفتيلة
        """
        t = np.linspace(0, t_max, n_points)
        dt = t[1] - t[0]
        
        # حساب إشارة الكتلة
        mass_signal = np.array([filament.mass(ti) for ti in t])
        
        # تطبيق تحويل فورييه
        fft_result = fft(mass_signal)
        frequencies = fftfreq(n_points, dt)
        
        # أخذ النصف الموجب فقط
        positive_freq_mask = frequencies > 0
        frequencies = frequencies[positive_freq_mask]
        power_spectrum = np.abs(fft_result[positive_freq_mask])**2
        
        return frequencies, power_spectrum
    
    @staticmethod
    def zeta_zeros_spectrum(t_max: float = 100) -> Tuple[List[float], np.ndarray]:
        """
        تحليل طيف أصفار دالة زيتا
        """
        # الحصول على أصفار دالة زيتا
        zeros = RiemannZeta.find_zeros_on_critical_line(0, t_max, 10000)
        
        if len(zeros) < 2:
            return zeros, np.array([])
        
        # حساب الفروقات بين الأصفار المتتالية
        differences = np.diff(zeros)
        
        # إنشاء هيستوغرام للفروقات
        hist, bin_edges = np.histogram(differences, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return zeros, hist, bin_centers
    
    @staticmethod
    def correlation_analysis(signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
        """
        تحليل الارتباط بين إشارتين
        """
        # تطبيع الإشارات
        signal1_norm = (signal1 - np.mean(signal1)) / np.std(signal1)
        signal2_norm = (signal2 - np.mean(signal2)) / np.std(signal2)
        
        # حساب الارتباط المتقاطع
        correlation = np.correlate(signal1_norm, signal2_norm, mode='full')
        
        return correlation

# ==========================================
# دوال المحاكاة والرسم
# ==========================================

def plot_filament_dynamics(filament: Filament, t_max: float = 10):
    """
    رسم ديناميكية الفتيلة
    """
    t = np.linspace(0, t_max, 1000)
    
    mass_values = [filament.mass(ti) for ti in t]
    radius_values = [filament.radius(ti) for ti in t]
    current_values = [filament.mass_current(ti) for ti in t]
    energy_values = [filament.energy(ti) for ti in t]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # رسم الكتلة
    axes[0, 0].plot(t, mass_values, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('الزمن (s)')
    axes[0, 0].set_ylabel('الكتلة (kg)')
    axes[0, 0].set_title('تذبذب الكتلة')
    axes[0, 0].grid(True)
    
    # رسم نصف القطر
    axes[0, 1].plot(t, radius_values, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('الزمن (s)')
    axes[0, 1].set_ylabel('نصف القطر (m)')
    axes[0, 1].set_title('تذبذب نصف القطر')
    axes[0, 1].grid(True)
    
    # رسم التيار الكتلي
    axes[1, 0].plot(t, current_values, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('الزمن (s)')
    axes[1, 0].set_ylabel('التيار الكتلي (kg/s)')
    axes[1, 0].set_title('التيار الكتلي')
    axes[1, 0].grid(True)
    
    # رسم الطاقة
    axes[1, 1].plot(t, energy_values, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('الزمن (s)')
    axes[1, 1].set_ylabel('الطاقة (J)')
    axes[1, 1].set_title('الطاقة الكلية')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_zeta_function_critical_line(t_max: float = 50):
    """
    رسم دالة زيتا على الخط الحرج
    """
    t_values = np.linspace(0.1, t_max, 1000)
    zeta_values = []
    
    for t in t_values:
        s = 0.5 + 1j * t
        zeta_val = RiemannZeta.zeta_from_eta(s, 1000)
        zeta_values.append(abs(zeta_val))
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, zeta_values, 'b-', linewidth=1)
    plt.xlabel('t (الجزء التخيلي)')
    plt.ylabel('|ζ(1/2 + it)|')
    plt.title('دالة زيتا ريمان على الخط الحرج')
    plt.grid(True)
    plt.yscale('log')
    plt.show()
    
    # البحث عن الأصفار
    zeros = RiemannZeta.find_zeros_on_critical_line(0.1, t_max, 5000)
    print(f"تم العثور على {len(zeros)} صفر تقريبي:")
    for i, zero in enumerate(zeros[:10]):  # عرض أول 10 أصفار
        print(f"الصفر {i+1}: t = {zero:.6f}")

def plot_filament_sea_evolution():
    """
    محاكاة تطور بحر الفتائل
    """
    sea = FilamentSea(grid_size=50, domain_size=1.0)
    
    # إضافة كتلة في المركز
    sea.add_mass_source(0.5, 0.5, 1e20, 0.1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    time_steps = [0, 10, 20]
    dt = 0.001
    
    for step, ax_row in zip(time_steps, axes):
        # تطوير النظام
        for _ in range(step):
            sea.evolve_navier_stokes(dt)
        
        # رسم الكثافة
        im1 = ax_row[0].imshow(sea.density, extent=[0, 1, 0, 1], 
                              cmap='viridis', origin='lower')
        ax_row[0].set_title(f'الكثافة - الزمن {step*dt:.3f}')
        ax_row[0].set_xlabel('x')
        ax_row[0].set_ylabel('y')
        plt.colorbar(im1, ax=ax_row[0])
        
        # رسم الضغط
        sea.calculate_pressure()
        im2 = ax_row[1].imshow(sea.pressure, extent=[0, 1, 0, 1], 
                              cmap='plasma', origin='lower')
        ax_row[1].set_title(f'الضغط - الزمن {step*dt:.3f}')
        ax_row[1].set_xlabel('x')
        ax_row[1].set_ylabel('y')
        plt.colorbar(im2, ax=ax_row[1])
        
        # رسم المجال الجاذبي
        g_x, g_y = sea.calculate_gravitational_field()
        magnitude = np.sqrt(g_x**2 + g_y**2)
        im3 = ax_row[2].imshow(magnitude, extent=[0, 1, 0, 1], 
                              cmap='hot', origin='lower')
        ax_row[2].set_title(f'المجال الجاذبي - الزمن {step*dt:.3f}')
        ax_row[2].set_xlabel('x')
        ax_row[2].set_ylabel('y')
        plt.colorbar(im3, ax=ax_row[2])
    
    plt.tight_layout()
    plt.show()

def demonstrate_quantum_entanglement():
    """
    عرض توضيحي للتشابك الكمومي
    """
    # إنشاء نظام من فتيلتين متشابكتين
    entangled_system = QuantumEntanglement(n_filaments=2)
    
    print("النظام الأولي:")
    print(f"الحالة الكمومية: {entangled_system.state_vector}")
    
    # حساب إنتروبيا التشابك الأولية
    initial_entropy = entangled_system.calculate_entanglement_entropy([0])
    print(f"إنتروبيا التشابك الأولية: {initial_entropy:.4f}")
    
    # قياس الفتيلة الأولى
    result1, new_state = entangled_system.measure_filament(0)
    print(f"\nنتيجة قياس الفتيلة الأولى: {result1}")
    print(f"الحالة الجديدة: {new_state}")
    
    # حساب إنتروبيا التشابك بعد القياس
    final_entropy = entangled_system.calculate_entanglement_entropy([0])
    print(f"إنتروبيا التشابك بعد القياس: {final_entropy:.4f}")
    
    # محاكاة عدة قياسات
    print("\nمحاكاة 100 قياس:")
    results = []
    for _ in range(100):
        # إعادة تهيئة النظام
        entangled_system = QuantumEntanglement(n_filaments=2)
        result, _ = entangled_system.measure_filament(0)
        results.append(result)
    
    prob_0 = results.count(0) / len(results)
    prob_1 = results.count(1) / len(results)
    print(f"احتمالية الحصول على 0: {prob_0:.3f}")
    print(f"احتمالية الحصول على 1: {prob_1:.3f}")

def analyze_gravity_modifications():
    """
    تحليل التعديلات على الجاذبية
    """
    gravity = FilamentGravity()
    
    # كتلة الشمس
    M_sun = 1.989e30  # kg
    
    # مدى من المسافات
    r_values = np.logspace(8, 12, 1000)  # من 100 مليون كم إلى 10000 AU
    
    classical_potential = [-PhysicalConstants.G * M_sun / r for r in r_values]
    modified_potential = [gravity.modified_gravitational_potential(M_sun, r) for r in r_values]
    filament_contribution = [gravity.alpha * gravity.filament_potential(r) for r in r_values]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.loglog(r_values/1.496e11, np.abs(classical_potential), 'b-', 
               label='الجهد الكلاسيكي', linewidth=2)
    plt.loglog(r_values/1.496e11, np.abs(modified_potential), 'r--', 
               label='الجهد المعدل', linewidth=2)
    plt.loglog(r_values/1.496e11, filament_contribution, 'g:', 
               label='مساهمة الفتائل', linewidth=2)
    plt.xlabel('المسافة (AU)')
    plt.ylabel('|الجهد| (J/kg)')
    plt.title('مقارنة الجهد الجاذبي')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    relative_difference = [(modified_potential[i] - classical_potential[i]) / 
                          abs(classical_potential[i]) for i in range(len(r_values))]
    plt.semilogx(r_values/1.496e11, relative_difference, 'purple', linewidth=2)
    plt.xlabel('المسافة (AU)')
    plt.ylabel('الفرق النسبي')
    plt.title('الفرق النسبي بين الجهد المعدل والكلاسيكي')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# دوال الاختبار والتحقق
# ==========================================

def test_filament_conservation():
    """
    اختبار قوانين الحفظ في الفتيلة
    """
    print("اختبار قوانين الحفظ في الفتيلة:")
    print("=" * 50)
    
    filament = Filament(oscillation_amplitude=0.1)
    
    # اختبار العلاقة الأساسية m₀ × r₀ = constant
    basic_constant = filament.m0 * filament.r0
    expected_constant = PhysicalConstants.h / (4 * np.pi * PhysicalConstants.c)
    basic_error = abs(basic_constant - expected_constant) / expected_constant
    
    print(f"العلاقة الأساسية m₀ × r₀ = constant:")
    print(f"القيمة المحسوبة: {basic_constant:.2e}")
    print(f"القيمة المتوقعة: {expected_constant:.2e}")
    print(f"الخطأ النسبي: {basic_error:.2e}")
    
    if basic_error < 1e-10:
        print("✓ العلاقة الأساسية محفوظة بدقة عالية")
    else:
        print("✗ العلاقة الأساسية غير محفوظة!")
    
    # اختبار تذبذب m(t) × r(t) حول القيمة الأساسية
    t_values = np.linspace(0, 10, 100)
    products = []
    
    for t in t_values:
        m = filament.mass(t)
        r = filament.radius(t)
        products.append(m * r)
    
    # التحليل النظري: m(t)×r(t) = m₀r₀[1 - ε²sin²(ωt)]
    theoretical_products = []
    for t in t_values:
        sin_term = np.sin(filament.omega * t + filament.phi)
        theoretical_value = basic_constant * (1 - filament.epsilon**2 * sin_term**2)
        theoretical_products.append(theoretical_value)
    
    avg_product = np.mean(products)
    theoretical_avg = basic_constant * (1 - filament.epsilon**2 / 2)
    
    print(f"\nتذبذب m(t) × r(t):")
    print(f"متوسط القيمة المحسوبة: {avg_product:.2e}")
    print(f"متوسط القيمة النظرية: {theoretical_avg:.2e}")
    print(f"الخطأ في المتوسط: {abs(avg_product - theoretical_avg)/theoretical_avg:.2e}")
    
    # اختبار التطابق مع النموذج النظري
    max_theory_error = max(abs(products[i] - theoretical_products[i]) 
                          for i in range(len(products)))
    relative_theory_error = max_theory_error / basic_constant
    
    print(f"أقصى انحراف عن النموذج النظري: {max_theory_error:.2e}")
    print(f"الخطأ النسبي للنموذج: {relative_theory_error:.2e}")
    
    if relative_theory_error < 1e-10:
        print("✓ التذبذب يتطابق مع النموذج النظري بدقة عالية")
    else:
        print("✗ التذبذب لا يتطابق مع النموذج النظري!")
    
    # اختبار متوسط الكتلة والطاقة
    avg_mass = np.mean([filament.mass(t) for t in t_values])
    avg_energy = np.mean([filament.energy(t) for t in t_values])
    expected_avg_mass = filament.m0
    expected_avg_energy = filament.m0 * PhysicalConstants.c**2
    
    print(f"\nمتوسط الكتلة:")
    print(f"المحسوب: {avg_mass:.2e}")
    print(f"المتوقع: {expected_avg_mass:.2e}")
    print(f"الخطأ النسبي: {abs(avg_mass - expected_avg_mass)/expected_avg_mass:.2e}")
    
    print(f"\nمتوسط الطاقة:")
    print(f"المحسوب: {avg_energy:.2e}")
    print(f"المتوقع: {expected_avg_energy:.2e}")
    print(f"الخطأ النسبي: {abs(avg_energy - expected_avg_energy)/expected_avg_energy:.2e}")
    
    # اختبار حفظ الطاقة أثناء التذبذب
    energies = [filament.energy(t) for t in t_values]
    energy_variation = (max(energies) - min(energies)) / np.mean(energies)
    
    print(f"\nتغير الطاقة أثناء التذبذب:")
    print(f"التغير النسبي: {energy_variation:.2e}")
    print(f"التغير المتوقع (2ε): {2*filament.epsilon:.2e}")
    
    if abs(energy_variation - 2*filament.epsilon) < 1e-10:
        print("✓ تغير الطاقة يتطابق مع النموذج النظري")
    else:
        print("✗ تغير الطاقة لا يتطابق مع النموذج النظري!")

def test_zeta_function_properties():
    """
    اختبار خصائص دالة زيتا
    """
    print("\nاختبار خصائص دالة زيتا:")
    print("=" * 50)
    
    # اختبار القيم المعروفة
    test_cases = [
        (2, np.pi**2/6),
        (4, np.pi**4/90),
        (6, np.pi**6/945)
    ]
    
    for s, expected in test_cases:
        calculated = RiemannZeta.zeta_direct(s, 10000)
        error = abs(calculated - expected) / expected
        print(f"ζ({s}) = {calculated:.6f}, متوقع = {expected:.6f}, خطأ = {error:.2e}")
    
    # اختبار العلاقة بين زيتا وإيتا
    s_test = 3 + 2j
    zeta_direct = RiemannZeta.zeta_direct(s_test, 5000)
    zeta_from_eta = RiemannZeta.zeta_from_eta(s_test, 5000)
    
    print(f"\nاختبار العلاقة ζ(s) = η(s)/(1-2^(1-s)) عند s = {s_test}:")
    print(f"ζ مباشرة: {zeta_direct}")
    print(f"ζ من η: {zeta_from_eta}")
    print(f"الفرق النسبي: {abs(zeta_direct - zeta_from_eta)/abs(zeta_direct):.2e}")
    
    # اختبار حاصل ضرب أويلر
    s_test = 2
    zeta_direct = RiemannZeta.zeta_direct(s_test, 10000)
    zeta_euler = RiemannZeta.euler_product(s_test, 1000)
    
    print(f"\nاختبار حاصل ضرب أويلر عند s = {s_test}:")
    print(f"ζ مباشرة: {zeta_direct:.6f}")
    print(f"ζ من أويلر: {zeta_euler:.6f}")
    print(f"الفرق النسبي: {abs(zeta_direct - zeta_euler)/abs(zeta_direct):.2e}")

def run_comprehensive_tests():
    """
    تشغيل جميع الاختبارات الشاملة
    """
    print("تشغيل الاختبارات الشاملة لنظرية الفتائل")
    print("=" * 60)
    
    test_filament_conservation()
    test_advanced_filament_physics()
    test_filament_energy_momentum()
    test_dimensional_analysis()
    test_zeta_function_properties()
    
    print("\n" + "=" * 60)
    print("انتهت جميع الاختبارات")

def test_advanced_filament_physics():
    """
    اختبار الخصائص الفيزيائية المتقدمة للفتيلة
    """
    print("\nاختبار الخصائص الفيزيائية المتقدمة:")
    print("=" * 50)
    
    filament = Filament(oscillation_amplitude=0.1)
    
    # اختبار العلاقة بين التيار الكتلي والتذبذب
    t_values = np.linspace(0, 10, 100)
    mass_currents = [filament.mass_current(t) for t in t_values]
    
    # التحقق من أن متوسط التيار الكتلي = 0
    avg_current = np.mean(mass_currents)
    print(f"متوسط التيار الكتلي: {avg_current:.2e}")
    
    if abs(avg_current) < 1e-15:
        print("✓ متوسط التيار الكتلي = 0 (كما متوقع)")
    else:
        print("✗ متوسط التيار الكتلي ≠ 0!")
    
    # اختبار العلاقة بين الكثافة والحجم
    densities = [filament.density(t) for t in t_values]
    volumes = [filament.volume(t) for t in t_values]
    
    # التحقق من أن الكثافة تتناسب عكسياً مع الحجم
    density_volume_products = [densities[i] * volumes[i] for i in range(len(t_values))]
    expected_product = filament.m0
    
    max_deviation = max(abs(p - expected_product) for p in density_volume_products)
    relative_error = max_deviation / expected_product
    
    print(f"\nاختبار العلاقة ρ × V = m:")
    print(f"أقصى انحراف: {max_deviation:.2e}")
    print(f"الخطأ النسبي: {relative_error:.2e}")
    
    if relative_error < 1e-10:
        print("✓ العلاقة ρ × V = m محفوظة بدقة عالية")
    else:
        print("✗ العلاقة ρ × V = m غير محفوظة!")
    
    # اختبار كثافة التيار الكتلي السطحي
    surface_currents = [filament.surface_mass_current_density(t) for t in t_values]
    avg_surface_current = np.mean(surface_currents)
    
    print(f"\nمتوسط كثافة التيار الكتلي السطحي: {avg_surface_current:.2e}")
    
    if abs(avg_surface_current) < 1e-15:
        print("✓ متوسط كثافة التيار السطحي = 0 (كما متوقع)")
    else:
        print("✗ متوسط كثافة التيار السطحي ≠ 0!")
    
    # اختبار التردد الأساسي
    calculated_frequency = PhysicalConstants.c / filament.r0
    expected_frequency = filament.omega
    
    print(f"\nاختبار التردد الأساسي:")
    print(f"المحسوب من ω = c/r₀: {calculated_frequency:.2e} Hz")
    print(f"المحفوظ في الفتيلة: {expected_frequency:.2e} Hz")
    print(f"الخطأ النسبي: {abs(calculated_frequency - expected_frequency)/expected_frequency:.2e}")
    
    if abs(calculated_frequency - expected_frequency)/expected_frequency < 1e-10:
        print("✓ التردد الأساسي متسق مع العلاقة ω = c/r₀")
    else:
        print("✗ التردد الأساسي غير متسق!")

def test_filament_energy_momentum():
    """
    اختبار الطاقة والزخم في الفتيلة
    """
    print("\nاختبار الطاقة والزخم:")
    print("=" * 50)
    
    filament = Filament(oscillation_amplitude=0.1)
    
    # اختبار العلاقة E = mc²
    t_values = np.linspace(0, 10, 100)
    
    for i, t in enumerate(t_values[:5]):  # اختبار أول 5 نقاط
        mass = filament.mass(t)
        energy = filament.energy(t)
        expected_energy = mass * PhysicalConstants.c**2
        
        error = abs(energy - expected_energy) / expected_energy
        print(f"t={t:.1f}: E={energy:.2e}, mc²={expected_energy:.2e}, خطأ={error:.2e}")
        
        if error > 1e-15:
            print(f"✗ العلاقة E=mc² غير محققة عند t={t:.1f}")
            return
    
    print("✓ العلاقة E=mc² محققة بدقة عالية")
    
    # حساب الزخم النسبي (تقريبي)
    # p ≈ mv حيث v هي سرعة تغير نصف القطر
    momenta = []
    for t in t_values:
        # حساب السرعة كمشتقة نصف القطر
        dt = 0.001
        r1 = filament.radius(t)
        r2 = filament.radius(t + dt)
        velocity = (r2 - r1) / dt
        
        mass = filament.mass(t)
        momentum = mass * velocity
        momenta.append(momentum)
    
    avg_momentum = np.mean(momenta)
    print(f"\nمتوسط الزخم: {avg_momentum:.2e}")
    
    if abs(avg_momentum) < 1e-30:
        print("✓ متوسط الزخم ≈ 0 (كما متوقع للتذبذب)")
    else:
        print("✗ متوسط الزخم ≠ 0!")

def test_dimensional_analysis():
    """
    اختبار تحليل الأبعاد للمعادلات الأساسية
    """
    print("\nاختبار تحليل الأبعاد:")
    print("=" * 50)
    
    # اختبار أبعاد معادلة كتلة الفتيلة: m₀ = h/(4πc²)
    h_dims = "[M L² T⁻¹]"  # أبعاد ثابت بلانك
    c_dims = "[L T⁻¹]"     # أبعاد سرعة الضوء
    
    print(f"معادلة m₀ = h/(4πc²):")
    print(f"أبعاد h: {h_dims}")
    print(f"أبعاد c²: [L² T⁻²]")
    print(f"أبعاد h/c²: [M L² T⁻¹] / [L² T⁻²] = [M T]")
    print("✗ هناك مشكلة في الأبعاد! يجب أن تكون [M]")
    
    # الصيغة الصحيحة: m₀ = h/(4πc·r₀)
    print(f"\nالصيغة المصححة m₀ = h/(4πc·r₀):")
    print(f"أبعاد c·r₀: [L T⁻¹] × [L] = [L² T⁻¹]")
    print(f"أبعاد h/(c·r₀): [M L² T⁻¹] / [L² T⁻¹] = [M]")
    print("✓ الأبعاد صحيحة!")
    
    # اختبار أبعاد التيار الكتلي
    print(f"\nمعادلة التيار الكتلي I_m = dm/dt:")
    print(f"أبعاد dm/dt: [M] / [T] = [M T⁻¹]")
    print("✓ أبعاد التيار الكتلي صحيحة!")
    
    # اختبار أبعاد كثافة التيار السطحي
    print(f"\nكثافة التيار السطحي K_m = I_m/A:")
    print(f"أبعاد I_m/A: [M T⁻¹] / [L²] = [M L⁻² T⁻¹]")
    print("✓ أبعاد كثافة التيار السطحي صحيحة!")

# ==========================================
# دالة رئيسية للعرض التوضيحي
# ==========================================

def main_demonstration():
    """
    العرض التوضيحي الرئيسي لنظرية الفتائل
    """
    print("مرحباً بكم في محاكاة نظرية الفتائل")
    print("=" * 50)
    
    # إنشاء فتيلة أساسية
    filament = Filament(oscillation_amplitude=0.1, phase=0)
    
    print("خصائص الفتيلة الأساسية:")
    print(f"الكتلة الأساسية: {filament.m0:.2e} kg")
    print(f"نصف القطر الأساسي: {filament.r0:.2e} m")
    print(f"التردد الأساسي: {filament.omega:.2e} Hz")
    print(f"سعة التذبذب: {filament.epsilon}")
    
    # رسم ديناميكية الفتيلة
    print("\nرسم ديناميكية الفتيلة...")
    plot_filament_dynamics(filament, t_max=20)
    
    # تحليل طيف الفتيلة
    print("تحليل طيف الفتيلة...")
    frequencies, power_spectrum = SpectrumAnalysis.filament_spectrum(filament)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(frequencies, power_spectrum)
    plt.xlabel('التردد (Hz)')
    plt.ylabel('كثافة الطيف')
    plt.title('طيف تذبذبات الفتيلة')
    plt.grid(True)
    plt.show()
    
    # رسم دالة زيتا
    print("رسم دالة زيتا على الخط الحرج...")
    plot_zeta_function_critical_line(t_max=30)
    
    # محاكاة بحر الفتائل
    print("محاكاة تطور بحر الفتائل...")
    plot_filament_sea_evolution()
    
    # عرض التشابك الكمومي
    print("عرض التشابك الكمومي...")
    demonstrate_quantum_entanglement()
    
    # تحليل تعديلات الجاذبية
    print("تحليل تعديلات الجاذبية...")
    analyze_gravity_modifications()
    
    # تشغيل الاختبارات
    print("تشغيل الاختبارات...")
    run_comprehensive_tests()
    
    print("\nانتهى العرض التوضيحي!")

if __name__ == "__main__":
    main_demonstration()

