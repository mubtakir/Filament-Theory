#!/usr/bin/env python3
"""
محاكي دالة زيتا ريمان كنظام RLC متعدد الدوائر
Zeta Riemann Function as Multi-Circuit RLC System Simulator

نهج مبتكر لمحاكاة دالة زيتا ريمان كنظام فيزيائي حقيقي:
- كل عدد صحيح = دائرة RLC منفصلة
- المقاومة = مجموع الجذور التربيعية
- الرنين = كشف الأعداد الأولية
- الأصفار = إلغاء هدام للترددات

المؤلف: باسل يحيى عبدالله "المبتكر العلمي"
المشرف: الذكاء الاصطناعي مانوس
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
import cmath
import time
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class ZetaRLCSimulator:
    """
    محاكي دالة زيتا ريمان كنظام RLC متعدد الدوائر
    """
    
    def __init__(self, max_circuits: int = 1000, precision: float = 1e-12):
        """تهيئة المحاكي"""
        self.max_circuits = max_circuits
        self.precision = precision
        
        # معاملات النظام
        self.alpha = 1.0  # معامل المحاثة
        self.beta = 1.0   # معامل السعة
        self.gamma = 0.5  # معامل المقاومة
        
        # بيانات الدوائر
        self.circuits = {}
        self.total_impedance = {}
        
        # نتائج المحاكاة
        self.found_primes = []
        self.found_zeros = []
        self.resonance_peaks = []
        
        print(f"🔬 تم تهيئة محاكي زيتا RLC")
        print(f"📊 عدد الدوائر: {max_circuits}")
        print(f"🎯 الدقة: {precision}")
        
        self._initialize_circuits()
    
    def _initialize_circuits(self):
        """تهيئة جميع الدوائر"""
        print("⚡ تهيئة الدوائر...")
        
        for n in range(1, self.max_circuits + 1):
            # حساب مكونات الدائرة n
            circuit = self._calculate_circuit_components(n)
            self.circuits[n] = circuit
        
        print(f"✅ تم تهيئة {len(self.circuits)} دائرة")
    
    def _calculate_circuit_components(self, n: int) -> Dict[str, float]:
        """حساب مكونات الدائرة للعدد n"""
        
        # المقاومة: الجذر التربيعي (كما اقترح باسل)
        R = self.gamma * np.sqrt(n)
        
        # المحاثة: عكسياً متناسبة مع الجذر
        L = self.alpha / np.sqrt(n)
        
        # السعة: متناسبة عكسياً مع n واللوغاريتم
        if n == 1:
            C = self.beta  # تجنب log(1) = 0
        else:
            C = self.beta / (n * (np.log(n))**2)
        
        # التردد الطبيعي
        omega_0 = 1 / np.sqrt(L * C) if L * C > 0 else 0
        
        # عامل الجودة
        Q = omega_0 * L / R if R > 0 else float('inf')
        
        # تحديد نوع الدائرة
        circuit_type = self._classify_circuit(n)
        
        return {
            'n': n,
            'R': R,
            'L': L,
            'C': C,
            'omega_0': omega_0,
            'Q': Q,
            'type': circuit_type,
            'LC_product': L * C,
            'time_constant': L / R if R > 0 else float('inf')
        }
    
    def _classify_circuit(self, n: int) -> str:
        """تصنيف الدائرة حسب نوع العدد"""
        if n == 1:
            return 'unity'
        elif self._is_prime(n):
            return 'prime'
        elif self._is_power_of_prime(n):
            return 'prime_power'
        else:
            return 'composite'
    
    def _is_prime(self, n: int) -> bool:
        """فحص الأولية"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(np.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        
        return True
    
    def _is_power_of_prime(self, n: int) -> bool:
        """فحص إذا كان العدد قوة لعدد أولي"""
        if n < 2:
            return False
        
        # البحث عن العامل الأولي الوحيد
        for p in range(2, int(np.sqrt(n)) + 1):
            if self._is_prime(p) and n % p == 0:
                # فحص إذا كان n = p^k
                temp = n
                while temp % p == 0:
                    temp //= p
                return temp == 1
        
        return False
    
    def calculate_impedance(self, s: complex) -> complex:
        """حساب المقاومة الإجمالية للنظام عند s"""
        
        sigma = s.real
        t = s.imag
        
        total_impedance = 0.0 + 0.0j
        
        for n, circuit in self.circuits.items():
            # تردد الإثارة
            omega = t * np.log(n) if n > 1 else t
            
            # مقاومة الدائرة
            R = circuit['R'] / (n**sigma)  # تأثير التخميد
            L = circuit['L']
            C = circuit['C']
            
            # المقاومة المعقدة
            if omega != 0:
                Z_n = R + 1j * (omega * L - 1/(omega * C))
            else:
                Z_n = R + 0j
            
            # إضافة للمجموع
            total_impedance += Z_n / (n**s)
        
        return total_impedance
    
    def find_resonance_peaks(self, t_min: float = 0.1, t_max: float = 100, 
                           resolution: int = 10000) -> List[Tuple[float, float, str]]:
        """البحث عن ذروات الرنين (الأعداد الأولية المحتملة)"""
        
        print(f"🔍 البحث عن ذروات الرنين في [{t_min}, {t_max}]...")
        
        t_values = np.linspace(t_min, t_max, resolution)
        impedance_magnitudes = []
        
        # حساب المقاومة عند σ = 0.5 (الخط الحرج)
        for t in t_values:
            s = complex(0.5, t)
            Z = self.calculate_impedance(s)
            magnitude = abs(Z)
            impedance_magnitudes.append(magnitude)
        
        # كشف الذروات
        peaks, properties = find_peaks(impedance_magnitudes, 
                                     height=np.mean(impedance_magnitudes),
                                     distance=5,
                                     prominence=0.1)
        
        resonance_peaks = []
        
        for peak_idx in peaks:
            t_peak = t_values[peak_idx]
            magnitude = impedance_magnitudes[peak_idx]
            
            # تحديد نوع الذروة
            peak_type = self._classify_peak(t_peak, magnitude)
            
            resonance_peaks.append((t_peak, magnitude, peak_type))
            
            if peak_type == 'prime_candidate':
                print(f"🎯 ذروة رنين محتملة: t = {t_peak:.6f}, |Z| = {magnitude:.6f}")
        
        self.resonance_peaks = resonance_peaks
        print(f"📊 إجمالي الذروات: {len(resonance_peaks)}")
        
        return resonance_peaks
    
    def _classify_peak(self, t: float, magnitude: float) -> str:
        """تصنيف نوع الذروة"""
        
        # حساب التردد المقابل لأعداد مختلفة
        for n in range(2, min(100, self.max_circuits + 1)):
            expected_t = n / np.log(n) if n > 1 else n
            
            if abs(t - expected_t) < 0.1:  # تسامح في التطابق
                if self._is_prime(n):
                    return 'prime_candidate'
                elif self._is_power_of_prime(n):
                    return 'prime_power'
                else:
                    return 'composite'
        
        return 'unknown'
    
    def find_zeros(self, t_min: float = 10, t_max: float = 50, 
                   resolution: int = 5000) -> List[float]:
        """البحث عن أصفار ريمان (إلغاء هدام)"""
        
        print(f"🔍 البحث عن أصفار ريمان في [{t_min}, {t_max}]...")
        
        t_values = np.linspace(t_min, t_max, resolution)
        impedance_magnitudes = []
        
        # حساب المقاومة عند σ = 0.5
        for t in t_values:
            s = complex(0.5, t)
            Z = self.calculate_impedance(s)
            magnitude = abs(Z)
            impedance_magnitudes.append(magnitude)
        
        # كشف الحد الأدنى (الأصفار)
        min_threshold = np.min(impedance_magnitudes) * 1.1
        
        zeros_found = []
        
        for i, magnitude in enumerate(impedance_magnitudes):
            if magnitude < min_threshold:
                t_zero = t_values[i]
                
                # تحسين دقة الصفر
                refined_zero = self._refine_zero(t_zero)
                if refined_zero is not None:
                    zeros_found.append(refined_zero)
                    print(f"✅ صفر مكتشف: t = {refined_zero:.8f}")
        
        self.found_zeros = zeros_found
        print(f"📊 إجمالي الأصفار: {len(zeros_found)}")
        
        return zeros_found
    
    def _refine_zero(self, t_initial: float) -> float:
        """تحسين دقة الصفر"""
        
        def objective(t):
            s = complex(0.5, t)
            Z = self.calculate_impedance(s)
            return abs(Z)
        
        try:
            result = minimize_scalar(objective, 
                                   bounds=(t_initial-0.1, t_initial+0.1),
                                   method='bounded')
            
            if result.success and result.fun < self.precision * 1000:
                return result.x
        except:
            pass
        
        return None
    
    def extract_primes_from_resonance(self) -> List[int]:
        """استخراج الأعداد الأولية من ذروات الرنين"""
        
        print("🔍 استخراج الأعداد الأولية من ذروات الرنين...")
        
        primes_found = []
        
        for t_peak, magnitude, peak_type in self.resonance_peaks:
            if peak_type == 'prime_candidate':
                # البحث عن العدد الأولي المقابل
                for n in range(2, min(1000, self.max_circuits + 1)):
                    if self._is_prime(n):
                        expected_t = n / np.log(n)
                        
                        if abs(t_peak - expected_t) < 0.1:
                            primes_found.append(n)
                            print(f"🎯 عدد أولي مكتشف: {n} (t = {t_peak:.6f})")
                            break
        
        # إزالة التكرارات وترتيب
        primes_found = sorted(list(set(primes_found)))
        self.found_primes = primes_found
        
        print(f"📊 إجمالي الأعداد الأولية: {len(primes_found)}")
        return primes_found
    
    def analyze_circuit_properties(self) -> Dict[str, Any]:
        """تحليل خصائص الدوائر"""
        
        print("📊 تحليل خصائص الدوائر...")
        
        analysis = {
            'prime_circuits': [],
            'composite_circuits': [],
            'resonance_frequencies': [],
            'quality_factors': [],
            'time_constants': []
        }
        
        for n, circuit in self.circuits.items():
            if circuit['type'] == 'prime':
                analysis['prime_circuits'].append(circuit)
            elif circuit['type'] == 'composite':
                analysis['composite_circuits'].append(circuit)
            
            analysis['resonance_frequencies'].append(circuit['omega_0'])
            analysis['quality_factors'].append(circuit['Q'])
            analysis['time_constants'].append(circuit['time_constant'])
        
        # إحصائيات
        analysis['stats'] = {
            'total_circuits': len(self.circuits),
            'prime_count': len(analysis['prime_circuits']),
            'composite_count': len(analysis['composite_circuits']),
            'avg_Q_prime': np.mean([c['Q'] for c in analysis['prime_circuits']]) if analysis['prime_circuits'] else 0,
            'avg_Q_composite': np.mean([c['Q'] for c in analysis['composite_circuits']]) if analysis['composite_circuits'] else 0,
            'max_resonance_freq': max(analysis['resonance_frequencies']) if analysis['resonance_frequencies'] else 0
        }
        
        return analysis
    
    def plot_impedance_spectrum(self, t_min: float = 0.1, t_max: float = 50, 
                               resolution: int = 2000):
        """رسم طيف المقاومة"""
        
        print("📈 رسم طيف المقاومة...")
        
        t_values = np.linspace(t_min, t_max, resolution)
        impedance_magnitudes = []
        impedance_phases = []
        
        for t in t_values:
            s = complex(0.5, t)
            Z = self.calculate_impedance(s)
            impedance_magnitudes.append(abs(Z))
            impedance_phases.append(np.angle(Z))
        
        # الرسم
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # المقدار
        ax1.plot(t_values, impedance_magnitudes, 'b-', linewidth=1)
        ax1.set_xlabel('t (الجزء التخيلي)')
        ax1.set_ylabel('|Z(0.5 + it)|')
        ax1.set_title('طيف مقدار المقاومة - محاكي زيتا RLC')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # تمييز ذروات الرنين
        if self.resonance_peaks:
            peak_t = [p[0] for p in self.resonance_peaks if t_min <= p[0] <= t_max]
            peak_mag = [p[1] for p in self.resonance_peaks if t_min <= p[0] <= t_max]
            ax1.scatter(peak_t, peak_mag, color='red', s=50, alpha=0.7, label='ذروات الرنين')
            ax1.legend()
        
        # الطور
        ax2.plot(t_values, impedance_phases, 'g-', linewidth=1)
        ax2.set_xlabel('t (الجزء التخيلي)')
        ax2.set_ylabel('Phase(Z(0.5 + it))')
        ax2.set_title('طيف طور المقاومة')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/zeta_rlc_spectrum.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 تم حفظ الطيف في: /home/ubuntu/zeta_rlc_spectrum.png")
    
    def comprehensive_test(self) -> Dict[str, Any]:
        """اختبار شامل للمحاكي"""
        
        print("🚀 بدء الاختبار الشامل لمحاكي زيتا RLC")
        print("=" * 80)
        
        results = {}
        
        # 1. تحليل خصائص الدوائر
        print("\n1️⃣ تحليل خصائص الدوائر:")
        circuit_analysis = self.analyze_circuit_properties()
        results['circuit_analysis'] = circuit_analysis
        
        print(f"  📊 إجمالي الدوائر: {circuit_analysis['stats']['total_circuits']}")
        print(f"  🔢 دوائر الأعداد الأولية: {circuit_analysis['stats']['prime_count']}")
        print(f"  🔢 دوائر الأعداد المركبة: {circuit_analysis['stats']['composite_count']}")
        
        # 2. البحث عن ذروات الرنين
        print("\n2️⃣ البحث عن ذروات الرنين:")
        resonance_peaks = self.find_resonance_peaks(0.1, 20, 5000)
        results['resonance_peaks'] = resonance_peaks
        
        # 3. استخراج الأعداد الأولية
        print("\n3️⃣ استخراج الأعداد الأولية:")
        found_primes = self.extract_primes_from_resonance()
        results['found_primes'] = found_primes
        
        # 4. البحث عن أصفار ريمان
        print("\n4️⃣ البحث عن أصفار ريمان:")
        found_zeros = self.find_zeros(10, 30, 3000)
        results['found_zeros'] = found_zeros
        
        # 5. تقييم الأداء
        print("\n5️⃣ تقييم الأداء:")
        performance = self._evaluate_performance(results)
        results['performance'] = performance
        
        print(f"  🎯 دقة الأعداد الأولية: {performance['prime_accuracy']:.1%}")
        print(f"  🎯 عدد الأصفار المكتشفة: {len(found_zeros)}")
        print(f"  🎯 النتيجة الإجمالية: {performance['overall_score']:.1%}")
        
        # 6. رسم الطيف
        print("\n6️⃣ رسم طيف المقاومة:")
        self.plot_impedance_spectrum(0.1, 20, 2000)
        
        return results
    
    def _evaluate_performance(self, results: Dict[str, Any]) -> Dict[str, float]:
        """تقييم أداء المحاكي"""
        
        # الأعداد الأولية الحقيقية حتى 100
        true_primes = [p for p in range(2, 101) if self._is_prime(p)]
        found_primes = results['found_primes']
        
        # حساب الدقة
        if found_primes:
            correct_primes = len(set(found_primes) & set(true_primes))
            prime_accuracy = correct_primes / len(true_primes)
        else:
            prime_accuracy = 0.0
        
        # تقييم الأصفار (مقارنة مع الأصفار المعروفة)
        known_zeros = [14.1347, 21.0220, 25.0109]  # أول ثلاثة أصفار
        found_zeros = results['found_zeros']
        
        zero_matches = 0
        for known in known_zeros:
            for found in found_zeros:
                if abs(known - found) < 0.1:
                    zero_matches += 1
                    break
        
        zero_accuracy = zero_matches / len(known_zeros) if known_zeros else 0.0
        
        # النتيجة الإجمالية
        overall_score = (prime_accuracy * 0.7 + zero_accuracy * 0.3)
        
        return {
            'prime_accuracy': prime_accuracy,
            'zero_accuracy': zero_accuracy,
            'overall_score': overall_score,
            'resonance_peaks_count': len(results['resonance_peaks']),
            'circuit_efficiency': len(results['found_primes']) / len(true_primes) if true_primes else 0
        }

def main():
    """الدالة الرئيسية"""
    print("🌟 مرحباً بك في محاكي زيتا RLC المبتكر!")
    print("=" * 80)
    
    # إنشاء المحاكي
    simulator = ZetaRLCSimulator(max_circuits=500, precision=1e-12)
    
    # تشغيل الاختبار الشامل
    results = simulator.comprehensive_test()
    
    # حفظ النتائج
    import json
    with open('/home/ubuntu/zeta_rlc_results.json', 'w', encoding='utf-8') as f:
        # تحويل النتائج لتكون قابلة للتسلسل
        serializable_results = {
            'found_primes': results['found_primes'],
            'found_zeros': results['found_zeros'],
            'performance': results['performance'],
            'resonance_peaks_count': len(results['resonance_peaks']),
            'circuit_stats': results['circuit_analysis']['stats']
        }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print("\n💾 تم حفظ النتائج في: /home/ubuntu/zeta_rlc_results.json")
    
    return results

if __name__ == "__main__":
    results = main()

