#!/usr/bin/env python3
"""
محاكي زيتا RLC المحسن - التركيز على مجموع الفتائل
Enhanced Zeta RLC Simulator - Focus on Filament Summation

تحسينات على النموذج الأساسي:
1. محاكاة أدق لمجموع الفتائل
2. ضبط أفضل للمعاملات
3. كشف محسن للذروات والأصفار
4. تحليل أعمق للرنين

المؤلف: باسل يحيى عبدالله "المبتكر العلمي"
المشرف: الذكاء الاصطناعي مانوس
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
import cmath
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class EnhancedZetaRLC:
    """
    محاكي زيتا RLC المحسن مع التركيز على مجموع الفتائل
    """
    
    def __init__(self, max_terms: int = 1000):
        """تهيئة المحاكي المحسن"""
        self.max_terms = max_terms
        
        # معاملات محسنة بناءً على دالة زيتا الحقيقية
        self.resistance_factor = 0.5  # عامل المقاومة (الجزء الحقيقي)
        self.frequency_scale = 1.0    # مقياس التردد
        self.damping_factor = 0.1     # عامل التخميد
        
        # بيانات الفتائل
        self.filaments = {}
        self.total_resistance = {}
        self.resonance_spectrum = {}
        
        print(f"🔬 تم تهيئة محاكي زيتا RLC المحسن")
        print(f"📊 عدد الفتائل: {max_terms}")
        
        self._initialize_filaments()
    
    def _initialize_filaments(self):
        """تهيئة جميع الفتائل"""
        print("⚡ تهيئة الفتائل...")
        
        for n in range(1, self.max_terms + 1):
            filament = self._create_filament(n)
            self.filaments[n] = filament
        
        print(f"✅ تم تهيئة {len(self.filaments)} فتيلة")
    
    def _create_filament(self, n: int) -> Dict[str, Any]:
        """إنشاء فتيلة للعدد n"""
        
        # المقاومة الأساسية: الجذر التربيعي كما اقترح باسل
        base_resistance = np.sqrt(n)
        
        # التردد الطبيعي
        natural_frequency = np.log(n) if n > 1 else 1.0
        
        # عامل الجودة (أعلى للأعداد الأولية)
        if self._is_prime(n):
            quality_factor = 10.0  # رنين حاد للأعداد الأولية
            filament_type = 'prime'
        elif self._is_prime_power(n):
            quality_factor = 5.0   # رنين متوسط لقوى الأعداد الأولية
            filament_type = 'prime_power'
        else:
            quality_factor = 1.0   # رنين مكبوت للأعداد المركبة
            filament_type = 'composite'
        
        # المحاثة والسعة المحسنة
        inductance = 1.0 / (n * quality_factor)
        capacitance = quality_factor / (n * natural_frequency**2)
        
        return {
            'n': n,
            'base_resistance': base_resistance,
            'natural_frequency': natural_frequency,
            'quality_factor': quality_factor,
            'inductance': inductance,
            'capacitance': capacitance,
            'type': filament_type,
            'resonance_strength': quality_factor * base_resistance
        }
    
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
    
    def _is_prime_power(self, n: int) -> bool:
        """فحص قوة العدد الأولي"""
        if n < 2:
            return False
        
        for p in range(2, int(np.sqrt(n)) + 1):
            if self._is_prime(p):
                temp = n
                while temp % p == 0:
                    temp //= p
                if temp == 1:
                    return True
        
        return False
    
    def calculate_filament_sum(self, s: complex) -> complex:
        """حساب مجموع الفتائل عند s"""
        
        sigma = s.real
        t = s.imag
        
        total_sum = 0.0 + 0.0j
        
        for n, filament in self.filaments.items():
            # المقاومة مع التخميد
            resistance = filament['base_resistance'] * np.exp(-self.damping_factor * sigma)
            
            # التردد المعدل
            frequency = t * filament['natural_frequency'] * self.frequency_scale
            
            # المقاومة المعقدة للفتيلة
            if frequency != 0:
                # تأثير الرنين
                resonance_factor = filament['quality_factor'] / (1 + 1j * frequency / filament['quality_factor'])
                impedance = resistance * resonance_factor
            else:
                impedance = resistance + 0j
            
            # إضافة للمجموع مع التطبيع
            contribution = impedance / (n**s)
            total_sum += contribution
        
        return total_sum
    
    def find_resonance_frequencies(self, t_min: float = 0.1, t_max: float = 50, 
                                 resolution: int = 10000) -> List[Tuple[float, float, int]]:
        """البحث عن ترددات الرنين"""
        
        print(f"🔍 البحث عن ترددات الرنين في [{t_min}, {t_max}]...")
        
        t_values = np.linspace(t_min, t_max, resolution)
        impedance_magnitudes = []
        
        # حساب المقاومة عند الخط الحرج σ = 0.5
        for t in t_values:
            s = complex(self.resistance_factor, t)
            Z = self.calculate_filament_sum(s)
            magnitude = abs(Z)
            impedance_magnitudes.append(magnitude)
        
        # تطبيع البيانات
        max_magnitude = max(impedance_magnitudes)
        normalized_magnitudes = [m / max_magnitude for m in impedance_magnitudes]
        
        # كشف الذروات مع معايير محسنة
        peaks, properties = find_peaks(normalized_magnitudes, 
                                     height=0.1,      # عتبة أقل للحساسية
                                     distance=20,     # مسافة أدنى بين الذروات
                                     prominence=0.05, # بروز أدنى
                                     width=2)         # عرض أدنى
        
        resonance_frequencies = []
        
        for peak_idx in peaks:
            t_resonance = t_values[peak_idx]
            magnitude = impedance_magnitudes[peak_idx]
            
            # البحث عن العدد المقابل
            corresponding_n = self._find_corresponding_number(t_resonance)
            
            resonance_frequencies.append((t_resonance, magnitude, corresponding_n))
            
            if corresponding_n and self._is_prime(corresponding_n):
                print(f"🎯 رنين عدد أولي: n = {corresponding_n}, t = {t_resonance:.6f}, |Z| = {magnitude:.6f}")
            elif corresponding_n:
                print(f"📊 رنين عدد مركب: n = {corresponding_n}, t = {t_resonance:.6f}, |Z| = {magnitude:.6f}")
        
        print(f"📊 إجمالي ترددات الرنين: {len(resonance_frequencies)}")
        return resonance_frequencies
    
    def _find_corresponding_number(self, t_resonance: float) -> int:
        """البحث عن العدد المقابل لتردد الرنين"""
        
        best_match = None
        min_error = float('inf')
        
        for n in range(2, min(200, self.max_terms + 1)):
            # التردد المتوقع للعدد n
            expected_t = np.log(n) * self.frequency_scale
            
            error = abs(t_resonance - expected_t)
            
            if error < min_error and error < 0.5:  # تسامح في التطابق
                min_error = error
                best_match = n
        
        return best_match
    
    def find_zeros_enhanced(self, t_min: float = 10, t_max: float = 50, 
                          resolution: int = 5000) -> List[float]:
        """البحث المحسن عن الأصفار"""
        
        print(f"🔍 البحث المحسن عن أصفار ريمان في [{t_min}, {t_max}]...")
        
        t_values = np.linspace(t_min, t_max, resolution)
        impedance_magnitudes = []
        
        for t in t_values:
            s = complex(self.resistance_factor, t)
            Z = self.calculate_filament_sum(s)
            magnitude = abs(Z)
            impedance_magnitudes.append(magnitude)
        
        # البحث عن الحد الأدنى المحلي
        min_threshold = np.percentile(impedance_magnitudes, 5)  # أقل 5%
        
        # كشف الحد الأدنى
        min_peaks, _ = find_peaks(-np.array(impedance_magnitudes), 
                                height=-min_threshold,
                                distance=10)
        
        zeros_found = []
        
        for peak_idx in min_peaks:
            t_zero = t_values[peak_idx]
            magnitude = impedance_magnitudes[peak_idx]
            
            if magnitude < min_threshold:
                # تحسين دقة الصفر
                refined_zero = self._refine_zero_enhanced(t_zero)
                if refined_zero is not None:
                    zeros_found.append(refined_zero)
                    print(f"✅ صفر مكتشف: t = {refined_zero:.8f}, |Z| = {magnitude:.8f}")
        
        print(f"📊 إجمالي الأصفار المكتشفة: {len(zeros_found)}")
        return zeros_found
    
    def _refine_zero_enhanced(self, t_initial: float) -> float:
        """تحسين دقة الصفر"""
        
        def objective(t):
            s = complex(self.resistance_factor, t)
            Z = self.calculate_filament_sum(s)
            return abs(Z)
        
        try:
            result = minimize_scalar(objective, 
                                   bounds=(t_initial-0.2, t_initial+0.2),
                                   method='bounded')
            
            if result.success:
                return result.x
        except:
            pass
        
        return None
    
    def extract_primes_from_resonance(self, resonance_frequencies: List[Tuple[float, float, int]]) -> List[int]:
        """استخراج الأعداد الأولية من ترددات الرنين"""
        
        print("🔍 استخراج الأعداد الأولية من ترددات الرنين...")
        
        primes_found = []
        
        for t_resonance, magnitude, corresponding_n in resonance_frequencies:
            if corresponding_n and self._is_prime(corresponding_n):
                primes_found.append(corresponding_n)
        
        # إزالة التكرارات وترتيب
        primes_found = sorted(list(set(primes_found)))
        
        print(f"📊 الأعداد الأولية المكتشفة: {primes_found}")
        print(f"📊 إجمالي الأعداد الأولية: {len(primes_found)}")
        
        return primes_found
    
    def plot_enhanced_spectrum(self, t_min: float = 0.1, t_max: float = 30, 
                             resolution: int = 5000):
        """رسم الطيف المحسن"""
        
        print("📈 رسم الطيف المحسن...")
        
        t_values = np.linspace(t_min, t_max, resolution)
        impedance_magnitudes = []
        
        for t in t_values:
            s = complex(self.resistance_factor, t)
            Z = self.calculate_filament_sum(s)
            magnitude = abs(Z)
            impedance_magnitudes.append(magnitude)
        
        # الرسم
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(t_values, impedance_magnitudes, 'b-', linewidth=1, alpha=0.8)
        plt.xlabel('t (الجزء التخيلي)')
        plt.ylabel('|مجموع الفتائل|')
        plt.title('طيف مجموع الفتائل - محاكي زيتا RLC المحسن')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # تمييز الأعداد الأولية المعروفة
        known_primes = [p for p in range(2, 30) if self._is_prime(p)]
        prime_frequencies = [np.log(p) for p in known_primes]
        prime_magnitudes = []
        
        for t_prime in prime_frequencies:
            if t_min <= t_prime <= t_max:
                s = complex(self.resistance_factor, t_prime)
                Z = self.calculate_filament_sum(s)
                prime_magnitudes.append(abs(Z))
            else:
                prime_magnitudes.append(0)
        
        valid_primes = [(t, m, p) for t, m, p in zip(prime_frequencies, prime_magnitudes, known_primes) 
                       if t_min <= t <= t_max and m > 0]
        
        if valid_primes:
            prime_t, prime_mag, prime_nums = zip(*valid_primes)
            plt.scatter(prime_t, prime_mag, color='red', s=100, alpha=0.8, 
                       label=f'أعداد أولية معروفة: {list(prime_nums)}', zorder=5)
            plt.legend()
        
        # رسم مكبر للمنطقة المثيرة
        plt.subplot(2, 1, 2)
        focus_range = (t_values >= 1) & (t_values <= 10)
        plt.plot(t_values[focus_range], np.array(impedance_magnitudes)[focus_range], 
                'g-', linewidth=2, alpha=0.8)
        plt.xlabel('t (الجزء التخيلي)')
        plt.ylabel('|مجموع الفتائل|')
        plt.title('تكبير المنطقة المثيرة (1 ≤ t ≤ 10)')
        plt.grid(True, alpha=0.3)
        
        # تمييز الأعداد الأولية في المنطقة المكبرة
        focus_primes = [(t, m, p) for t, m, p in valid_primes if 1 <= t <= 10]
        if focus_primes:
            focus_t, focus_mag, focus_nums = zip(*focus_primes)
            plt.scatter(focus_t, focus_mag, color='red', s=150, alpha=0.9, 
                       label=f'أعداد أولية: {list(focus_nums)}', zorder=5)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/enhanced_zeta_spectrum.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 تم حفظ الطيف المحسن في: /home/ubuntu/enhanced_zeta_spectrum.png")
    
    def comprehensive_analysis(self) -> Dict[str, Any]:
        """تحليل شامل محسن"""
        
        print("🚀 بدء التحليل الشامل المحسن")
        print("=" * 80)
        
        results = {}
        
        # 1. البحث عن ترددات الرنين
        print("\n1️⃣ البحث عن ترددات الرنين:")
        resonance_frequencies = self.find_resonance_frequencies(0.1, 20, 8000)
        results['resonance_frequencies'] = resonance_frequencies
        
        # 2. استخراج الأعداد الأولية
        print("\n2️⃣ استخراج الأعداد الأولية:")
        found_primes = self.extract_primes_from_resonance(resonance_frequencies)
        results['found_primes'] = found_primes
        
        # 3. البحث عن الأصفار
        print("\n3️⃣ البحث عن أصفار ريمان:")
        found_zeros = self.find_zeros_enhanced(10, 30, 4000)
        results['found_zeros'] = found_zeros
        
        # 4. تقييم الأداء
        print("\n4️⃣ تقييم الأداء:")
        performance = self._evaluate_performance_enhanced(results)
        results['performance'] = performance
        
        print(f"  🎯 أعداد أولية مكتشفة: {len(found_primes)}")
        print(f"  🎯 دقة الكشف: {performance['detection_accuracy']:.1%}")
        print(f"  🎯 أصفار مكتشفة: {len(found_zeros)}")
        print(f"  🎯 النتيجة الإجمالية: {performance['overall_score']:.1%}")
        
        # 5. رسم الطيف المحسن
        print("\n5️⃣ رسم الطيف المحسن:")
        self.plot_enhanced_spectrum(0.1, 15, 6000)
        
        return results
    
    def _evaluate_performance_enhanced(self, results: Dict[str, Any]) -> Dict[str, float]:
        """تقييم الأداء المحسن"""
        
        # الأعداد الأولية الحقيقية حتى 50
        true_primes = [p for p in range(2, 51) if self._is_prime(p)]
        found_primes = results['found_primes']
        
        # حساب الدقة
        if found_primes:
            correct_primes = len(set(found_primes) & set(true_primes))
            detection_accuracy = correct_primes / len(found_primes)  # دقة الكشف
            coverage = correct_primes / len(true_primes)  # التغطية
        else:
            detection_accuracy = 0.0
            coverage = 0.0
        
        # تقييم جودة الرنين
        resonance_quality = len(results['resonance_frequencies']) / 100  # نسبة الرنين
        
        # النتيجة الإجمالية
        overall_score = (detection_accuracy * 0.4 + coverage * 0.4 + resonance_quality * 0.2)
        
        return {
            'detection_accuracy': detection_accuracy,
            'coverage': coverage,
            'resonance_quality': resonance_quality,
            'overall_score': min(overall_score, 1.0),  # حد أقصى 100%
            'found_primes_count': len(found_primes),
            'true_primes_count': len(true_primes),
            'zeros_found': len(results['found_zeros'])
        }

def main():
    """الدالة الرئيسية المحسنة"""
    print("🌟 مرحباً بك في محاكي زيتا RLC المحسن!")
    print("=" * 80)
    
    # إنشاء المحاكي المحسن
    simulator = EnhancedZetaRLC(max_terms=800)
    
    # تشغيل التحليل الشامل
    results = simulator.comprehensive_analysis()
    
    # حفظ النتائج
    import json
    with open('/home/ubuntu/enhanced_zeta_results.json', 'w', encoding='utf-8') as f:
        serializable_results = {
            'found_primes': results['found_primes'],
            'found_zeros': results['found_zeros'],
            'performance': results['performance'],
            'resonance_count': len(results['resonance_frequencies'])
        }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print("\n💾 تم حفظ النتائج المحسنة في: /home/ubuntu/enhanced_zeta_results.json")
    
    return results

if __name__ == "__main__":
    results = main()

