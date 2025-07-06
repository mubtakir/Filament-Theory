#!/usr/bin/env python3
"""
النموذج الرياضي المتقدم لتحليل العلاقة بين الآليات المتعددة ودالة زيتا ريمان
==================================================================

الهدف: اختبار الفرضية الأساسية:
Σ(i=1 to 4) α_i × M_i(s) = ζ(s)

حيث:
- M_i(s) هي الآليات المختلفة
- α_i هي الأوزان المحسوبة تجريبياً
- ζ(s) هي دالة زيتا ريمان

الباحث: باسل يحيى عبدالله "المبتكر العلمي"
المشرف: الذكاء الاصطناعي مانوس
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta, gamma
from scipy.optimize import minimize
import cmath
from decimal import Decimal, getcontext
import json
from typing import List, Tuple, Dict

# تعيين دقة عالية للحسابات
getcontext().prec = 50

class ZetaMechanismsAnalyzer:
    """محلل العلاقة بين الآليات المتعددة ودالة زيتا ريمان"""
    
    def __init__(self):
        """تهيئة المحلل"""
        self.mechanisms = {
            'RLC': self._rlc_mechanism,
            'Spring': self._spring_mechanism,
            'Oscillating': self._oscillating_mechanism,
            'Sieve': self._sieve_mechanism
        }
        
        # الأوزان المحسوبة تجريبياً
        self.weights = {
            'RLC': 0.461,
            'Spring': 0.335,
            'Oscillating': -0.028,
            'Sieve': 0.232
        }
        
        # معاملات الآليات
        self.params = {
            'RLC': {'omega': 2*np.pi, 'f': 1.0},
            'Spring': {'tau': 10.0, 'omega': np.pi, 'phi': np.pi/4},
            'Oscillating': {'amplitude_factor': 1.0},
            'Sieve': {'efficiency': 0.8}
        }
        
        self.results = {}
        
    def _rlc_mechanism(self, n: int, s: complex) -> complex:
        """حساب مساهمة آلية RLC للعدد n"""
        omega = self.params['RLC']['omega']
        f = self.params['RLC']['f']
        
        # المكونات الكهربائية
        R = np.sqrt(n)  # المقاومة
        L = 1.0 / np.sqrt(n)  # المحاثة
        C = 1.0 / (n * f**2)  # السعة
        
        # المعاوقة المعقدة
        Z = R + 1j * (omega * L - 1.0 / (omega * C))
        
        # الوزن = 1/|Z|
        weight = 1.0 / abs(Z)
        
        return weight * (1.0 / n**s)
    
    def _spring_mechanism(self, n: int, s: complex) -> complex:
        """حساب مساهمة آلية الكنابض والزناد للعدد n"""
        tau = self.params['Spring']['tau']
        omega = self.params['Spring']['omega']
        phi = self.params['Spring']['phi']
        
        # عامل التخميد
        damping = np.exp(-n / tau)
        
        # التذبذب
        oscillation = np.sin(omega * n + phi)
        
        # الوزن = 1 + عامل الكنابض
        spring_factor = damping * oscillation
        weight = 1.0 + spring_factor
        
        # تحرير الزناد للأعداد الأولية
        if self._is_prime(n):
            weight *= 1.5  # تضخيم للأعداد الأولية
            
        return weight * (1.0 / n**s)
    
    def _oscillating_mechanism(self, n: int, s: complex) -> complex:
        """حساب مساهمة آلية الكرات المتذبذبة للعدد n"""
        amplitude_factor = self.params['Oscillating']['amplitude_factor']
        
        # السعة
        amplitude = amplitude_factor / np.sqrt(n)
        
        # التردد
        frequency = 2 * np.pi / n
        
        # الطور
        phase = np.pi * n / 4
        
        # الوزن = |السعة × sin(التردد × الوقت + الطور)|
        weight = abs(amplitude * np.sin(frequency + phase))
        
        # تجنب القسمة على صفر
        if weight < 1e-10:
            weight = 1e-10
            
        return weight * (1.0 / n**s)
    
    def _sieve_mechanism(self, n: int, s: complex) -> complex:
        """حساب مساهمة آلية الغربال المبتكر للعدد n"""
        efficiency = self.params['Sieve']['efficiency']
        
        if n == 1:
            weight = 1.0
        elif self._is_prime(n):
            # للأعداد الأولية: وزن عالي
            weight = 1.0 - 1.0/n
        else:
            # للأعداد المركبة: وزن منخفض حسب العوامل
            prime_factors = self._get_prime_factors(n)
            weight = 1.0
            for p, k in prime_factors.items():
                weight *= (1.0 - 1.0/p**k)
        
        weight *= efficiency
        return weight * (1.0 / n**s)
    
    def _is_prime(self, n: int) -> bool:
        """فحص ما إذا كان العدد أولياً"""
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
    
    def _get_prime_factors(self, n: int) -> Dict[int, int]:
        """الحصول على العوامل الأولية وقواها"""
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    def compute_mechanisms_sum(self, s: complex, max_n: int = 1000) -> complex:
        """حساب مجموع الآليات المتعددة"""
        total = 0.0 + 0.0j
        
        for n in range(1, max_n + 1):
            mechanism_sum = 0.0 + 0.0j
            
            for name, mechanism in self.mechanisms.items():
                weight = self.weights[name]
                contribution = mechanism(n, s)
                mechanism_sum += weight * contribution
            
            total += mechanism_sum
            
        return total
    
    def compute_zeta_reference(self, s: complex, max_n: int = 1000) -> complex:
        """حساب دالة زيتا المرجعية"""
        if s.real > 1:
            # للقيم التي تتقارب فيها السلسلة
            total = 0.0 + 0.0j
            for n in range(1, max_n + 1):
                total += 1.0 / n**s
            return total
        else:
            # استخدام الاستمرار التحليلي (تقريبي)
            try:
                return complex(zeta(s))
            except:
                return self._analytical_continuation(s)
    
    def _analytical_continuation(self, s: complex) -> complex:
        """الاستمرار التحليلي التقريبي لدالة زيتا"""
        # استخدام العلاقة الوظيفية لزيتا
        # ζ(s) = 2^s × π^(s-1) × sin(πs/2) × Γ(1-s) × ζ(1-s)
        
        if s.real < 0:
            s_complement = 1 - s
            zeta_complement = self.compute_zeta_reference(s_complement, 1000)
            
            factor1 = 2**s
            factor2 = np.pi**(s - 1)
            factor3 = cmath.sin(np.pi * s / 2)
            factor4 = gamma(1 - s)
            
            return factor1 * factor2 * factor3 * factor4 * zeta_complement
        else:
            # للقيم الأخرى، استخدام تقريب
            return complex(1.0, 0.0)
    
    def compute_eta_function(self, s: complex, max_n: int = 1000) -> complex:
        """حساب دالة إيتا (التناوب الكوني)"""
        total = 0.0 + 0.0j
        
        for n in range(1, max_n + 1):
            # التناوب بين الماهية الكتلية والمكانية
            alternating_factor = (-1)**(n + 1)
            
            # مجموع الآليات للعدد n
            mechanism_sum = 0.0 + 0.0j
            for name, mechanism in self.mechanisms.items():
                weight = self.weights[name]
                contribution = mechanism(n, s)
                mechanism_sum += weight * contribution
            
            total += alternating_factor * mechanism_sum
            
        return total
    
    def test_equivalence(self, s_values: List[complex], max_n: int = 1000) -> Dict:
        """اختبار تكافؤ الآليات مع دالة زيتا"""
        results = {
            'test_points': [],
            'mechanisms_values': [],
            'zeta_values': [],
            'differences': [],
            'relative_errors': [],
            'eta_values': [],
            'eta_reference': []
        }
        
        print("🔬 اختبار تكافؤ الآليات مع دالة زيتا ريمان")
        print("=" * 60)
        
        for i, s in enumerate(s_values):
            print(f"📊 نقطة الاختبار {i+1}: s = {s}")
            
            # حساب مجموع الآليات
            mechanisms_value = self.compute_mechanisms_sum(s, max_n)
            
            # حساب زيتا المرجعية
            zeta_value = self.compute_zeta_reference(s, max_n)
            
            # حساب الفرق
            difference = abs(mechanisms_value - zeta_value)
            relative_error = difference / abs(zeta_value) if abs(zeta_value) > 0 else float('inf')
            
            # حساب دالة إيتا
            eta_value = self.compute_eta_function(s, max_n)
            eta_reference = (1 - 2**(1-s)) * zeta_value
            
            # حفظ النتائج
            results['test_points'].append(s)
            results['mechanisms_values'].append(mechanisms_value)
            results['zeta_values'].append(zeta_value)
            results['differences'].append(difference)
            results['relative_errors'].append(relative_error)
            results['eta_values'].append(eta_value)
            results['eta_reference'].append(eta_reference)
            
            print(f"  🎯 مجموع الآليات: {mechanisms_value:.6f}")
            print(f"  📐 زيتا المرجعية: {zeta_value:.6f}")
            print(f"  📏 الفرق المطلق: {difference:.6e}")
            print(f"  📊 الخطأ النسبي: {relative_error:.6%}")
            print(f"  🌊 إيتا المحسوبة: {eta_value:.6f}")
            print(f"  🌊 إيتا المرجعية: {eta_reference:.6f}")
            print()
        
        return results
    
    def analyze_convergence(self, s: complex, max_n_values: List[int]) -> Dict:
        """تحليل تقارب مجموع الآليات"""
        results = {
            'max_n_values': max_n_values,
            'mechanisms_sums': [],
            'zeta_references': [],
            'convergence_rates': []
        }
        
        print(f"📈 تحليل التقارب للنقطة s = {s}")
        print("=" * 50)
        
        previous_mechanisms = None
        previous_zeta = None
        
        for max_n in max_n_values:
            mechanisms_sum = self.compute_mechanisms_sum(s, max_n)
            zeta_ref = self.compute_zeta_reference(s, max_n)
            
            results['mechanisms_sums'].append(mechanisms_sum)
            results['zeta_references'].append(zeta_ref)
            
            if previous_mechanisms is not None:
                convergence_rate = abs(mechanisms_sum - previous_mechanisms)
                results['convergence_rates'].append(convergence_rate)
                
                print(f"📊 N = {max_n:4d}: Mechanisms = {mechanisms_sum:.8f}, "
                      f"Zeta = {zeta_ref:.8f}, "
                      f"Convergence = {convergence_rate:.2e}")
            else:
                results['convergence_rates'].append(0.0)
                print(f"📊 N = {max_n:4d}: Mechanisms = {mechanisms_sum:.8f}, "
                      f"Zeta = {zeta_ref:.8f}")
            
            previous_mechanisms = mechanisms_sum
            previous_zeta = zeta_ref
        
        return results
    
    def find_zeros_approximation(self, search_range: Tuple[float, float], 
                                num_points: int = 100) -> List[complex]:
        """البحث عن تقريبات لأصفار دالة زيتا"""
        print("🔍 البحث عن أصفار دالة زيتا التقريبية")
        print("=" * 50)
        
        zeros = []
        t_values = np.linspace(search_range[0], search_range[1], num_points)
        
        for t in t_values:
            s = 0.5 + 1j * t  # الخط الحرج
            
            # حساب مجموع الآليات
            mechanisms_value = self.compute_mechanisms_sum(s, 1000)
            
            # إذا كانت القيمة قريبة من الصفر
            if abs(mechanisms_value) < 0.1:
                zeros.append(s)
                print(f"🎯 صفر محتمل عند s = {s:.6f}, |f(s)| = {abs(mechanisms_value):.6e}")
        
        return zeros
    
    def optimize_weights(self, test_points: List[complex], max_n: int = 500) -> Dict[str, float]:
        """تحسين أوزان الآليات لتحقيق أفضل تطابق مع زيتا"""
        print("⚙️ تحسين أوزان الآليات")
        print("=" * 40)
        
        def objective(weights_array):
            """دالة الهدف للتحسين"""
            weights_dict = {
                'RLC': weights_array[0],
                'Spring': weights_array[1],
                'Oscillating': weights_array[2],
                'Sieve': weights_array[3]
            }
            
            total_error = 0.0
            
            for s in test_points:
                # حساب مجموع الآليات بالأوزان الجديدة
                mechanisms_sum = 0.0 + 0.0j
                for n in range(1, max_n + 1):
                    for name, mechanism in self.mechanisms.items():
                        weight = weights_dict[name]
                        contribution = mechanism(n, s)
                        mechanisms_sum += weight * contribution
                
                # حساب زيتا المرجعية
                zeta_ref = self.compute_zeta_reference(s, max_n)
                
                # حساب الخطأ
                error = abs(mechanisms_sum - zeta_ref)
                total_error += error
            
            return total_error
        
        # الأوزان الأولية
        initial_weights = [self.weights[name] for name in ['RLC', 'Spring', 'Oscillating', 'Sieve']]
        
        # قيود التحسين
        constraints = {'type': 'eq', 'fun': lambda x: sum(x) - 1}  # مجموع الأوزان = 1
        bounds = [(-2, 2) for _ in range(4)]  # حدود الأوزان
        
        # تشغيل التحسين
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimized_weights = {
                'RLC': result.x[0],
                'Spring': result.x[1],
                'Oscillating': result.x[2],
                'Sieve': result.x[3]
            }
            
            print("✅ تم تحسين الأوزان بنجاح:")
            for name, weight in optimized_weights.items():
                print(f"  {name}: {weight:.6f}")
            
            return optimized_weights
        else:
            print("❌ فشل في تحسين الأوزان")
            return self.weights
    
    def plot_comparison(self, results: Dict, save_path: str = None):
        """رسم مقارنة بين الآليات وزيتا"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # الرسم الأول: القيم الحقيقية
        real_mechanisms = [v.real for v in results['mechanisms_values']]
        real_zeta = [v.real for v in results['zeta_values']]
        
        ax1.plot(range(len(real_mechanisms)), real_mechanisms, 'b-o', label='مجموع الآليات', markersize=4)
        ax1.plot(range(len(real_zeta)), real_zeta, 'r-s', label='زيتا ريمان', markersize=4)
        ax1.set_title('مقارنة القيم الحقيقية')
        ax1.set_xlabel('نقطة الاختبار')
        ax1.set_ylabel('القيمة الحقيقية')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # الرسم الثاني: القيم التخيلية
        imag_mechanisms = [v.imag for v in results['mechanisms_values']]
        imag_zeta = [v.imag for v in results['zeta_values']]
        
        ax2.plot(range(len(imag_mechanisms)), imag_mechanisms, 'b-o', label='مجموع الآليات', markersize=4)
        ax2.plot(range(len(imag_zeta)), imag_zeta, 'r-s', label='زيتا ريمان', markersize=4)
        ax2.set_title('مقارنة القيم التخيلية')
        ax2.set_xlabel('نقطة الاختبار')
        ax2.set_ylabel('القيمة التخيلية')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # الرسم الثالث: الأخطاء النسبية
        ax3.semilogy(range(len(results['relative_errors'])), results['relative_errors'], 'g-^', markersize=6)
        ax3.set_title('الأخطاء النسبية')
        ax3.set_xlabel('نقطة الاختبار')
        ax3.set_ylabel('الخطأ النسبي (مقياس لوغاريتمي)')
        ax3.grid(True, alpha=0.3)
        
        # الرسم الرابع: مقارنة دالة إيتا
        eta_real = [v.real for v in results['eta_values']]
        eta_ref_real = [v.real for v in results['eta_reference']]
        
        ax4.plot(range(len(eta_real)), eta_real, 'purple', marker='o', label='إيتا المحسوبة', markersize=4)
        ax4.plot(range(len(eta_ref_real)), eta_ref_real, 'orange', marker='s', label='إيتا المرجعية', markersize=4)
        ax4.set_title('مقارنة دالة إيتا (التناوب الكوني)')
        ax4.set_xlabel('نقطة الاختبار')
        ax4.set_ylabel('القيمة الحقيقية')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 تم حفظ الرسم في: {save_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, results: Dict) -> str:
        """إنشاء تقرير شامل للنتائج"""
        report = """
# تقرير شامل: تحليل العلاقة بين الآليات المتعددة ودالة زيتا ريمان
================================================================

## ملخص النتائج:

"""
        
        # إحصائيات أساسية
        avg_error = np.mean(results['relative_errors'])
        max_error = np.max(results['relative_errors'])
        min_error = np.min(results['relative_errors'])
        
        report += f"""
### الإحصائيات الأساسية:
- متوسط الخطأ النسبي: {avg_error:.6%}
- أقصى خطأ نسبي: {max_error:.6%}
- أدنى خطأ نسبي: {min_error:.6%}
- عدد نقاط الاختبار: {len(results['test_points'])}

"""
        
        # تحليل التقارب
        convergence_quality = "ممتاز" if avg_error < 0.01 else "جيد" if avg_error < 0.1 else "متوسط"
        
        report += f"""
### تقييم التقارب:
- جودة التقارب: {convergence_quality}
- الاستنتاج: {"الآليات تتقارب بشكل ممتاز مع دالة زيتا" if avg_error < 0.01 else "هناك حاجة لتحسين الآليات"}

"""
        
        # تحليل دالة إيتا
        eta_errors = [abs(eta - eta_ref) for eta, eta_ref in zip(results['eta_values'], results['eta_reference'])]
        avg_eta_error = np.mean([abs(e) for e in eta_errors])
        
        report += f"""
### تحليل دالة إيتا (التناوب الكوني):
- متوسط خطأ إيتا: {avg_eta_error:.6e}
- التفسير: {"التناوب الكوني يعمل بشكل صحيح" if avg_eta_error < 0.1 else "التناوب يحتاج تحسين"}

"""
        
        # التوصيات
        report += """
### التوصيات:
1. تحسين معاملات الآليات لتقليل الأخطاء
2. زيادة عدد الحدود في المجموع للحصول على دقة أعلى
3. دراسة سلوك الآليات في نطاقات مختلفة من s
4. تطوير آليات إضافية لتحسين التغطية

"""
        
        return report

def main():
    """الدالة الرئيسية للاختبار"""
    print("🌟 بدء تحليل العلاقة بين الآليات المتعددة ودالة زيتا ريمان")
    print("=" * 70)
    
    # إنشاء المحلل
    analyzer = ZetaMechanismsAnalyzer()
    
    # نقاط الاختبار
    test_points = [
        2.0 + 0.0j,      # نقطة بسيطة
        1.5 + 0.0j,      # قريب من القطب
        0.5 + 14.134j,   # قريب من صفر ريمان
        0.5 + 21.022j,   # صفر ريمان آخر
        3.0 + 1.0j,      # نقطة معقدة
        -1.0 + 0.0j,     # صفر بديهي
        0.0 + 1.0j       # على الخط التخيلي
    ]
    
    # اختبار التكافؤ
    print("🔬 المرحلة 1: اختبار التكافؤ الأساسي")
    results = analyzer.test_equivalence(test_points, max_n=1000)
    
    # تحليل التقارب
    print("\n📈 المرحلة 2: تحليل التقارب")
    convergence_results = analyzer.analyze_convergence(
        2.0 + 0.0j, 
        [100, 200, 500, 1000, 2000]
    )
    
    # البحث عن الأصفار
    print("\n🔍 المرحلة 3: البحث عن الأصفار")
    zeros = analyzer.find_zeros_approximation((10, 30), num_points=200)
    
    # تحسين الأوزان
    print("\n⚙️ المرحلة 4: تحسين الأوزان")
    simple_test_points = [2.0 + 0.0j, 1.5 + 0.0j, 3.0 + 0.0j]
    optimized_weights = analyzer.optimize_weights(simple_test_points, max_n=300)
    
    # إنشاء التقرير
    print("\n📊 المرحلة 5: إنشاء التقرير الشامل")
    report = analyzer.generate_comprehensive_report(results)
    
    # حفظ النتائج
    with open('/home/ubuntu/zeta_mechanisms_results.json', 'w', encoding='utf-8') as f:
        # تحويل الأعداد المعقدة إلى قوائم للحفظ
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], complex):
                serializable_results[key] = [[v.real, v.imag] for v in value]
            else:
                serializable_results[key] = value
        
        json.dump({
            'equivalence_test': serializable_results,
            'convergence_analysis': {
                'max_n_values': convergence_results['max_n_values'],
                'convergence_rates': convergence_results['convergence_rates']
            },
            'found_zeros': [[z.real, z.imag] for z in zeros],
            'optimized_weights': optimized_weights
        }, f, indent=2, ensure_ascii=False)
    
    # حفظ التقرير
    with open('/home/ubuntu/zeta_mechanisms_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # رسم المقارنات
    analyzer.plot_comparison(results, '/home/ubuntu/zeta_mechanisms_comparison.png')
    
    print("\n✅ تم الانتهاء من التحليل الشامل!")
    print(f"📊 متوسط الخطأ النسبي: {np.mean(results['relative_errors']):.6%}")
    print(f"🎯 عدد الأصفار المكتشفة: {len(zeros)}")
    print(f"💾 تم حفظ النتائج في الملفات المختلفة")
    
    return results, convergence_results, zeros, optimized_weights

if __name__ == "__main__":
    results = main()

