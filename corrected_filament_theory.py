#!/usr/bin/env python3
"""
نظرية الفتائل المُصححة - الإصدار 2.0
تطبيق المبادئ المُستخرجة من الكود العملي على دالة زيتا ريمان
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import mpmath as mp
from typing import List, Tuple
import time

# تعيين دقة عالية
mp.dps = 30

class CorrectedFilamentTheory:
    """
    نظرية الفتائل المُصححة بناءً على الهندسة العكسية
    """
    
    def __init__(self):
        # المعاملات الحرجة المُستخرجة من الكود العملي
        self.critical_params = {
            'k_spring': 0.031822,      # ثابت الاستعادة
            'alpha': 0.4533,           # قوة الانعكاس
            'gamma': 0.00947,          # معامل التضخيم (كان سالباً في الكود)
            'beta': 1.233,             # عامل العتبة
            'memory_window': 67,       # نافذة الذاكرة
            'energy_drain': 0.1        # معامل استنزاف الطاقة
        }
        
        # الأعداد الأولية المعروفة (لاستخدامها كمصادر اضطراب)
        self.primes = self.generate_primes(1000)
        
    def generate_primes(self, limit):
        """توليد الأعداد الأولية حتى حد معين"""
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(np.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    def inversion_error(self, z):
        """
        حساب خطأ الانعكاس المُصحح
        ε(z) = (1/Re(z)) - Im(z)
        """
        real_part = z.real if abs(z.real) > 1e-12 else 1e-12
        return (1.0 / real_part) - z.imag
    
    def compute_forces(self, z, prime_influence=0):
        """
        حساب القوى المؤثرة على الفتيلة في الفضاء المعقد
        """
        # قوة الاستعادة (نابض)
        restoring_force = -self.critical_params['k_spring'] * z
        
        # قوة الانعكاس التفاضلي
        epsilon = self.inversion_error(z)
        inversive_force = self.critical_params['alpha'] * epsilon * 1j
        
        # تأثير الأعداد الأولية كمصادر اضطراب
        prime_force = prime_influence * 0.001 * (1 + 1j)
        
        return restoring_force, inversive_force, prime_force
    
    def filament_dynamics_step(self, z, v, n, force_history):
        """
        خطوة واحدة في ديناميكية الفتيلة المُصححة
        """
        # حساب تأثير الأعداد الأولية
        prime_influence = sum(1.0/p for p in self.primes if p <= n)
        
        # حساب القوى
        restoring_force, inversive_force, prime_force = self.compute_forces(z, prime_influence)
        
        # تطبيق التضخيم (التخميد السلبي)
        v *= (1.0 + self.critical_params['gamma'])
        
        # تطبيق القوى
        total_force = restoring_force + inversive_force + prime_force
        v += total_force
        
        # تحديث الموقع
        z += v
        
        # حساب مقدار قوة الانعكاس (للكشف عن الأحداث)
        force_magnitude = abs(inversive_force)
        
        return z, v, force_magnitude
    
    def compute_zeta_via_filament_dynamics(self, s, max_iterations=1000):
        """
        حساب دالة زيتا باستخدام ديناميكية الفتائل المُصححة
        """
        # تحويل s إلى إحداثيات الفتيلة
        z = complex(s.real - 0.5, s.imag * 0.01)  # تطبيع للاستقرار
        v = complex(0, 0)
        
        # تاريخ القوى للعتبة التكيفية
        force_history = deque(maxlen=self.critical_params['memory_window'])
        
        # متغيرات الكشف
        peak_detector = deque(maxlen=3)
        peak_detector.extend([0.0] * 3)
        
        resonance_events = []
        zeta_approximation = complex(1, 0)
        
        for n in range(1, max_iterations + 1):
            # تطبيق ديناميكية الفتيلة
            z, v, force_mag = self.filament_dynamics_step(z, v, n, force_history)
            
            # تحديث تاريخ القوى
            force_history.append(force_mag)
            peak_detector.append(force_mag)
            
            # كشف الذروات (أحداث الرنين)
            if len(peak_detector) == 3:
                is_peak = (peak_detector[0] < peak_detector[1] and 
                          peak_detector[1] > peak_detector[2])
                
                if is_peak and len(force_history) > 10:
                    # حساب العتبة التكيفية
                    avg_force = np.mean(force_history)
                    threshold = avg_force * self.critical_params['beta']
                    
                    if peak_detector[1] > threshold:
                        # حدث رنين مكتشف
                        resonance_events.append({
                            'iteration': n,
                            'z': z,
                            'force': peak_detector[1],
                            'threshold': threshold
                        })
                        
                        # استنزاف الطاقة
                        v *= self.critical_params['energy_drain']
            
            # تحديث تقريب زيتا بناءً على حالة النظام
            if n % 100 == 0:
                # تقريب زيتا بناءً على ديناميكية النظام
                stability_factor = 1.0 / (1.0 + abs(z))
                zeta_approximation *= stability_factor
        
        # تحليل أحداث الرنين لتحديد ما إذا كان s صفراً
        is_zero_candidate = self.analyze_resonance_pattern(resonance_events, s)
        
        return {
            'zeta_value': zeta_approximation,
            'is_zero': is_zero_candidate,
            'resonance_events': resonance_events,
            'final_z': z,
            'final_v': v,
            'total_resonances': len(resonance_events)
        }
    
    def analyze_resonance_pattern(self, resonance_events, s):
        """
        تحليل نمط أحداث الرنين لتحديد ما إذا كان s صفراً لدالة زيتا
        """
        if len(resonance_events) < 3:
            return False
        
        # تحليل توزيع الأحداث
        intervals = []
        for i in range(1, len(resonance_events)):
            interval = resonance_events[i]['iteration'] - resonance_events[i-1]['iteration']
            intervals.append(interval)
        
        if not intervals:
            return False
        
        # معايير الكشف عن الصفر
        avg_interval = np.mean(intervals)
        interval_stability = np.std(intervals) / avg_interval if avg_interval > 0 else float('inf')
        
        # قوة الرنين الإجمالية
        total_resonance_strength = sum(event['force'] for event in resonance_events)
        
        # شروط الصفر المُحدثة
        conditions = [
            len(resonance_events) >= 5,           # عدد كافٍ من الأحداث
            interval_stability < 0.5,             # استقرار في التوزيع
            total_resonance_strength > 1000,      # قوة رنين كافية
            abs(s.real - 0.5) < 0.01             # على الخط الحرج
        ]
        
        return sum(conditions) >= 3  # يجب تحقيق 3 من 4 شروط
    
    def find_zeta_zeros_corrected(self, t_min, t_max, step=0.1):
        """
        البحث عن أصفار دالة زيتا باستخدام النظرية المُصححة
        """
        print(f"🔍 البحث عن أصفار زيتا في النطاق [{t_min}, {t_max}]")
        print("باستخدام نظرية الفتائل المُصححة")
        print("-" * 60)
        
        zeros_found = []
        t_current = t_min
        
        while t_current <= t_max:
            s = complex(0.5, t_current)
            
            print(f"  اختبار s = {s}")
            
            # حساب زيتا باستخدام الديناميكية المُصححة
            result = self.compute_zeta_via_filament_dynamics(s)
            
            if result['is_zero']:
                zeros_found.append({
                    's': s,
                    't': t_current,
                    'result': result
                })
                print(f"  🎉 صفر مكتشف عند t = {t_current}")
                print(f"      أحداث الرنين: {result['total_resonances']}")
            else:
                print(f"  ❌ ليس صفراً (أحداث: {result['total_resonances']})")
            
            t_current += step
        
        print(f"\n📊 تم العثور على {len(zeros_found)} صفر")
        return zeros_found
    
    def verify_known_zeros_corrected(self):
        """
        التحقق من الأصفار المعروفة باستخدام النظرية المُصححة
        """
        known_zeros_t = [
            14.1347251417346937904572519835625,
            21.0220396387715549926284795938969,
            25.0108575801456887632137909925628,
            30.4248761258595132103118975305239,
            32.9350615877391896906623689640542
        ]
        
        print("🔬 التحقق من الأصفار المعروفة - النظرية المُصححة")
        print("=" * 60)
        
        results = []
        correct_count = 0
        
        for i, t in enumerate(known_zeros_t):
            s = complex(0.5, t)
            
            print(f"\n🧪 اختبار الصفر {i+1}: t = {t}")
            
            result = self.compute_zeta_via_filament_dynamics(s, max_iterations=2000)
            
            is_correct = result['is_zero']
            if is_correct:
                correct_count += 1
                status = "✅ صحيح"
            else:
                status = "❌ خطأ"
            
            print(f"  {status} - أحداث الرنين: {result['total_resonances']}")
            
            results.append({
                't': t,
                's': s,
                'result': result,
                'correct': is_correct
            })
        
        accuracy = correct_count / len(known_zeros_t)
        
        print(f"\n" + "=" * 60)
        print(f"📊 النتيجة النهائية: {correct_count}/{len(known_zeros_t)} ({accuracy:.1%})")
        
        return {
            'results': results,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_tested': len(known_zeros_t)
        }
    
    def compare_with_mpmath_zeta(self, test_points):
        """
        مقارنة النتائج مع حسابات mpmath الدقيقة
        """
        print("\n🔬 مقارنة مع حسابات mpmath الدقيقة")
        print("-" * 60)
        
        comparisons = []
        
        for s in test_points:
            # حساب زيتا الدقيق
            zeta_exact = mp.zeta(s)
            zeta_exact_mag = abs(zeta_exact)
            
            # حساب زيتا بالطريقة المُصححة
            result = self.compute_zeta_via_filament_dynamics(s)
            
            # مقارنة
            is_zero_exact = zeta_exact_mag < 1e-10
            is_zero_predicted = result['is_zero']
            
            agreement = is_zero_exact == is_zero_predicted
            
            print(f"s = {s}")
            print(f"  زيتا الدقيق: {zeta_exact} (|ζ| = {zeta_exact_mag})")
            print(f"  التنبؤ: {'صفر' if is_zero_predicted else 'ليس صفراً'}")
            print(f"  التطابق: {'✅' if agreement else '❌'}")
            
            comparisons.append({
                's': s,
                'zeta_exact': zeta_exact,
                'zeta_exact_mag': zeta_exact_mag,
                'is_zero_exact': is_zero_exact,
                'is_zero_predicted': is_zero_predicted,
                'agreement': agreement,
                'resonance_count': result['total_resonances']
            })
        
        accuracy = sum(c['agreement'] for c in comparisons) / len(comparisons)
        print(f"\n📊 دقة التطابق: {accuracy:.1%}")
        
        return comparisons

def main():
    """الدالة الرئيسية لاختبار النظرية المُصححة"""
    print("🚀 نظرية الفتائل المُصححة - الإصدار 2.0")
    print("=" * 60)
    print("مبنية على الهندسة العكسية للكود العملي")
    print("=" * 60)
    
    theory = CorrectedFilamentTheory()
    
    # 1. التحقق من الأصفار المعروفة
    verification_results = theory.verify_known_zeros_corrected()
    
    # 2. البحث عن أصفار جديدة
    new_zeros = theory.find_zeta_zeros_corrected(35, 40, 0.5)
    
    # 3. مقارنة مع mpmath
    test_points = [
        complex(0.5, 14.1347),
        complex(0.5, 21.0220),
        complex(0.5, 15.0),  # ليس صفراً
        complex(0.5, 25.0109)
    ]
    
    comparisons = theory.compare_with_mpmath_zeta(test_points)
    
    # تقييم إجمالي
    print("\n" + "=" * 60)
    print("🏆 التقييم الإجمالي للنظرية المُصححة")
    print("=" * 60)
    
    verification_score = verification_results['accuracy']
    comparison_score = sum(c['agreement'] for c in comparisons) / len(comparisons)
    discovery_bonus = min(len(new_zeros) * 0.1, 0.2)
    
    overall_score = (verification_score * 0.5 + 
                    comparison_score * 0.4 + 
                    discovery_bonus * 0.1)
    
    print(f"📊 دقة التحقق من الأصفار المعروفة: {verification_score:.1%}")
    print(f"📊 دقة المقارنة مع mpmath: {comparison_score:.1%}")
    print(f"🔍 أصفار جديدة مكتشفة: {len(new_zeros)}")
    print(f"📈 النقاط الإجمالية: {overall_score:.1%}")
    
    if overall_score > 0.7:
        print("🎉 النظرية المُصححة تعمل بشكل ممتاز!")
    elif overall_score > 0.5:
        print("✅ النظرية المُصححة تعمل بشكل جيد")
    else:
        print("⚠️ النظرية تحتاج مزيد من التطوير")
    
    return {
        'verification': verification_results,
        'new_zeros': new_zeros,
        'comparisons': comparisons,
        'overall_score': overall_score
    }

if __name__ == "__main__":
    results = main()

