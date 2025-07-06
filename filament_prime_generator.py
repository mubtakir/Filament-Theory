#!/usr/bin/env python3
"""
مولد الأعداد الأولية الفتيلي - النموذج المطور
مبني على نظرية الفتائل والنظام الديناميكي المعقد
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

class FilamentPrimeGenerator:
    """
    مولد الأعداد الأولية باستخدام نظرية الفتائل
    النظرية: الأعداد الأولية تظهر عند ذروات المقاومة الداخلية للنظام
    """
    
    def __init__(self, params):
        self.params = params
        self.reset()
        
    def reset(self):
        """إعادة تعيين النظام للحالة الأولية"""
        # الفتيلة الأساسية (موقع + سرعة في الفضاء المعقد)
        self.z = 0.1 + 0.1j  # موقع الفتيلة
        self.v = 0.0 + 0.0j  # سرعة الفتيلة
        
        # تاريخ النظام
        self.history = {
            'n': [], 
            'resistance_force': [], 
            'dynamic_threshold': [],
            'z_real': [],
            'z_imag': [],
            'velocity_mag': []
        }
        
        self.prime_sites = []
        
        # نافذة للكشف عن الذروات
        window_size = max(1, int(round(self.params.get('avg_window_size', 50))))
        self.force_peak_detector = deque(maxlen=3)
        self.force_peak_detector.extend([0.0] * 3)
        self.force_history = deque(maxlen=window_size)
        
    def compute_forces(self, n):
        """حساب القوى المؤثرة على الفتيلة"""
        c, L = self.z.real, self.z.imag
        
        # تجنب القسمة على صفر
        if abs(c) < 1e-9:
            c = np.sign(c) * 1e-9 if c != 0 else 1e-9
            
        # قوة الاستعادة (نابض)
        restoring_force = -self.params['k_spring'] * self.z
        
        # خطأ الانعكاس (1/c - L)
        inversion_error = (1.0 / c) - L
        
        # قوة الانعكاس (تعمل في الاتجاه التخيلي)
        inversive_force = self.params['inversion_strength'] * inversion_error * 1j
        
        return restoring_force, inversive_force, inversion_error
    
    def update_system(self, n):
        """تحديث حالة النظام للخطوة التالية"""
        # حساب القوى
        restoring_force, inversive_force, inversion_error = self.compute_forces(n)
        
        # تطبيق التخميد
        self.v *= (1.0 - self.params['damping'])
        
        # تطبيق القوى
        self.v += restoring_force + inversive_force
        
        # تحديث الموقع
        self.z += self.v
        
        # حساب مقدار قوة المقاومة
        resistance_force_mag = np.abs(inversive_force)
        
        return resistance_force_mag, inversion_error
    
    def detect_prime_event(self, n, resistance_force_mag):
        """كشف حدث العدد الأولي"""
        # تحديث كاشف الذروات
        self.force_peak_detector.append(resistance_force_mag)
        self.force_history.append(resistance_force_mag)
        
        # كشف الذروة (القيمة الوسطى أكبر من الجانبين)
        is_peak = (self.force_peak_detector[0] < self.force_peak_detector[1] and 
                  self.force_peak_detector[1] > self.force_peak_detector[2])
        
        # حساب العتبة الديناميكية
        moving_average = np.mean(self.force_history) if len(self.force_history) > 0 else 0
        dynamic_threshold = moving_average * self.params['threshold_factor']
        
        prime_detected = False
        
        if is_peak:
            peak_force = self.force_peak_detector[1]
            
            if peak_force > dynamic_threshold:
                candidate = n - 1  # المرشح للعدد الأولي
                
                # التحقق من الشروط
                last_prime = self.prime_sites[-1] if self.prime_sites else 0
                
                if (candidate > last_prime and 
                    candidate > 1 and 
                    self.is_prime(candidate)):
                    
                    self.prime_sites.append(candidate)
                    prime_detected = True
                    
                    # تقليل السرعة بعد اكتشاف العدد الأولي
                    self.v *= 0.1
                    
                    print(f"🌟 عدد أولي مكتشف: {candidate} "
                          f"(قوة الذروة: {peak_force:.2f} > العتبة: {dynamic_threshold:.2f})")
        
        return prime_detected, dynamic_threshold
    
    def is_prime(self, n):
        """فحص بسيط للأعداد الأولية"""
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
    
    def run_simulation(self, max_n):
        """تشغيل المحاكاة الكاملة"""
        print(f"🚀 بدء محاكاة مولد الأعداد الأولية الفتيلي")
        print(f"📊 المعاملات: {self.params}")
        print("-" * 60)
        
        self.reset()
        
        # إضافة 2 كعدد أولي أساسي
        if 2 <= max_n:
            self.prime_sites.append(2)
            
        for n in range(1, max_n + 1):
            # تحديث النظام
            resistance_force_mag, inversion_error = self.update_system(n)
            
            # كشف الأعداد الأولية
            prime_detected, dynamic_threshold = self.detect_prime_event(n, resistance_force_mag)
            
            # حفظ التاريخ
            self.history['n'].append(n)
            self.history['resistance_force'].append(resistance_force_mag)
            self.history['dynamic_threshold'].append(dynamic_threshold)
            self.history['z_real'].append(self.z.real)
            self.history['z_imag'].append(self.z.imag)
            self.history['velocity_mag'].append(np.abs(self.v))
            
        print("-" * 60)
        print(f"✅ المحاكاة مكتملة. تم اكتشاف {len(self.prime_sites)} عدد أولي")
        
    def analyze_performance(self, max_n):
        """تحليل أداء المولد"""
        # الأعداد الأولية الفعلية
        actual_primes = set()
        for i in range(2, max_n + 1):
            if self.is_prime(i):
                actual_primes.add(i)
        
        # الأعداد المولدة
        generated_primes = set(self.prime_sites)
        
        # التحليل
        common = generated_primes & actual_primes
        false_positives = generated_primes - actual_primes
        false_negatives = actual_primes - generated_primes
        
        precision = len(common) / len(generated_primes) if generated_primes else 0
        recall = len(common) / len(actual_primes) if actual_primes else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 تحليل الأداء")
        print("=" * 60)
        print(f"🎯 الأعداد الأولية الفعلية: {len(actual_primes)}")
        print(f"🔍 الأعداد المولدة: {len(generated_primes)}")
        print(f"✅ الصحيحة: {len(common)}")
        print(f"❌ الخاطئة الإيجابية: {len(false_positives)}")
        print(f"❌ الخاطئة السلبية: {len(false_negatives)}")
        print("-" * 40)
        print(f"📈 الدقة (Precision): {precision:.2%}")
        print(f"📈 الاستدعاء (Recall): {recall:.2%}")
        print(f"📈 F1-Score: {f1_score:.2%}")
        
        if false_positives:
            print(f"\n❌ أعداد خاطئة تم توليدها: {sorted(list(false_positives))}")
        
        # عرض أول 20 عدد أولي مولد
        print(f"\n🔢 أول 20 عدد أولي مولد: {self.prime_sites[:20]}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'actual_count': len(actual_primes),
            'generated_count': len(generated_primes),
            'correct_count': len(common),
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def visualize_system(self):
        """رسم تصور للنظام"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('نظام مولد الأعداد الأولية الفتيلي', fontsize=16)
        
        # الرسم الأول: قوة المقاومة والعتبة
        ax1 = axes[0, 0]
        ax1.plot(self.history['n'], self.history['resistance_force'], 
                color='purple', label='قوة المقاومة', linewidth=1)
        ax1.plot(self.history['n'], self.history['dynamic_threshold'], 
                color='red', linestyle='--', label='العتبة الديناميكية')
        
        # إضافة نقاط الأعداد الأولية
        if self.prime_sites:
            prime_indices = [p for p in self.prime_sites if p <= len(self.history['n'])]
            prime_forces = [self.history['resistance_force'][p-1] for p in prime_indices 
                          if p-1 < len(self.history['resistance_force'])]
            ax1.scatter(prime_indices, prime_forces, color='gold', s=50, 
                       edgecolor='black', label='أعداد أولية', zorder=5)
        
        ax1.set_xlabel('العدد (n)')
        ax1.set_ylabel('قوة المقاومة')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('كشف الأعداد الأولية')
        
        # الرسم الثاني: مسار الفتيلة في الفضاء المعقد
        ax2 = axes[0, 1]
        ax2.plot(self.history['z_real'], self.history['z_imag'], 
                color='blue', alpha=0.7, linewidth=1)
        ax2.scatter(self.history['z_real'][0], self.history['z_imag'][0], 
                   color='green', s=100, label='البداية', zorder=5)
        ax2.scatter(self.history['z_real'][-1], self.history['z_imag'][-1], 
                   color='red', s=100, label='النهاية', zorder=5)
        ax2.set_xlabel('الجزء الحقيقي')
        ax2.set_ylabel('الجزء التخيلي')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('مسار الفتيلة في الفضاء المعقد')
        
        # الرسم الثالث: سرعة النظام
        ax3 = axes[1, 0]
        ax3.plot(self.history['n'], self.history['velocity_mag'], 
                color='orange', linewidth=1)
        ax3.set_xlabel('العدد (n)')
        ax3.set_ylabel('مقدار السرعة')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('ديناميكية السرعة')
        
        # الرسم الرابع: توزيع الأعداد الأولية
        ax4 = axes[1, 1]
        if len(self.prime_sites) > 1:
            gaps = [self.prime_sites[i+1] - self.prime_sites[i] 
                   for i in range(len(self.prime_sites)-1)]
            ax4.hist(gaps, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax4.set_xlabel('الفجوة بين الأعداد الأولية')
            ax4.set_ylabel('التكرار')
            ax4.set_title('توزيع الفجوات')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def test_different_parameters():
    """اختبار معاملات مختلفة"""
    print("🧪 اختبار معاملات مختلفة لمولد الأعداد الأولية")
    print("=" * 60)
    
    # المعاملات الذهبية الأصلية
    golden_params = {
        'k_spring': 0.031822,
        'damping': -0.00947,
        'inversion_strength': 0.4533,
        'avg_window_size': 67,
        'threshold_factor': 1.233,
    }
    
    # معاملات للاختبار
    test_params = [
        golden_params,
        {**golden_params, 'threshold_factor': 1.5},
        {**golden_params, 'inversion_strength': 0.6},
        {**golden_params, 'avg_window_size': 50},
    ]
    
    results = []
    
    for i, params in enumerate(test_params):
        print(f"\n🔬 اختبار المجموعة {i+1}: {params}")
        
        generator = FilamentPrimeGenerator(params)
        generator.run_simulation(max_n=200)
        performance = generator.analyze_performance(max_n=200)
        
        results.append({
            'params': params,
            'performance': performance,
            'generator': generator
        })
    
    # مقارنة النتائج
    print("\n" + "=" * 60)
    print("📊 مقارنة النتائج")
    print("=" * 60)
    
    for i, result in enumerate(results):
        perf = result['performance']
        print(f"المجموعة {i+1}: F1={perf['f1_score']:.2%}, "
              f"دقة={perf['precision']:.2%}, "
              f"استدعاء={perf['recall']:.2%}")
    
    return results

def main():
    """الدالة الرئيسية"""
    print("🌟 مولد الأعداد الأولية الفتيلي - الإصدار المطور")
    print("=" * 60)
    
    # المعاملات الذهبية
    golden_laws = {
        'k_spring': 0.031822,
        'damping': -0.00947,
        'inversion_strength': 0.4533,
        'avg_window_size': 67,
        'threshold_factor': 1.233,
    }
    
    # تشغيل المحاكاة الأساسية
    generator = FilamentPrimeGenerator(golden_laws)
    
    start_time = time.time()
    generator.run_simulation(max_n=1000)
    end_time = time.time()
    
    # تحليل الأداء
    performance = generator.analyze_performance(max_n=1000)
    
    print(f"\n⏱️ وقت التنفيذ: {end_time - start_time:.2f} ثانية")
    
    # رسم النتائج
    generator.visualize_system()
    
    # اختبار معاملات مختلفة
    test_results = test_different_parameters()
    
    return generator, performance, test_results

if __name__ == "__main__":
    generator, performance, test_results = main()

