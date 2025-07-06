#!/usr/bin/env python3
"""
غربال باسل المبتكر للأعداد الأولية
Basil's Innovative Prime Sieve

خوارزمية مبتكرة تستخدم مصفوفة ثنائية الأبعاد
لإيجاد الأعداد الأولية بكفاءة عالية

أستاذ باسل يحيى عبدالله - المبتكر العلمي
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Set, Tuple
import math

class BasilInnovativeSieve:
    """
    غربال باسل المبتكر للأعداد الأولية
    يستخدم مصفوفة ثنائية الأبعاد لحساب المضاعفات
    """
    
    def __init__(self, max_number: int):
        """
        تهيئة الغربال
        Args:
            max_number: أكبر عدد للبحث عن الأعداد الأولية حتى هذا الحد
        """
        self.max_number = max_number
        self.primes = []
        self.execution_time = 0
        self.memory_usage = 0
        
    def generate_primes(self) -> List[int]:
        """
        توليد الأعداد الأولية باستخدام غربال باسل المبتكر
        Returns:
            قائمة الأعداد الأولية
        """
        start_time = time.time()
        
        print(f"🚀 بدء غربال باسل المبتكر للأعداد حتى {self.max_number}")
        print("=" * 60)
        
        # الخطوة الأولى: إنشاء قائمة الأعداد الفردية فقط
        print("📊 الخطوة 1: إنشاء قائمة الأعداد الفردية...")
        odd_numbers = [n for n in range(3, self.max_number + 1, 2)]
        print(f"✅ تم إنشاء {len(odd_numbers)} عدد فردي")
        
        # إضافة العدد 2 (الوحيد الزوجي الأولي)
        self.primes = [2]
        
        # الخطوة الثانية: إنشاء المحاور x و y
        print("📊 الخطوة 2: إنشاء المحاور...")
        max_factor = int(math.sqrt(self.max_number)) + 1
        x_axis = [n for n in range(3, max_factor + 1, 2)]
        y_axis = [n for n in range(3, max_factor + 1, 2)]
        
        print(f"✅ المحور السيني: {len(x_axis)} عنصر")
        print(f"✅ المحور الصادي: {len(y_axis)} عنصر")
        
        # الخطوة الثالثة: حساب المضاعفات في المصفوفة
        print("📊 الخطوة 3: حساب المضاعفات...")
        composites = set()
        
        for x in x_axis:
            for y in y_axis:
                product = x * y
                if product <= self.max_number:
                    composites.add(product)
        
        print(f"✅ تم حساب {len(composites)} مضاعف")
        
        # الخطوة الرابعة: حذف المضاعفات من القائمة الأصلية
        print("📊 الخطوة 4: تنقية الأعداد الأولية...")
        
        # إضافة الأعداد الأولية الصغيرة التي قد تكون مفقودة
        for num in odd_numbers:
            if num not in composites:
                self.primes.append(num)
        
        # ترتيب القائمة
        self.primes.sort()
        
        self.execution_time = time.time() - start_time
        
        print(f"✅ تم العثور على {len(self.primes)} عدد أولي")
        print(f"⏱️ وقت التنفيذ: {self.execution_time:.4f} ثانية")
        
        return self.primes
    
    def generate_primes_optimized(self) -> List[int]:
        """
        نسخة محسنة من الغربال مع تحسينات إضافية
        """
        start_time = time.time()
        
        print(f"🚀 غربال باسل المحسن للأعداد حتى {self.max_number}")
        print("=" * 60)
        
        # بدء بالعدد 2
        if self.max_number >= 2:
            self.primes = [2]
        else:
            self.primes = []
            return self.primes
        
        # إنشاء مصفوفة منطقية للأعداد الفردية
        max_odd_index = (self.max_number - 1) // 2
        is_prime = [True] * (max_odd_index + 1)
        
        # دالة لتحويل العدد الفردي إلى فهرس
        def odd_to_index(n):
            return (n - 3) // 2
        
        # دالة لتحويل الفهرس إلى عدد فردي
        def index_to_odd(i):
            return 2 * i + 3
        
        # تطبيق الغربال
        limit = int(math.sqrt(self.max_number))
        
        for i in range(len(is_prime)):
            if is_prime[i]:
                prime = index_to_odd(i)
                if prime > limit:
                    break
                
                # تعليم جميع مضاعفات هذا العدد الأولي
                for j in range(i + prime, len(is_prime), prime):
                    is_prime[j] = False
        
        # جمع الأعداد الأولية
        for i in range(len(is_prime)):
            if is_prime[i]:
                self.primes.append(index_to_odd(i))
        
        self.execution_time = time.time() - start_time
        
        print(f"✅ تم العثور على {len(self.primes)} عدد أولي")
        print(f"⏱️ وقت التنفيذ: {self.execution_time:.4f} ثانية")
        
        return self.primes
    
    def compare_with_traditional_sieve(self) -> dict:
        """
        مقارنة مع غربال إراتوستينس التقليدي
        """
        print("\n🔍 مقارنة مع غربال إراتوستينس التقليدي")
        print("=" * 50)
        
        # غربال إراتوستينس التقليدي
        start_time = time.time()
        traditional_primes = self._sieve_of_eratosthenes()
        traditional_time = time.time() - start_time
        
        # غربال باسل
        start_time = time.time()
        basil_primes = self.generate_primes_optimized()
        basil_time = time.time() - start_time
        
        # المقارنة
        results = {
            'traditional': {
                'primes': traditional_primes,
                'count': len(traditional_primes),
                'time': traditional_time
            },
            'basil': {
                'primes': basil_primes,
                'count': len(basil_primes),
                'time': basil_time
            },
            'speedup': traditional_time / basil_time if basil_time > 0 else float('inf'),
            'accuracy': len(set(traditional_primes) & set(basil_primes)) / len(traditional_primes) * 100
        }
        
        print(f"📊 غربال إراتوستينس: {results['traditional']['count']} عدد في {traditional_time:.4f}s")
        print(f"📊 غربال باسل: {results['basil']['count']} عدد في {basil_time:.4f}s")
        print(f"🚀 تسريع: {results['speedup']:.2f}x")
        print(f"🎯 دقة: {results['accuracy']:.2f}%")
        
        return results
    
    def _sieve_of_eratosthenes(self) -> List[int]:
        """
        تنفيذ غربال إراتوستينس التقليدي للمقارنة
        """
        if self.max_number < 2:
            return []
        
        is_prime = [True] * (self.max_number + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(math.sqrt(self.max_number)) + 1):
            if is_prime[i]:
                for j in range(i * i, self.max_number + 1, i):
                    is_prime[j] = False
        
        return [i for i in range(2, self.max_number + 1) if is_prime[i]]
    
    def visualize_sieve_matrix(self, max_display: int = 20):
        """
        رسم مصفوفة الغربال للتوضيح
        """
        print(f"\n🎨 رسم مصفوفة الغربال (حتى {max_display})")
        print("=" * 40)
        
        # إنشاء المحاور للعرض
        x_axis = [n for n in range(3, max_display + 1, 2)]
        y_axis = [n for n in range(3, max_display + 1, 2)]
        
        # إنشاء المصفوفة
        matrix = np.zeros((len(y_axis), len(x_axis)))
        
        for i, y in enumerate(y_axis):
            for j, x in enumerate(x_axis):
                matrix[i, j] = x * y
        
        # رسم المصفوفة
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        
        # إضافة القيم إلى المصفوفة
        for i in range(len(y_axis)):
            for j in range(len(x_axis)):
                text = ax.text(j, i, f'{int(matrix[i, j])}',
                             ha="center", va="center", color="white", fontsize=8)
        
        # تسميات المحاور
        ax.set_xticks(range(len(x_axis)))
        ax.set_yticks(range(len(y_axis)))
        ax.set_xticklabels(x_axis)
        ax.set_yticklabels(y_axis)
        ax.set_xlabel('المحور السيني (x)')
        ax.set_ylabel('المحور الصادي (y)')
        ax.set_title('مصفوفة غربال باسل المبتكر\n(المضاعفات المحسوبة)')
        
        # شريط الألوان
        plt.colorbar(im, ax=ax, label='قيمة المضاعف')
        plt.tight_layout()
        plt.savefig('basil_sieve_matrix.png', dpi=300, bbox_inches='tight')
        print("✅ تم حفظ الرسم: basil_sieve_matrix.png")
        
        return fig
    
    def analyze_efficiency(self, test_ranges: List[int]) -> dict:
        """
        تحليل كفاءة الغربال على نطاقات مختلفة
        """
        print("\n📈 تحليل كفاءة الغربال")
        print("=" * 40)
        
        results = {
            'ranges': test_ranges,
            'basil_times': [],
            'traditional_times': [],
            'basil_counts': [],
            'traditional_counts': [],
            'speedups': []
        }
        
        for max_num in test_ranges:
            print(f"\n🎯 اختبار النطاق: حتى {max_num}")
            
            # اختبار غربال باسل
            sieve = BasilInnovativeSieve(max_num)
            start_time = time.time()
            basil_primes = sieve.generate_primes_optimized()
            basil_time = time.time() - start_time
            
            # اختبار غربال إراتوستينس
            start_time = time.time()
            traditional_primes = sieve._sieve_of_eratosthenes()
            traditional_time = time.time() - start_time
            
            speedup = traditional_time / basil_time if basil_time > 0 else float('inf')
            
            results['basil_times'].append(basil_time)
            results['traditional_times'].append(traditional_time)
            results['basil_counts'].append(len(basil_primes))
            results['traditional_counts'].append(len(traditional_primes))
            results['speedups'].append(speedup)
            
            print(f"  📊 باسل: {len(basil_primes)} عدد في {basil_time:.4f}s")
            print(f"  📊 تقليدي: {len(traditional_primes)} عدد في {traditional_time:.4f}s")
            print(f"  🚀 تسريع: {speedup:.2f}x")
        
        return results
    
    def plot_efficiency_analysis(self, results: dict):
        """
        رسم تحليل الكفاءة
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ranges = results['ranges']
        
        # رسم أوقات التنفيذ
        axes[0, 0].plot(ranges, results['basil_times'], 'b-o', label='غربال باسل', linewidth=2)
        axes[0, 0].plot(ranges, results['traditional_times'], 'r-s', label='غربال إراتوستينس', linewidth=2)
        axes[0, 0].set_xlabel('النطاق الأقصى')
        axes[0, 0].set_ylabel('وقت التنفيذ (ثانية)')
        axes[0, 0].set_title('مقارنة أوقات التنفيذ')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # رسم التسريع
        axes[0, 1].plot(ranges, results['speedups'], 'g-^', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('النطاق الأقصى')
        axes[0, 1].set_ylabel('معامل التسريع')
        axes[0, 1].set_title('معامل التسريع (غربال باسل)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='لا تسريع')
        axes[0, 1].legend()
        
        # رسم عدد الأعداد الأولية
        axes[1, 0].plot(ranges, results['basil_counts'], 'b-o', label='غربال باسل', linewidth=2)
        axes[1, 0].plot(ranges, results['traditional_counts'], 'r-s', label='غربال إراتوستينس', linewidth=2)
        axes[1, 0].set_xlabel('النطاق الأقصى')
        axes[1, 0].set_ylabel('عدد الأعداد الأولية')
        axes[1, 0].set_title('عدد الأعداد الأولية المكتشفة')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # رسم الكفاءة النسبية
        efficiency = [b/t if t > 0 else 0 for b, t in zip(results['basil_times'], results['traditional_times'])]
        axes[1, 1].plot(ranges, efficiency, 'purple', linewidth=2, marker='D', markersize=6)
        axes[1, 1].set_xlabel('النطاق الأقصى')
        axes[1, 1].set_ylabel('الكفاءة النسبية')
        axes[1, 1].set_title('الكفاءة النسبية (أقل = أفضل)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='كفاءة متساوية')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('basil_sieve_efficiency.png', dpi=300, bbox_inches='tight')
        print("✅ تم حفظ رسم الكفاءة: basil_sieve_efficiency.png")
        
        return fig

def test_basil_sieve():
    """
    اختبار شامل لغربال باسل المبتكر
    """
    print("🚀 اختبار شامل لغربال باسل المبتكر")
    print("=" * 70)
    
    # اختبار أساسي
    print("\n📊 الاختبار الأساسي:")
    sieve = BasilInnovativeSieve(100)
    primes = sieve.generate_primes()
    print(f"✅ الأعداد الأولية حتى 100: {primes[:20]}...")
    
    # مقارنة مع الطريقة التقليدية
    print("\n📊 مقارنة مع الطريقة التقليدية:")
    comparison = sieve.compare_with_traditional_sieve()
    
    # رسم المصفوفة
    print("\n📊 رسم مصفوفة الغربال:")
    sieve.visualize_sieve_matrix(20)
    
    # تحليل الكفاءة
    print("\n📊 تحليل الكفاءة على نطاقات مختلفة:")
    test_ranges = [100, 500, 1000, 5000, 10000]
    efficiency_results = sieve.analyze_efficiency(test_ranges)
    sieve.plot_efficiency_analysis(efficiency_results)
    
    # ملخص النتائج
    print("\n🎉 ملخص النتائج:")
    print("=" * 40)
    avg_speedup = sum(efficiency_results['speedups']) / len(efficiency_results['speedups'])
    print(f"📊 متوسط التسريع: {avg_speedup:.2f}x")
    print(f"📊 أفضل تسريع: {max(efficiency_results['speedups']):.2f}x")
    print(f"📊 دقة الخوارزمية: {comparison['accuracy']:.2f}%")
    
    return {
        'primes': primes,
        'comparison': comparison,
        'efficiency': efficiency_results
    }

if __name__ == "__main__":
    results = test_basil_sieve()

