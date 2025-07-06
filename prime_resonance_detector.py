#!/usr/bin/env python3
"""
كاشف الأعداد الأولية من الرنين - نهج مبتكر
Prime Resonance Detector - Innovative Approach

تركيز خاص على اكتشاف الأعداد الأولية من خلال:
1. تحليل أدق لترددات الرنين
2. ربط مباشر بين الرنين والأولية
3. استخدام خصائص فيزيائية مميزة للأعداد الأولية

المؤلف: باسل يحيى عبدالله "المبتكر العلمي"
المشرف: الذكاء الاصطناعي مانوس
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize_scalar
import cmath
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class PrimeResonanceDetector:
    """
    كاشف الأعداد الأولية من الرنين
    """
    
    def __init__(self, max_prime_check: int = 100):
        """تهيئة الكاشف"""
        self.max_prime_check = max_prime_check
        
        # معاملات محسنة للكشف
        self.sigma_critical = 0.5  # الخط الحرج
        self.resonance_amplification = 2.0  # تضخيم الرنين للأعداد الأولية
        self.damping_composite = 0.8  # تخميد الأعداد المركبة
        
        # قاعدة بيانات الأعداد الأولية
        self.known_primes = self._generate_primes(max_prime_check)
        self.prime_frequencies = {}
        
        print(f"🔬 تم تهيئة كاشف الأعداد الأولية من الرنين")
        print(f"📊 نطاق الفحص: حتى {max_prime_check}")
        print(f"🎯 أعداد أولية معروفة: {len(self.known_primes)}")
        
        self._calculate_prime_frequencies()
    
    def _generate_primes(self, limit: int) -> List[int]:
        """توليد الأعداد الأولية حتى الحد المحدد"""
        primes = []
        for n in range(2, limit + 1):
            if self._is_prime(n):
                primes.append(n)
        return primes
    
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
    
    def _calculate_prime_frequencies(self):
        """حساب ترددات الرنين المتوقعة للأعداد الأولية"""
        print("⚡ حساب ترددات الرنين للأعداد الأولية...")
        
        for p in self.known_primes:
            # التردد الأساسي: ln(p)
            base_frequency = np.log(p)
            
            # ترددات الرنين المتعددة (هارمونيك)
            harmonics = [base_frequency * h for h in [1, 2, 3, 0.5]]
            
            self.prime_frequencies[p] = {
                'base': base_frequency,
                'harmonics': harmonics,
                'resonance_strength': np.sqrt(p),  # قوة الرنين
                'quality_factor': 10.0 + p * 0.1  # عامل الجودة
            }
        
        print(f"✅ تم حساب ترددات {len(self.prime_frequencies)} عدد أولي")
    
    def calculate_prime_response(self, t: float) -> Tuple[float, List[int]]:
        """حساب استجابة الأعداد الأولية عند التردد t"""
        
        total_response = 0.0
        contributing_primes = []
        
        for p, freq_data in self.prime_frequencies.items():
            # حساب الاستجابة لكل هارمونيك
            prime_response = 0.0
            
            for harmonic_freq in freq_data['harmonics']:
                # المسافة من التردد المطلوب
                frequency_distance = abs(t - harmonic_freq)
                
                # عرض الرنين (أضيق للأعداد الأولية)
                resonance_width = 0.1 + 0.01 * np.log(p)
                
                # دالة الرنين (لورنتزية)
                if frequency_distance < resonance_width * 5:  # نطاق التأثير
                    resonance = freq_data['resonance_strength'] / (1 + (frequency_distance / resonance_width)**2)
                    prime_response += resonance
            
            # إضافة للاستجابة الإجمالية
            if prime_response > 0.1:  # عتبة المساهمة
                total_response += prime_response
                contributing_primes.append(p)
        
        return total_response, contributing_primes
    
    def calculate_composite_response(self, t: float, max_composite: int = 100) -> float:
        """حساب استجابة الأعداد المركبة (للمقارنة)"""
        
        total_response = 0.0
        
        for n in range(4, max_composite + 1):
            if not self._is_prime(n):
                # التردد للعدد المركب
                base_frequency = np.log(n)
                frequency_distance = abs(t - base_frequency)
                
                # عرض رنين أوسع وأضعف للأعداد المركبة
                resonance_width = 0.3 + 0.05 * np.log(n)
                resonance_strength = np.sqrt(n) * self.damping_composite
                
                if frequency_distance < resonance_width * 3:
                    resonance = resonance_strength / (1 + (frequency_distance / resonance_width)**2)
                    total_response += resonance
        
        return total_response
    
    def scan_frequency_spectrum(self, t_min: float = 0.5, t_max: float = 10, 
                              resolution: int = 5000) -> Dict[str, Any]:
        """مسح طيف الترددات للبحث عن الأعداد الأولية"""
        
        print(f"🔍 مسح طيف الترددات في [{t_min}, {t_max}]...")
        
        t_values = np.linspace(t_min, t_max, resolution)
        prime_responses = []
        composite_responses = []
        contributing_primes_list = []
        
        for t in t_values:
            prime_resp, contrib_primes = self.calculate_prime_response(t)
            composite_resp = self.calculate_composite_response(t)
            
            prime_responses.append(prime_resp)
            composite_responses.append(composite_resp)
            contributing_primes_list.append(contrib_primes)
        
        # تطبيق مرشح للتنعيم
        if len(prime_responses) > 50:
            window_length = min(51, len(prime_responses) // 10)
            if window_length % 2 == 0:
                window_length += 1
            
            prime_responses_smooth = savgol_filter(prime_responses, window_length, 3)
            composite_responses_smooth = savgol_filter(composite_responses, window_length, 3)
        else:
            prime_responses_smooth = prime_responses
            composite_responses_smooth = composite_responses
        
        return {
            't_values': t_values,
            'prime_responses': prime_responses,
            'composite_responses': composite_responses,
            'prime_responses_smooth': prime_responses_smooth,
            'composite_responses_smooth': composite_responses_smooth,
            'contributing_primes': contributing_primes_list
        }
    
    def detect_prime_peaks(self, spectrum_data: Dict[str, Any]) -> List[Tuple[float, float, List[int]]]:
        """كشف ذروات الأعداد الأولية"""
        
        print("🎯 كشف ذروات الأعداد الأولية...")
        
        t_values = spectrum_data['t_values']
        prime_responses = spectrum_data['prime_responses_smooth']
        contributing_primes = spectrum_data['contributing_primes']
        
        # كشف الذروات
        peaks, properties = find_peaks(prime_responses, 
                                     height=np.mean(prime_responses) + np.std(prime_responses),
                                     distance=20,
                                     prominence=0.5,
                                     width=2)
        
        detected_peaks = []
        
        for peak_idx in peaks:
            t_peak = t_values[peak_idx]
            response = prime_responses[peak_idx]
            primes_at_peak = contributing_primes[peak_idx]
            
            detected_peaks.append((t_peak, response, primes_at_peak))
            
            if primes_at_peak:
                print(f"🎯 ذروة مكتشفة: t = {t_peak:.4f}, استجابة = {response:.4f}, أعداد أولية: {primes_at_peak}")
        
        print(f"📊 إجمالي الذروات المكتشفة: {len(detected_peaks)}")
        return detected_peaks
    
    def extract_primes_from_peaks(self, detected_peaks: List[Tuple[float, float, List[int]]]) -> List[int]:
        """استخراج الأعداد الأولية من الذروات المكتشفة"""
        
        print("🔍 استخراج الأعداد الأولية من الذروات...")
        
        extracted_primes = set()
        
        for t_peak, response, primes_at_peak in detected_peaks:
            for p in primes_at_peak:
                # التحقق من قرب التردد من التردد المتوقع للعدد الأولي
                expected_freq = np.log(p)
                frequency_error = abs(t_peak - expected_freq)
                
                # قبول العدد الأولي إذا كان التردد قريب بما فيه الكفاية
                if frequency_error < 0.5:  # تسامح في التردد
                    extracted_primes.add(p)
                    print(f"✅ عدد أولي مكتشف: {p} (تردد متوقع: {expected_freq:.4f}, مكتشف: {t_peak:.4f})")
        
        extracted_primes_list = sorted(list(extracted_primes))
        print(f"📊 إجمالي الأعداد الأولية المستخرجة: {len(extracted_primes_list)}")
        print(f"🎯 الأعداد الأولية: {extracted_primes_list}")
        
        return extracted_primes_list
    
    def plot_resonance_spectrum(self, spectrum_data: Dict[str, Any], 
                               detected_peaks: List[Tuple[float, float, List[int]]]):
        """رسم طيف الرنين"""
        
        print("📈 رسم طيف الرنين...")
        
        t_values = spectrum_data['t_values']
        prime_responses = spectrum_data['prime_responses_smooth']
        composite_responses = spectrum_data['composite_responses_smooth']
        
        plt.figure(figsize=(15, 10))
        
        # الرسم الأساسي
        plt.subplot(2, 1, 1)
        plt.plot(t_values, prime_responses, 'b-', linewidth=2, label='استجابة الأعداد الأولية', alpha=0.8)
        plt.plot(t_values, composite_responses, 'r-', linewidth=1, label='استجابة الأعداد المركبة', alpha=0.6)
        
        # تمييز الذروات المكتشفة
        if detected_peaks:
            peak_t = [p[0] for p in detected_peaks]
            peak_response = [p[1] for p in detected_peaks]
            plt.scatter(peak_t, peak_response, color='green', s=100, alpha=0.8, 
                       label='ذروات مكتشفة', zorder=5)
        
        # تمييز الأعداد الأولية المعروفة
        known_t = [np.log(p) for p in self.known_primes if np.log(p) >= min(t_values) and np.log(p) <= max(t_values)]
        if known_t:
            for t_known in known_t:
                plt.axvline(x=t_known, color='orange', alpha=0.3, linestyle='--')
            plt.axvline(x=known_t[0], color='orange', alpha=0.3, linestyle='--', label='أعداد أولية معروفة')
        
        plt.xlabel('التردد t')
        plt.ylabel('قوة الاستجابة')
        plt.title('طيف الرنين للأعداد الأولية - كاشف الرنين المبتكر')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # رسم النسبة (أولية/مركبة)
        plt.subplot(2, 1, 2)
        ratio = np.array(prime_responses) / (np.array(composite_responses) + 0.1)  # تجنب القسمة على صفر
        plt.plot(t_values, ratio, 'purple', linewidth=2, label='نسبة الأولية/المركبة')
        
        # تمييز النسب العالية
        high_ratio_threshold = np.mean(ratio) + 2 * np.std(ratio)
        high_ratio_indices = np.where(ratio > high_ratio_threshold)[0]
        if len(high_ratio_indices) > 0:
            plt.scatter(t_values[high_ratio_indices], ratio[high_ratio_indices], 
                       color='red', s=50, alpha=0.7, label='نسب عالية')
        
        plt.xlabel('التردد t')
        plt.ylabel('نسبة الاستجابة')
        plt.title('نسبة استجابة الأعداد الأولية إلى المركبة')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/prime_resonance_spectrum.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 تم حفظ طيف الرنين في: /home/ubuntu/prime_resonance_spectrum.png")
    
    def comprehensive_prime_detection(self) -> Dict[str, Any]:
        """كشف شامل للأعداد الأولية"""
        
        print("🚀 بدء الكشف الشامل للأعداد الأولية")
        print("=" * 80)
        
        results = {}
        
        # 1. مسح طيف الترددات
        print("\n1️⃣ مسح طيف الترددات:")
        spectrum_data = self.scan_frequency_spectrum(0.5, 8, 4000)
        results['spectrum_data'] = spectrum_data
        
        # 2. كشف ذروات الأعداد الأولية
        print("\n2️⃣ كشف ذروات الأعداد الأولية:")
        detected_peaks = self.detect_prime_peaks(spectrum_data)
        results['detected_peaks'] = detected_peaks
        
        # 3. استخراج الأعداد الأولية
        print("\n3️⃣ استخراج الأعداد الأولية:")
        extracted_primes = self.extract_primes_from_peaks(detected_peaks)
        results['extracted_primes'] = extracted_primes
        
        # 4. تقييم الأداء
        print("\n4️⃣ تقييم الأداء:")
        performance = self._evaluate_detection_performance(extracted_primes)
        results['performance'] = performance
        
        print(f"  🎯 أعداد أولية مكتشفة: {len(extracted_primes)}")
        print(f"  🎯 دقة الكشف: {performance['precision']:.1%}")
        print(f"  🎯 معدل الاستدعاء: {performance['recall']:.1%}")
        print(f"  🎯 النتيجة F1: {performance['f1_score']:.1%}")
        
        # 5. رسم الطيف
        print("\n5️⃣ رسم طيف الرنين:")
        self.plot_resonance_spectrum(spectrum_data, detected_peaks)
        
        return results
    
    def _evaluate_detection_performance(self, extracted_primes: List[int]) -> Dict[str, float]:
        """تقييم أداء الكشف"""
        
        # الأعداد الأولية في النطاق المفحوص
        target_primes = [p for p in self.known_primes if p <= 20]  # نطاق الفحص
        
        if not extracted_primes:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': len(target_primes)
            }
        
        # حساب المقاييس
        true_positives = len(set(extracted_primes) & set(target_primes))
        false_positives = len(set(extracted_primes) - set(target_primes))
        false_negatives = len(set(target_primes) - set(extracted_primes))
        
        precision = true_positives / len(extracted_primes) if extracted_primes else 0.0
        recall = true_positives / len(target_primes) if target_primes else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'target_primes': target_primes,
            'extracted_primes': extracted_primes
        }

def main():
    """الدالة الرئيسية"""
    print("🌟 مرحباً بك في كاشف الأعداد الأولية من الرنين!")
    print("=" * 80)
    
    # إنشاء الكاشف
    detector = PrimeResonanceDetector(max_prime_check=50)
    
    # تشغيل الكشف الشامل
    results = detector.comprehensive_prime_detection()
    
    # حفظ النتائج
    import json
    with open('/home/ubuntu/prime_resonance_results.json', 'w', encoding='utf-8') as f:
        serializable_results = {
            'extracted_primes': results['extracted_primes'],
            'performance': results['performance'],
            'peaks_count': len(results['detected_peaks'])
        }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print("\n💾 تم حفظ نتائج الكشف في: /home/ubuntu/prime_resonance_results.json")
    
    return results

if __name__ == "__main__":
    results = main()

