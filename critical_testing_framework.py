#!/usr/bin/env python3
"""
إطار اختبار شامل لنظرية الفتائل وحل فرضية ريمان
تقييم علمي صارم للادعاءات والنتائج
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import time
import warnings
warnings.filterwarnings('ignore')

class RiemannHypothesisValidator:
    """فئة للتحقق من صحة حل فرضية ريمان"""
    
    def __init__(self):
        self.known_zeros = [
            14.1347251417346937904572519835625,
            21.0220396387715549926284795938969,
            25.0108575801456887632137909925628,
            30.4248761258595132103118975305239,
            32.9350615877391896906623689640542
        ]
        self.test_results = {}
        
    def test_dimensional_consistency(self) -> Dict[str, Any]:
        """اختبار الاتساق الأبعادي للمعادلات"""
        print("🔍 اختبار الاتساق الأبعادي...")
        
        results = {}
        
        # اختبار معادلة كتلة الفتيلة
        h = 6.62607015e-34  # [M L² T⁻¹]
        c = 299792458       # [L T⁻¹]
        
        # المعادلة الأصلية: m₀ = h/(4πc²)
        # الأبعاد: [M L² T⁻¹] / [L² T⁻²] = [M T] ❌
        original_dims = "[M T]"  # خطأ أبعادي
        
        # المعادلة المصححة: m₀ = h/(4πc·r₀)
        # نحتاج r₀ بوحدة [L] لنحصل على [M]
        r0_needed = "[L]"
        corrected_dims = "[M]"  # صحيح أبعادياً
        
        results['mass_equation'] = {
            'original_correct': False,
            'original_dimensions': original_dims,
            'corrected_correct': True,
            'corrected_dimensions': corrected_dims,
            'r0_requirement': r0_needed
        }
        
        return results
    
    def test_rlc_consistency(self) -> Dict[str, Any]:
        """اختبار اتساق معادلات دوائر RLC"""
        print("🔍 اختبار اتساق دوائر RLC...")
        
        results = {}
        test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        for p in test_primes:
            R = np.sqrt(p)
            L = 1 / (4 * p**(3/2))
            C = 1 / np.sqrt(p)
            
            # شرط الرنين: ω₀ = 1/√(LC)
            omega_0 = 1 / np.sqrt(L * C)
            expected_omega = 2 * p
            
            error = abs(omega_0 - expected_omega) / expected_omega
            
            results[f'prime_{p}'] = {
                'R': R,
                'L': L,
                'C': C,
                'omega_calculated': omega_0,
                'omega_expected': expected_omega,
                'relative_error': error,
                'consistent': error < 1e-10
            }
        
        # إحصائيات عامة
        all_consistent = all(results[key]['consistent'] for key in results)
        avg_error = np.mean([results[key]['relative_error'] for key in results])
        
        results['summary'] = {
            'all_consistent': all_consistent,
            'average_error': avg_error,
            'max_error': max([results[key]['relative_error'] for key in results])
        }
        
        return results
    
    def test_vector_path_closure(self, s: complex, max_terms: int = 10000) -> Dict[str, Any]:
        """اختبار إغلاق المسار المتجهي"""
        print(f"🔍 اختبار إغلاق المسار عند s = {s}...")
        
        current_pos = 0 + 0j
        path = [current_pos]
        
        for n in range(1, max_terms + 1):
            term = 1 / (n**s)
            current_pos += term
            path.append(current_pos)
            
            # توقف مبكر إذا تباعد المسار كثيراً
            if abs(current_pos) > 1000:
                break
        
        final_position = path[-1]
        closure_error = abs(final_position)
        
        return {
            'final_position': final_position,
            'closure_error': closure_error,
            'path_length': len(path),
            'converged': closure_error < 1e-3,
            'path_sample': path[::len(path)//10] if len(path) > 10 else path
        }
    
    def test_known_zeros(self) -> Dict[str, Any]:
        """اختبار الأصفار المعروفة لدالة زيتا"""
        print("🔍 اختبار الأصفار المعروفة...")
        
        results = {}
        
        for i, t in enumerate(self.known_zeros):
            s = 0.5 + 1j * t
            
            # اختبار إغلاق المسار
            path_result = self.test_vector_path_closure(s)
            
            # اختبار أنها على الخط الحرج
            on_critical_line = abs(s.real - 0.5) < 1e-10
            
            results[f'zero_{i+1}'] = {
                't': t,
                's': s,
                'on_critical_line': on_critical_line,
                'path_closure': path_result,
                'valid_zero': path_result['converged'] and on_critical_line
            }
        
        # إحصائيات
        valid_count = sum(1 for key in results if results[key]['valid_zero'])
        total_count = len(results)
        
        results['summary'] = {
            'valid_zeros': valid_count,
            'total_tested': total_count,
            'success_rate': valid_count / total_count,
            'all_valid': valid_count == total_count
        }
        
        return results
    
    def test_off_critical_line(self) -> Dict[str, Any]:
        """اختبار نقاط خارج الخط الحرج"""
        print("🔍 اختبار نقاط خارج الخط الحرج...")
        
        results = {}
        test_points = [
            0.6 + 14.1347j,  # نفس t للصفر الأول لكن σ ≠ 0.5
            0.7 + 21.0220j,  # نفس t للصفر الثاني لكن σ ≠ 0.5
            0.4 + 25.0109j,  # نفس t للصفر الثالث لكن σ ≠ 0.5
            0.8 + 15.0000j,  # نقطة عشوائية
            0.3 + 20.0000j   # نقطة عشوائية أخرى
        ]
        
        for i, s in enumerate(test_points):
            path_result = self.test_vector_path_closure(s, max_terms=5000)
            
            results[f'point_{i+1}'] = {
                's': s,
                'sigma': s.real,
                't': s.imag,
                'path_closure': path_result,
                'appears_zero': path_result['converged']
            }
        
        # هل وجدنا أي "أصفار" خارج الخط الحرج؟
        false_zeros = sum(1 for key in results if results[key]['appears_zero'])
        
        results['summary'] = {
            'false_zeros_found': false_zeros,
            'total_tested': len(test_points),
            'hypothesis_violated': false_zeros > 0
        }
        
        return results
    
    def test_prime_prediction_accuracy(self) -> Dict[str, Any]:
        """اختبار دقة التنبؤ بالأعداد الأولية"""
        print("🔍 اختبار دقة التنبؤ بالأعداد الأولية...")
        
        # قائمة الأعداد الأولية المعروفة للاختبار
        known_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
                       53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
        
        def simple_prime_predictor(p):
            """خوارزمية بسيطة للتنبؤ (للاختبار)"""
            # هذه خوارزمية مبسطة - يجب استبدالها بالخوارزمية الفعلية
            gap = int(2 + np.log(p))
            candidate = p + gap
            while not self.is_prime(candidate):
                candidate += 1
            return candidate
        
        results = {}
        correct_predictions = 0
        
        for i in range(len(known_primes) - 1):
            current_prime = known_primes[i]
            actual_next = known_primes[i + 1]
            predicted_next = simple_prime_predictor(current_prime)
            
            is_correct = predicted_next == actual_next
            if is_correct:
                correct_predictions += 1
            
            results[f'prediction_{i+1}'] = {
                'current_prime': current_prime,
                'actual_next': actual_next,
                'predicted_next': predicted_next,
                'correct': is_correct,
                'gap_actual': actual_next - current_prime,
                'gap_predicted': predicted_next - current_prime
            }
        
        accuracy = correct_predictions / (len(known_primes) - 1)
        
        results['summary'] = {
            'total_predictions': len(known_primes) - 1,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'high_accuracy': accuracy > 0.9
        }
        
        return results
    
    def is_prime(self, n: int) -> bool:
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
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """تشغيل جميع الاختبارات"""
        print("🚀 بدء الاختبار الشامل لنظرية الفتائل...")
        print("=" * 60)
        
        start_time = time.time()
        
        # تشغيل جميع الاختبارات
        tests = {
            'dimensional_consistency': self.test_dimensional_consistency(),
            'rlc_consistency': self.test_rlc_consistency(),
            'known_zeros': self.test_known_zeros(),
            'off_critical_line': self.test_off_critical_line(),
            'prime_prediction': self.test_prime_prediction_accuracy()
        }
        
        end_time = time.time()
        
        # تحليل النتائج الإجمالية
        overall_results = self.analyze_overall_results(tests)
        overall_results['execution_time'] = end_time - start_time
        
        return {
            'individual_tests': tests,
            'overall_assessment': overall_results
        }
    
    def analyze_overall_results(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل النتائج الإجمالية"""
        
        # تقييم كل اختبار
        assessments = {}
        
        # الاتساق الأبعادي
        dim_test = tests['dimensional_consistency']
        assessments['dimensional'] = {
            'passed': dim_test['mass_equation']['corrected_correct'],
            'issues': ['معادلة الكتلة الأصلية لها مشكلة أبعادية'] if not dim_test['mass_equation']['original_correct'] else []
        }
        
        # اتساق RLC
        rlc_test = tests['rlc_consistency']
        assessments['rlc'] = {
            'passed': rlc_test['summary']['all_consistent'],
            'average_error': rlc_test['summary']['average_error']
        }
        
        # الأصفار المعروفة
        zeros_test = tests['known_zeros']
        assessments['known_zeros'] = {
            'passed': zeros_test['summary']['all_valid'],
            'success_rate': zeros_test['summary']['success_rate']
        }
        
        # نقاط خارج الخط الحرج
        off_line_test = tests['off_critical_line']
        assessments['off_critical'] = {
            'passed': not off_line_test['summary']['hypothesis_violated'],
            'false_zeros': off_line_test['summary']['false_zeros_found']
        }
        
        # التنبؤ بالأعداد الأولية
        prime_test = tests['prime_prediction']
        assessments['prime_prediction'] = {
            'passed': prime_test['summary']['high_accuracy'],
            'accuracy': prime_test['summary']['accuracy']
        }
        
        # التقييم الإجمالي
        passed_tests = sum(1 for test in assessments.values() if test['passed'])
        total_tests = len(assessments)
        
        return {
            'individual_assessments': assessments,
            'tests_passed': passed_tests,
            'total_tests': total_tests,
            'overall_score': passed_tests / total_tests,
            'riemann_solved': passed_tests == total_tests and assessments['known_zeros']['success_rate'] == 1.0,
            'major_issues': self.identify_major_issues(assessments)
        }
    
    def identify_major_issues(self, assessments: Dict[str, Any]) -> List[str]:
        """تحديد المشاكل الرئيسية"""
        issues = []
        
        if not assessments['dimensional']['passed']:
            issues.append("مشاكل في الاتساق الأبعادي للمعادلات")
        
        if not assessments['known_zeros']['passed']:
            issues.append("فشل في التحقق من الأصفار المعروفة")
        
        if not assessments['off_critical']['passed']:
            issues.append("وجود أصفار محتملة خارج الخط الحرج")
        
        if assessments['prime_prediction']['accuracy'] < 0.5:
            issues.append("دقة منخفضة في التنبؤ بالأعداد الأولية")
        
        return issues
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """إنتاج تقرير شامل"""
        report = []
        report.append("📊 تقرير التقييم الشامل لنظرية الفتائل")
        report.append("=" * 60)
        
        overall = results['overall_assessment']
        
        report.append(f"🎯 النتيجة الإجمالية: {overall['tests_passed']}/{overall['total_tests']} ({overall['overall_score']:.1%})")
        report.append(f"⏱️ وقت التنفيذ: {overall['execution_time']:.2f} ثانية")
        report.append("")
        
        # هل تم حل فرضية ريمان؟
        if overall['riemann_solved']:
            report.append("✅ النتيجة: تم حل فرضية ريمان بنجاح!")
        else:
            report.append("❌ النتيجة: لم يتم حل فرضية ريمان بشكل كامل")
        
        report.append("")
        
        # المشاكل الرئيسية
        if overall['major_issues']:
            report.append("⚠️ المشاكل الرئيسية:")
            for issue in overall['major_issues']:
                report.append(f"  • {issue}")
        else:
            report.append("✅ لا توجد مشاكل رئيسية")
        
        report.append("")
        
        # تفاصيل الاختبارات
        report.append("📋 تفاصيل الاختبارات:")
        
        assessments = overall['individual_assessments']
        
        for test_name, assessment in assessments.items():
            status = "✅" if assessment['passed'] else "❌"
            report.append(f"  {status} {test_name}")
            
            if test_name == 'prime_prediction':
                report.append(f"      دقة التنبؤ: {assessment['accuracy']:.1%}")
            elif test_name == 'known_zeros':
                report.append(f"      معدل النجاح: {assessment['success_rate']:.1%}")
        
        return "\n".join(report)

def main():
    """الدالة الرئيسية للاختبار"""
    validator = RiemannHypothesisValidator()
    
    print("🔬 بدء التقييم العلمي الشامل لنظرية الفتائل")
    print("=" * 60)
    
    # تشغيل الاختبارات
    results = validator.run_comprehensive_test()
    
    # إنتاج التقرير
    report = validator.generate_report(results)
    
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    
    # حفظ النتائج
    with open('/home/ubuntu/test_results.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    return results

if __name__ == "__main__":
    results = main()

