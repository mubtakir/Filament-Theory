#!/usr/bin/env python3
"""
نموذج الانتروبيا الفتيلية المتقدم
دمج قوانين الانتروبيا في نظرية الفتائل مع البناء اللاحتمي والانهيار الفجائي

تطوير: باسل يحيى عبدالله
تحليل وبرمجة: مساعد ذكي متخصص
التاريخ: 7 يناير 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
from scipy.integrate import odeint
import json
from datetime import datetime

class FilamentEntropyModel:
    """
    نموذج الانتروبيا الفتيلية المتكامل
    يدمج البناء اللاحتمي والانهيار الفجائي مع قوانين الانتروبيا
    """
    
    def __init__(self, params=None):
        """تهيئة النموذج مع المعاملات الافتراضية"""
        
        # المعاملات الافتراضية
        self.default_params = {
            # معاملات البناء اللاحتمي
            'lambda_base': 1.0,          # معدل القفزات الأساسي
            'alpha_weibull': 2.0,        # معامل شكل ويبل
            'beta_weibull': 1.0,         # معامل مقياس ويبل
            'phi_critical': 10.0,        # النقطة الحرجة للانهيار
            
            # معاملات الانهيار الفجائي
            'collapse_rate': 1000.0,     # معدل الانهيار
            'collapse_threshold': 0.99,  # عتبة الانهيار
            
            # معاملات الانتروبيا
            'k_boltzmann': 1.0,          # ثابت بولتزمان (مقياس)
            'entropy_coupling': 0.1,     # قوة ربط الانتروبيا
            'max_entropy': 100.0,        # الانتروبيا القصوى
            
            # معاملات التغذية الراجعة
            'feedback_strength': 0.5,    # قوة التغذية الراجعة
            'memory_decay': 0.1,         # تلاشي الذاكرة
            
            # معاملات البيئة
            'noise_level': 0.01,         # مستوى الضوضاء
            'temperature': 1.0,          # درجة الحرارة الفعالة
        }
        
        # دمج المعاملات المخصصة
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
        
        # متغيرات الحالة
        self.reset_state()
    
    def reset_state(self):
        """إعادة تعيين حالة النموذج"""
        self.time = 0.0
        self.phi = 0.0
        self.entropy = 0.0
        self.history = {
            'time': [],
            'phi': [],
            'entropy': [],
            'jumps': [],
            'collapses': []
        }
        self.last_jump_time = 0.0
        self.total_jumps = 0
        self.total_collapses = 0
    
    def calculate_entropy(self, phi, dphi_dt):
        """
        حساب الانتروبيا الفتيلية
        
        Args:
            phi: قيمة الفتيلة الحالية
            dphi_dt: معدل تغير الفتيلة
        
        Returns:
            entropy: الانتروبيا الحالية
        """
        if phi <= 0:
            return 0.0
        
        # الانتروبيا الأساسية (تعتمد على التعقد)
        base_entropy = self.params['k_boltzmann'] * np.log(1 + phi)
        
        # تأثير معدل التغيير
        if dphi_dt > 0:  # البناء
            # الانتروبيا تزداد أثناء البناء
            change_entropy = self.params['entropy_coupling'] * dphi_dt
        else:  # الانهيار
            # الانتروبيا تقل أثناء الانهيار (محلياً)
            change_entropy = -self.params['entropy_coupling'] * abs(dphi_dt) * 10
        
        # الانتروبيا الحرارية
        thermal_entropy = self.params['temperature'] * np.sqrt(phi)
        
        # الانتروبيا الكلية
        total_entropy = base_entropy + change_entropy + thermal_entropy
        
        # تطبيق الحدود
        return max(0, min(total_entropy, self.params['max_entropy']))
    
    def calculate_jump_intensity(self, phi, entropy, time):
        """
        حساب شدة القفزات بناءً على الحالة الحالية
        
        Args:
            phi: قيمة الفتيلة
            entropy: الانتروبيا الحالية
            time: الزمن الحالي
        
        Returns:
            intensity: شدة القفزات
        """
        # الشدة الأساسية
        base_intensity = self.params['lambda_base']
        
        # تأثير الحالة الحالية (تغذية راجعة)
        state_factor = 1 + self.params['feedback_strength'] * (phi / self.params['phi_critical'])
        
        # تأثير الانتروبيا
        entropy_factor = 1 + 0.1 * (entropy / self.params['max_entropy'])
        
        # تأثير الذاكرة (الزمن منذ آخر قفزة)
        time_since_jump = time - self.last_jump_time
        memory_factor = 1 + self.params['memory_decay'] * time_since_jump
        
        # الضوضاء البيئية
        noise_factor = 1 + self.params['noise_level'] * np.random.normal(0, 1)
        
        # الشدة الكلية
        total_intensity = base_intensity * state_factor * entropy_factor * memory_factor * noise_factor
        
        return max(0, total_intensity)
    
    def generate_jump(self, phi, entropy):
        """
        توليد قفزة بناء جديدة
        
        Args:
            phi: قيمة الفتيلة الحالية
            entropy: الانتروبيا الحالية
        
        Returns:
            jump_size: حجم القفزة
        """
        # توليد حجم القفزة من توزيع ويبل
        base_size = np.random.weibull(self.params['alpha_weibull']) * self.params['beta_weibull']
        
        # تعديل الحجم بناءً على الحالة
        state_modifier = 1 + 0.5 * (phi / self.params['phi_critical'])
        entropy_modifier = 1 + 0.2 * (entropy / self.params['max_entropy'])
        
        # الحجم النهائي
        jump_size = base_size * state_modifier * entropy_modifier
        
        return max(0, jump_size)
    
    def check_collapse_condition(self, phi):
        """
        فحص شرط الانهيار الفجائي
        
        Args:
            phi: قيمة الفتيلة الحالية
        
        Returns:
            should_collapse: هل يجب الانهيار
        """
        # الشرط الأساسي: الوصول للنقطة الحرجة
        critical_reached = phi >= self.params['phi_critical']
        
        # شرط إضافي: احتمالية الانهيار
        collapse_probability = max(0, (phi - self.params['phi_critical'] * self.params['collapse_threshold']) / 
                                     (self.params['phi_critical'] * (1 - self.params['collapse_threshold'])))
        
        random_trigger = np.random.random() < collapse_probability
        
        return critical_reached and random_trigger
    
    def execute_collapse(self, phi, entropy):
        """
        تنفيذ الانهيار الفجائي
        
        Args:
            phi: قيمة الفتيلة قبل الانهيار
            entropy: الانتروبيا قبل الانهيار
        
        Returns:
            new_phi: قيمة الفتيلة بعد الانهيار (صفر)
            entropy_released: الانتروبيا المحررة
        """
        # الطاقة المحررة (تتناسب مع مربع الفتيلة)
        energy_released = 0.5 * phi**2
        
        # الانتروبيا المحررة للبيئة
        entropy_released = entropy + energy_released / self.params['temperature']
        
        # تسجيل الانهيار
        self.total_collapses += 1
        self.history['collapses'].append({
            'time': self.time,
            'phi_before': phi,
            'entropy_before': entropy,
            'energy_released': energy_released,
            'entropy_released': entropy_released
        })
        
        return 0.0, entropy_released
    
    def simulate_step(self, dt):
        """
        محاكاة خطوة زمنية واحدة
        
        Args:
            dt: الخطوة الزمنية
        
        Returns:
            phi: قيمة الفتيلة الجديدة
            entropy: الانتروبيا الجديدة
        """
        # حساب شدة القفزات الحالية
        jump_intensity = self.calculate_jump_intensity(self.phi, self.entropy, self.time)
        
        # احتمالية حدوث قفزة في هذه الخطوة
        jump_probability = 1 - np.exp(-jump_intensity * dt)
        
        dphi_dt = 0.0
        
        # فحص حدوث قفزة
        if np.random.random() < jump_probability:
            # توليد قفزة
            jump_size = self.generate_jump(self.phi, self.entropy)
            self.phi += jump_size
            dphi_dt = jump_size / dt
            
            # تسجيل القفزة
            self.last_jump_time = self.time
            self.total_jumps += 1
            self.history['jumps'].append({
                'time': self.time,
                'size': jump_size,
                'phi_after': self.phi
            })
        
        # فحص شرط الانهيار
        if self.check_collapse_condition(self.phi):
            # تنفيذ الانهيار الفجائي
            old_phi = self.phi
            self.phi, entropy_released = self.execute_collapse(self.phi, self.entropy)
            dphi_dt = -(old_phi) / dt  # معدل انهيار سالب كبير
        
        # حساب الانتروبيا الجديدة
        self.entropy = self.calculate_entropy(self.phi, dphi_dt)
        
        # تحديث الزمن
        self.time += dt
        
        # تسجيل التاريخ
        self.history['time'].append(self.time)
        self.history['phi'].append(self.phi)
        self.history['entropy'].append(self.entropy)
        
        return self.phi, self.entropy
    
    def simulate(self, total_time, dt=0.01):
        """
        تشغيل المحاكاة الكاملة
        
        Args:
            total_time: الزمن الكلي للمحاكاة
            dt: الخطوة الزمنية
        
        Returns:
            results: نتائج المحاكاة
        """
        # إعادة تعيين الحالة
        self.reset_state()
        
        # تشغيل المحاكاة
        steps = int(total_time / dt)
        
        print(f"بدء المحاكاة: {steps} خطوة، الزمن الكلي = {total_time}")
        
        for i in range(steps):
            self.simulate_step(dt)
            
            # طباعة التقدم
            if i % (steps // 10) == 0:
                progress = (i / steps) * 100
                print(f"التقدم: {progress:.1f}% - الفتيلة: {self.phi:.3f}, الانتروبيا: {self.entropy:.3f}")
        
        # إعداد النتائج
        results = {
            'parameters': self.params,
            'final_state': {
                'time': self.time,
                'phi': self.phi,
                'entropy': self.entropy,
                'total_jumps': self.total_jumps,
                'total_collapses': self.total_collapses
            },
            'history': self.history,
            'statistics': self.calculate_statistics()
        }
        
        print(f"انتهت المحاكاة - القفزات: {self.total_jumps}, الانهيارات: {self.total_collapses}")
        
        return results
    
    def calculate_statistics(self):
        """حساب الإحصائيات الأساسية"""
        if len(self.history['time']) == 0:
            return {}
        
        phi_values = np.array(self.history['phi'])
        entropy_values = np.array(self.history['entropy'])
        
        # إحصائيات الفتيلة
        phi_stats = {
            'mean': np.mean(phi_values),
            'std': np.std(phi_values),
            'max': np.max(phi_values),
            'min': np.min(phi_values)
        }
        
        # إحصائيات الانتروبيا
        entropy_stats = {
            'mean': np.mean(entropy_values),
            'std': np.std(entropy_values),
            'max': np.max(entropy_values),
            'min': np.min(entropy_values)
        }
        
        # إحصائيات القفزات
        if self.history['jumps']:
            jump_sizes = [j['size'] for j in self.history['jumps']]
            jump_times = [j['time'] for j in self.history['jumps']]
            
            # أزمنة الانتظار
            waiting_times = []
            for i in range(1, len(jump_times)):
                waiting_times.append(jump_times[i] - jump_times[i-1])
            
            jump_stats = {
                'total_jumps': len(jump_sizes),
                'mean_size': np.mean(jump_sizes),
                'std_size': np.std(jump_sizes),
                'mean_waiting_time': np.mean(waiting_times) if waiting_times else 0,
                'std_waiting_time': np.std(waiting_times) if waiting_times else 0
            }
        else:
            jump_stats = {'total_jumps': 0}
        
        # إحصائيات الانهيارات
        collapse_stats = {
            'total_collapses': len(self.history['collapses']),
            'collapse_rate': len(self.history['collapses']) / self.time if self.time > 0 else 0
        }
        
        return {
            'phi': phi_stats,
            'entropy': entropy_stats,
            'jumps': jump_stats,
            'collapses': collapse_stats
        }
    
    def plot_results(self, results, save_path=None):
        """رسم نتائج المحاكاة"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('نتائج محاكاة النموذج الفتيلي مع الانتروبيا', fontsize=16)
        
        times = results['history']['time']
        phi_values = results['history']['phi']
        entropy_values = results['history']['entropy']
        
        # الرسم الأول: تطور الفتيلة مع الزمن
        axes[0, 0].plot(times, phi_values, 'b-', linewidth=2, label='الفتيلة')
        axes[0, 0].axhline(y=self.params['phi_critical'], color='r', linestyle='--', label='النقطة الحرجة')
        axes[0, 0].set_xlabel('الزمن')
        axes[0, 0].set_ylabel('قيمة الفتيلة')
        axes[0, 0].set_title('تطور الفتيلة مع الزمن')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # الرسم الثاني: تطور الانتروبيا مع الزمن
        axes[0, 1].plot(times, entropy_values, 'g-', linewidth=2, label='الانتروبيا')
        axes[0, 1].set_xlabel('الزمن')
        axes[0, 1].set_ylabel('الانتروبيا')
        axes[0, 1].set_title('تطور الانتروبيا مع الزمن')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # الرسم الثالث: العلاقة بين الفتيلة والانتروبيا
        axes[1, 0].scatter(phi_values, entropy_values, alpha=0.6, s=1)
        axes[1, 0].set_xlabel('قيمة الفتيلة')
        axes[1, 0].set_ylabel('الانتروبيا')
        axes[1, 0].set_title('العلاقة بين الفتيلة والانتروبيا')
        axes[1, 0].grid(True, alpha=0.3)
        
        # الرسم الرابع: توزيع أحجام القفزات
        if results['history']['jumps']:
            jump_sizes = [j['size'] for j in results['history']['jumps']]
            axes[1, 1].hist(jump_sizes, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].set_xlabel('حجم القفزة')
            axes[1, 1].set_ylabel('التكرار')
            axes[1, 1].set_title('توزيع أحجام القفزات')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'لا توجد قفزات', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('توزيع أحجام القفزات')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"تم حفظ الرسم في: {save_path}")
        
        plt.show()
        
        return fig
    
    def save_results(self, results, filename):
        """حفظ النتائج في ملف JSON"""
        # تحويل numpy arrays إلى lists للتسلسل
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                        serializable_results[key][k] = [arr.tolist() for arr in v]
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        # إضافة معلومات التشغيل
        serializable_results['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'model_version': '1.0',
            'description': 'نتائج محاكاة النموذج الفتيلي مع الانتروبيا'
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"تم حفظ النتائج في: {filename}")

def run_comprehensive_simulation():
    """تشغيل محاكاة شاملة مع معاملات مختلفة"""
    
    print("=== محاكاة النموذج الفتيلي مع الانتروبيا ===")
    print("تطوير: باسل يحيى عبدالله")
    print("التاريخ: 7 يناير 2025")
    print()
    
    # إعداد المعاملات للاختبار
    test_scenarios = [
        {
            'name': 'السيناريو الأساسي',
            'params': {}  # استخدام المعاملات الافتراضية
        },
        {
            'name': 'بناء سريع',
            'params': {
                'lambda_base': 2.0,
                'alpha_weibull': 1.5
            }
        },
        {
            'name': 'انتروبيا عالية',
            'params': {
                'entropy_coupling': 0.3,
                'temperature': 2.0
            }
        }
    ]
    
    all_results = {}
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        # إنشاء النموذج
        model = FilamentEntropyModel(scenario['params'])
        
        # تشغيل المحاكاة
        results = model.simulate(total_time=50.0, dt=0.01)
        
        # حفظ النتائج
        all_results[scenario['name']] = results
        
        # طباعة الملخص
        stats = results['statistics']
        print(f"الإحصائيات النهائية:")
        print(f"  - متوسط الفتيلة: {stats['phi']['mean']:.3f}")
        print(f"  - متوسط الانتروبيا: {stats['entropy']['mean']:.3f}")
        print(f"  - عدد القفزات: {stats['jumps']['total_jumps']}")
        print(f"  - عدد الانهيارات: {stats['collapses']['total_collapses']}")
        
        # رسم النتائج
        model.plot_results(results, f'/home/ubuntu/filament_entropy_{scenario["name"].replace(" ", "_")}.png')
        
        # حفظ النتائج
        model.save_results(results, f'/home/ubuntu/results_{scenario["name"].replace(" ", "_")}.json')
    
    return all_results

if __name__ == "__main__":
    # تشغيل المحاكاة الشاملة
    results = run_comprehensive_simulation()
    print("\n=== انتهت جميع المحاكاات ===")

