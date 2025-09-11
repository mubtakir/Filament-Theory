#!/usr/bin/env python3
"""
محاكاة شاملة لنظرية الفتيلة وعلاقتها بالفيزياء الكونية
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
from scipy.optimize import minimize
import matplotlib.patches as patches

# إعداد الخط لدعم الرموز الرياضية
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 10

class ComprehensiveFilamentSimulation:
    """محاكاة شاملة لنظرية الفتيلة"""
    
    def __init__(self):
        # الثوابت الفيزيائية
        self.c = 299792458  # سرعة الضوء (m/s)
        self.hbar = 1.054571817e-34  # ثابت بلانك المخفض (J⋅s)
        self.G = 6.67430e-11  # ثابت الجاذبية (m³/kg⋅s²)
        self.e = 1.602176634e-19  # شحنة الإلكترون (C)
        self.m_e = 9.1093837015e-31  # كتلة الإلكترون (kg)
        
        # ثوابت كونية
        self.H0 = 70 * 1000 / 3.086e22  # ثابت هابل (1/s)
        self.rho_critical = 3 * self.H0**2 / (8 * np.pi * self.G)  # الكثافة الحرجة
        
        # نسب الكون
        self.Omega_DE = 0.68  # الطاقة المظلمة
        self.Omega_DM = 0.27  # المادة المظلمة
        self.Omega_b = 0.05   # المادة العادية
        
    def filament_LC_simulation(self, r_range=(0.1, 10.0), n_points=1000):
        """محاكاة العلاقة بين L و C في نظرية الفتيلة"""
        
        print("=== محاكاة العلاقة L-C ===")
        
        # مجال قيم نصف القطر
        r_m_values = np.linspace(r_range[0], r_range[1], n_points)
        r_s_values = 1.0 / r_m_values  # العلاقة العكسية الأساسية
        
        # معامل التناسب k (نفترض k=1 من النظرية)
        k = 1.0
        
        # حساب L و C
        L_values = k * r_m_values
        C_values = k * r_s_values
        
        # حساب حاصل الضرب والتردد
        LC_product = L_values * C_values
        omega_values = 1.0 / np.sqrt(LC_product)
        
        # التحقق من الثبات
        LC_mean = np.mean(LC_product)
        LC_std = np.std(LC_product)
        omega_mean = np.mean(omega_values)
        omega_std = np.std(omega_values)
        
        print(f"متوسط L⋅C: {LC_mean:.6f} ± {LC_std:.2e}")
        print(f"متوسط ω: {omega_mean:.6f} ± {omega_std:.2e}")
        
        # البحث عن الانحرافات
        max_deviation = np.max(np.abs(LC_product - 1.0))
        print(f"أقصى انحراف عن L⋅C = 1: {max_deviation:.2e}")
        
        return r_m_values, r_s_values, L_values, C_values, LC_product, omega_values
    
    def electron_charge_mass_correlation(self):
        """تحليل العلاقة بين شحنة الإلكترون والفتيلة"""
        
        print("\n=== تحليل شحنة الإلكترون ===")
        
        # التردد الطبيعي للإلكترون
        omega_electron = self.m_e * self.c**2 / self.hbar
        
        # نسبة الشحنة إلى الكتلة
        e_over_m = self.e / self.m_e
        
        # البحث عن علاقة مع التردد الأساسي
        # اقتراح: e/m ∝ ω₀ × f(c, ħ)
        
        # تجربة عدة نماذج
        models = {
            'Model 1: e/m ∝ ω₀⋅c²/ħ': 1.0 * self.c**2 / self.hbar,
            'Model 2: e/m ∝ ω₀⋅c/√(ħ⋅c)': 1.0 * self.c / np.sqrt(self.hbar * self.c),
            'Model 3: e/m ∝ ω₀⋅√(c³/ħ)': 1.0 * np.sqrt(self.c**3 / self.hbar),
            'Model 4: e/m ∝ ω₀⋅c/ħ': 1.0 * self.c / self.hbar
        }
        
        print(f"النسبة الفعلية e/m: {e_over_m:.3e} C/kg")
        print(f"تردد الإلكترون: {omega_electron:.3e} rad/s")
        
        best_model = None
        best_ratio = float('inf')
        
        for model_name, theoretical_value in models.items():
            ratio = e_over_m / theoretical_value
            print(f"{model_name}: {theoretical_value:.3e}, النسبة: {ratio:.3e}")
            
            if abs(np.log10(ratio)) < abs(np.log10(best_ratio)):
                best_ratio = ratio
                best_model = model_name
        
        print(f"أفضل نموذج: {best_model} بنسبة {best_ratio:.3e}")
        
        return omega_electron, e_over_m, models, best_model
    
    def dark_matter_energy_connection(self):
        """تحليل العلاقة مع المادة والطاقة المظلمة"""
        
        print("\n=== تحليل المادة والطاقة المظلمة ===")
        
        # الكثافات
        rho_DE = self.Omega_DE * self.rho_critical
        rho_DM = self.Omega_DM * self.rho_critical
        rho_b = self.Omega_b * self.rho_critical
        
        print(f"كثافة الطاقة المظلمة: {rho_DE:.3e} kg/m³")
        print(f"كثافة المادة المظلمة: {rho_DM:.3e} kg/m³")
        print(f"كثافة المادة العادية: {rho_b:.3e} kg/m³")
        
        # نسبة الطاقة المظلمة إلى المادة المظلمة
        DE_DM_ratio = self.Omega_DE / self.Omega_DM
        print(f"نسبة الطاقة المظلمة إلى المادة المظلمة: {DE_DM_ratio:.3f}")
        
        # اقتراح: هل هذه النسبة مرتبطة بثوابت الفتيلة؟
        # نسبة الذهبي، e، π، إلخ
        golden_ratio = (1 + np.sqrt(5)) / 2
        e_number = np.e
        pi_number = np.pi
        
        print(f"مقارنة مع النسبة الذهبية φ = {golden_ratio:.3f}: فرق = {abs(DE_DM_ratio - golden_ratio):.3f}")
        print(f"مقارنة مع e = {e_number:.3f}: فرق = {abs(DE_DM_ratio - e_number):.3f}")
        print(f"مقارنة مع π = {pi_number:.3f}: فرق = {abs(DE_DM_ratio - pi_number):.3f}")
        
        # اقتراح نموذج: انكسار التناظر يؤدي إلى هذه النسب
        # إذا كان التناظر الأولي ينكسر بنسبة ε
        epsilon_suggested = DE_DM_ratio - 1
        print(f"عامل انكسار التناظر المقترح: ε = {epsilon_suggested:.3f}")
        
        return rho_DE, rho_DM, rho_b, DE_DM_ratio, epsilon_suggested
    
    def cosmic_expansion_simulation(self, t_max=14e9*365.25*24*3600):
        """محاكاة التوسع الكوني في إطار نظرية الفتيلة"""
        
        print("\n=== محاكاة التوسع الكوني ===")
        
        # معادلة فريدمان المبسطة
        def friedmann_equation(a, t):
            """معادلة فريدمان: da/dt"""
            H = self.H0 * np.sqrt(
                self.Omega_b / a**3 + 
                self.Omega_DM / a**3 + 
                self.Omega_DE
            )
            return a * H
        
        # حل المعادلة
        t_span = np.linspace(1e6*365.25*24*3600, t_max, 1000)  # من مليون سنة إلى الآن
        a_initial = 1e-3  # عامل المقياس الأولي
        
        a_solution = odeint(friedmann_equation, a_initial, t_span)
        a_values = a_solution.flatten()
        
        # حساب معدل التوسع
        H_values = []
        for i, (a, t) in enumerate(zip(a_values, t_span)):
            H = self.H0 * np.sqrt(
                self.Omega_b / a**3 + 
                self.Omega_DM / a**3 + 
                self.Omega_DE
            )
            H_values.append(H)
        
        H_values = np.array(H_values)
        
        # تحويل الزمن إلى سنوات
        t_years = t_span / (365.25 * 24 * 3600)
        
        print(f"عامل المقياس الحالي: a = {a_values[-1]:.3f}")
        print(f"ثابت هابل الحالي: H = {H_values[-1]*3.086e22/1000:.1f} km/s/Mpc")
        
        # البحث عن علاقة مع التردد الأساسي للفتيلة
        # اقتراح: H ∝ ω₀ × f(cosmic parameters)
        omega_fundamental = 1.0  # rad/s
        
        # نسبة ثابت هابل إلى التردد الأساسي
        H_omega_ratio = self.H0 / omega_fundamental
        print(f"نسبة H₀/ω₀: {H_omega_ratio:.3e}")
        
        # عمر الكون
        age_universe = 1 / self.H0
        print(f"عمر الكون التقريبي: {age_universe/(365.25*24*3600):.2e} سنة")
        
        return t_years, a_values, H_values, H_omega_ratio
    
    def symmetry_breaking_evolution(self, A_values=[0.0, 0.1, 0.5, 1.0]):
        """محاكاة تطور انكسار التناظر"""
        
        print("\n=== محاكاة انكسار التناظر ===")
        
        def epsilon_function(t, A, lambda_param=1.0):
            """دالة انكسار التناظر"""
            return 1 + A * (1 - np.exp(-lambda_param * t))
        
        t_values = np.linspace(0, 10, 1000)
        
        results = {}
        for A in A_values:
            epsilon_values = [epsilon_function(t, A) for t in t_values]
            results[A] = epsilon_values
            
            # حساب الطاقة النهائية
            final_epsilon = 1 + A
            print(f"A = {A}: ε(∞) = {final_epsilon:.3f}")
        
        return t_values, results
    
    def optimize_filament_parameters(self):
        """تحسين معاملات النظرية لتطابق الرصدات"""
        
        print("\n=== تحسين معاملات النظرية ===")
        
        def objective_function(params):
            """دالة الهدف للتحسين"""
            k, A, lambda_param = params
            
            # حساب الانحرافات عن القيم المرصودة
            # 1. L⋅C = 1
            LC_error = abs(k**2 - 1.0)
            
            # 2. نسبة الطاقة المظلمة إلى المادة المظلمة
            epsilon_final = 1 + A
            DE_DM_ratio_theoretical = epsilon_final + 1  # نموذج مبسط
            DE_DM_ratio_observed = self.Omega_DE / self.Omega_DM
            ratio_error = abs(DE_DM_ratio_theoretical - DE_DM_ratio_observed)
            
            # 3. ثابت البنية الدقيقة (محاولة ربطه بالمعاملات)
            alpha_theoretical = A / (2 * np.pi)  # نموذج مقترح
            alpha_observed = 1/137.036
            alpha_error = abs(alpha_theoretical - alpha_observed)
            
            # الخطأ الكلي
            total_error = LC_error + ratio_error + alpha_error * 1000  # وزن أكبر لثابت البنية الدقيقة
            
            return total_error
        
        # التحسين
        initial_guess = [1.0, 0.2, 1.0]  # k, A, lambda
        bounds = [(0.5, 2.0), (0.0, 2.0), (0.1, 10.0)]
        
        result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        optimal_k, optimal_A, optimal_lambda = result.x
        
        print(f"المعاملات المحسنة:")
        print(f"  k = {optimal_k:.6f}")
        print(f"  A = {optimal_A:.6f}")
        print(f"  λ = {optimal_lambda:.6f}")
        print(f"الخطأ الكلي: {result.fun:.6f}")
        
        # التحقق من النتائج
        LC_optimized = optimal_k**2
        epsilon_final_optimized = 1 + optimal_A
        alpha_optimized = optimal_A / (2 * np.pi)
        
        print(f"النتائج المحسنة:")
        print(f"  L⋅C = {LC_optimized:.6f} (الهدف: 1.000)")
        print(f"  ε(∞) = {epsilon_final_optimized:.6f}")
        print(f"  α المقترح = {alpha_optimized:.6f} (الفعلي: {1/137.036:.6f})")
        
        return optimal_k, optimal_A, optimal_lambda, result
    
    def create_comprehensive_plots(self):
        """إنشاء رسوم بيانية شاملة"""
        
        print("\n=== إنشاء الرسوم البيانية ===")
        
        # تشغيل جميع المحاكاات
        r_m, r_s, L, C, LC, omega = self.filament_LC_simulation()
        omega_e, e_over_m, models, best_model = self.electron_charge_mass_correlation()
        rho_DE, rho_DM, rho_b, DE_DM_ratio, epsilon_sugg = self.dark_matter_energy_connection()
        t_years, a_values, H_values, H_omega_ratio = self.cosmic_expansion_simulation()
        t_sym, sym_results = self.symmetry_breaking_evolution()
        opt_k, opt_A, opt_lambda, opt_result = self.optimize_filament_parameters()
        
        # إنشاء الرسوم
        fig = plt.figure(figsize=(20, 16))
        
        # الرسم 1: العلاقة L-C
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(r_m, LC, 'b-', linewidth=2, label='L⋅C')
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='L⋅C = 1')
        ax1.set_xlabel('r_m')
        ax1.set_ylabel('L⋅C')
        ax1.set_title('L-C Relationship')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # الرسم 2: التردد
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(r_m, omega, 'g-', linewidth=2, label='ω = 1/√(LC)')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='ω = 1')
        ax2.set_xlabel('r_m')
        ax2.set_ylabel('ω (rad/s)')
        ax2.set_title('Fundamental Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # الرسم 3: نسب الكون
        ax3 = plt.subplot(3, 3, 3)
        labels = ['Dark Energy', 'Dark Matter', 'Ordinary Matter']
        sizes = [self.Omega_DE, self.Omega_DM, self.Omega_b]
        colors = ['red', 'blue', 'green']
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Universe Composition')
        
        # الرسم 4: التوسع الكوني
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(t_years/1e9, a_values, 'purple', linewidth=2)
        ax4.set_xlabel('Time (Gyr)')
        ax4.set_ylabel('Scale Factor a(t)')
        ax4.set_title('Cosmic Expansion')
        ax4.grid(True, alpha=0.3)
        
        # الرسم 5: ثابت هابل
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t_years/1e9, H_values*3.086e22/1000, 'orange', linewidth=2)
        ax5.set_xlabel('Time (Gyr)')
        ax5.set_ylabel('Hubble Parameter (km/s/Mpc)')
        ax5.set_title('Hubble Parameter Evolution')
        ax5.grid(True, alpha=0.3)
        
        # الرسم 6: انكسار التناظر
        ax6 = plt.subplot(3, 3, 6)
        for A, epsilon_vals in sym_results.items():
            ax6.plot(t_sym, epsilon_vals, linewidth=2, label=f'A = {A}')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('ε(t)')
        ax6.set_title('Symmetry Breaking Evolution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # الرسم 7: مقارنة النماذج للإلكترون
        ax7 = plt.subplot(3, 3, 7)
        model_names = list(models.keys())
        model_values = list(models.values())
        ratios = [e_over_m / val for val in model_values]
        
        bars = ax7.bar(range(len(model_names)), [np.log10(abs(r)) for r in ratios], 
                      color=['red' if name == best_model else 'blue' for name in model_names])
        ax7.set_xlabel('Models')
        ax7.set_ylabel('log₁₀(Ratio)')
        ax7.set_title('Electron e/m Models Comparison')
        ax7.set_xticks(range(len(model_names)))
        ax7.set_xticklabels([f'M{i+1}' for i in range(len(model_names))], rotation=45)
        ax7.grid(True, alpha=0.3)
        
        # الرسم 8: طيف الترددات
        ax8 = plt.subplot(3, 3, 8)
        frequencies = [1.0, omega_e, self.H0, 1/(365.25*24*3600)]  # أساسي، إلكترون، هابل، سنوي
        freq_labels = ['Fundamental', 'Electron', 'Hubble', 'Annual']
        colors_freq = ['blue', 'red', 'green', 'orange']
        
        log_freqs = [np.log10(f) for f in frequencies]
        bars_freq = ax8.bar(range(len(frequencies)), log_freqs, color=colors_freq, alpha=0.7)
        ax8.set_xlabel('Physical Systems')
        ax8.set_ylabel('log₁₀(Frequency) [Hz]')
        ax8.set_title('Frequency Spectrum')
        ax8.set_xticks(range(len(freq_labels)))
        ax8.set_xticklabels(freq_labels, rotation=45)
        ax8.grid(True, alpha=0.3)
        
        # الرسم 9: المعاملات المحسنة
        ax9 = plt.subplot(3, 3, 9)
        param_names = ['k', 'A', 'λ']
        param_values = [opt_k, opt_A, opt_lambda]
        ideal_values = [1.0, 0.2, 1.0]  # قيم مثالية افتراضية
        
        x_pos = np.arange(len(param_names))
        width = 0.35
        
        bars1 = ax9.bar(x_pos - width/2, param_values, width, label='Optimized', alpha=0.7)
        bars2 = ax9.bar(x_pos + width/2, ideal_values, width, label='Initial', alpha=0.7)
        
        ax9.set_xlabel('Parameters')
        ax9.set_ylabel('Values')
        ax9.set_title('Optimized Parameters')
        ax9.set_xticks(x_pos)
        ax9.set_xticklabels(param_names)
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/comprehensive_simulation.png', dpi=300, bbox_inches='tight')
        
        return fig


def main():
    """الدالة الرئيسية"""
    
    print("=" * 80)
    print("محاكاة شاملة لنظرية الفتيلة وعلاقتها بالفيزياء الكونية")
    print("=" * 80)
    
    # إنشاء المحاكي
    simulator = ComprehensiveFilamentSimulation()
    
    # تشغيل جميع المحاكاات
    print("\n1. محاكاة العلاقة L-C...")
    r_m, r_s, L, C, LC, omega = simulator.filament_LC_simulation()
    
    print("\n2. تحليل شحنة الإلكترون...")
    omega_e, e_over_m, models, best_model = simulator.electron_charge_mass_correlation()
    
    print("\n3. تحليل المادة والطاقة المظلمة...")
    rho_DE, rho_DM, rho_b, DE_DM_ratio, epsilon_sugg = simulator.dark_matter_energy_connection()
    
    print("\n4. محاكاة التوسع الكوني...")
    t_years, a_values, H_values, H_omega_ratio = simulator.cosmic_expansion_simulation()
    
    print("\n5. محاكاة انكسار التناظر...")
    t_sym, sym_results = simulator.symmetry_breaking_evolution()
    
    print("\n6. تحسين معاملات النظرية...")
    opt_k, opt_A, opt_lambda, opt_result = simulator.optimize_filament_parameters()
    
    print("\n7. إنشاء الرسوم البيانية...")
    fig = simulator.create_comprehensive_plots()
    print("   - تم حفظ الرسم الشامل في comprehensive_simulation.png")
    
    # ملخص النتائج
    print("\n" + "=" * 80)
    print("ملخص النتائج الرئيسية")
    print("=" * 80)
    
    print(f"1. العلاقة الأساسية L⋅C = {np.mean(LC):.6f} ± {np.std(LC):.2e}")
    print(f"2. التردد الأساسي ω = {np.mean(omega):.6f} ± {np.std(omega):.2e} rad/s")
    print(f"3. تردد الإلكترون: {omega_e:.3e} rad/s")
    print(f"4. أفضل نموذج للإلكترون: {best_model}")
    print(f"5. نسبة الطاقة المظلمة/المادة المظلمة: {DE_DM_ratio:.3f}")
    print(f"6. عامل انكسار التناظر المقترح: ε = {epsilon_sugg:.3f}")
    print(f"7. نسبة ثابت هابل/التردد الأساسي: {H_omega_ratio:.3e}")
    print(f"8. المعاملات المحسنة: k={opt_k:.3f}, A={opt_A:.3f}, λ={opt_lambda:.3f}")
    
    # توصيات للبحث المستقبلي
    print("\n" + "=" * 80)
    print("توصيات للبحث المستقبلي")
    print("=" * 80)
    print("1. تطوير نموذج أكثر دقة لربط انكسار التناظر بالمكونات الكونية")
    print("2. البحث عن تنبؤات قابلة للاختبار من النظرية")
    print("3. دراسة كيفية ظهور القوى الأساسية من الفتيلة الأولية")
    print("4. تطوير نموذج لتشكل البنى الكونية في إطار نظرية الفتيلة")
    print("5. البحث عن علاقات رياضية أعمق بين الثوابت الفيزيائية")
    
    plt.show()
    
    return simulator


if __name__ == "__main__":
    comprehensive_simulator = main()

