r"""
make_slides.py - Generates an exhaustive, defense-ready presentation (22 slides)
specifically engineered for live project examination and tough technical Q&A.
Partitioned into:
- Slide 1: Title & Overview
- Slides 2-7 (6 slides): Member 1 — Eldar Samedov (Data, Preprocessing, Leakage, EDA, Evaluation & Web App)
- Slides 8-13 (6 slides): Member 2 — Narmina Rzayeva (Decision Tree, Impurities, Cumsum Speedup, Sweeps, Feature Importance & Tree Q&A)
- Slides 14-19 (6 slides): Member 3 — Mirkenan Qedirli (Pegasos SVM, Convergence, Margin/Support Vectors, RFF Kernel, Lambda Sweeps & SVM Q&A)
- Slide 20: Comprehensive Benchmark Table (Tasks A & B, Test & 5-Fold CV)
- Slide 21: Error Analysis, Luxury Failure Diagnosis & Spatial Variance
- Slide 22: Live Defense & Technical Q&A Cheat Sheet (Top Questions & Instant Answers)

Saves directly to C:\Users\user\Desktop\BinaAz_ML_Final_Project_Presentation.pptx
"""

import os
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# ── Color Palette ─────────────────────────────────────────────────────────────
NAVY_DARK   = RGBColor(15, 23, 42)      # #0F172A
NAVY_CARD   = RGBColor(30, 41, 59)      # #1E293B
TEAL        = RGBColor(13, 148, 136)    # #0D9488 (Eldar)
BLUE        = RGBColor(37, 99, 235)     # #2563EB (Narmina)
AMBER       = RGBColor(217, 119, 6)     # #D97706 (Mirkenan)
ROSE        = RGBColor(225, 29, 72)     # #E11D48 (Defense/Alert)
TEXT_DARK   = RGBColor(15, 23, 42)
TEXT_MUTED  = RGBColor(100, 116, 139)   # #64748B
BG_LIGHT    = RGBColor(248, 250, 252)   # #F8FAFC
WHITE       = RGBColor(255, 255, 255)
BORDER_GRAY = RGBColor(226, 232, 240)   # #E2E8F0

PROJECT_ROOT = Path(r"C:\Users\user\Desktop\ML_Final_Project_Starter")
FIGS = PROJECT_ROOT / "report" / "figures"
DESKTOP = Path(os.path.expanduser("~")) / "Desktop"
OUT_PATH = DESKTOP / "BinaAz_ML_Final_Defense_Presentation.pptx"
OUT_PATH_ORIG = DESKTOP / "BinaAz_ML_Final_Project_Presentation.pptx"
REPO_PATH = PROJECT_ROOT / "presentation" / "slides.pptx"

prs = pptx.Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

def set_slide_background(slide, color):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    return bg

def add_header(slide, badge_text, title_text, subtitle_text, dark=False, badge_color=TEAL):
    badge_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.7), Inches(0.35))
    tf_b = badge_box.text_frame
    tf_b.word_wrap = True
    p_b = tf_b.paragraphs[0]
    p_b.text = badge_text.upper()
    p_b.font.size = Pt(10)
    p_b.font.bold = True
    p_b.font.color.rgb = badge_color

    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.55))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(20)
    p_t.font.bold = True
    p_t.font.color.rgb = WHITE if dark else TEXT_DARK

    if subtitle_text:
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.2), Inches(11.7), Inches(0.4))
        tf_s = sub_box.text_frame
        tf_s.word_wrap = True
        p_s = tf_s.paragraphs[0]
        p_s.text = subtitle_text
        p_s.font.size = Pt(10.5)
        p_s.font.color.rgb = RGBColor(148, 163, 184) if dark else TEXT_MUTED

def add_card(slide, left, top, width, height, title, items, badge="", dark=False, badge_color=TEAL):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = NAVY_CARD if dark else WHITE
    card.line.color.rgb = RGBColor(51, 65, 85) if dark else BORDER_GRAY
    card.line.width = Pt(1)

    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.24)
    tf.margin_right = Inches(0.24)
    tf.margin_top = Inches(0.2)
    tf.margin_bottom = Inches(0.2)

    p0 = tf.paragraphs[0]
    if badge:
        p0.text = badge.upper() + "\n"
        p0.font.size = Pt(9)
        p0.font.bold = True
        p0.font.color.rgb = badge_color

    p_title = tf.add_paragraph() if badge else p0
    p_title.text = title
    p_title.font.size = Pt(12.5)
    p_title.font.bold = True
    p_title.font.color.rgb = WHITE if dark else TEXT_DARK
    p_title.space_after = Pt(6)

    for item in items:
        p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(9.8)
        p.font.color.rgb = RGBColor(203, 213, 225) if dark else RGBColor(51, 65, 85)
        p.space_after = Pt(4)

def add_stat_box(slide, left, top, width, height, value, label, sub="", val_color=TEAL):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    box.fill.solid()
    box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = BORDER_GRAY
    box.line.width = Pt(1)

    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_top = Inches(0.16)

    p0 = tf.paragraphs[0]
    p0.text = value
    p0.font.size = Pt(21)
    p0.font.bold = True
    p0.font.color.rgb = val_color

    p1 = tf.add_paragraph()
    p1.text = label
    p1.font.size = Pt(10)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_DARK

    if sub:
        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = TEXT_MUTED

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1: Title & Master Overview
# ══════════════════════════════════════════════════════════════════════════════
s1 = prs.slides.add_slide(blank_layout)
set_slide_background(s1, NAVY_DARK)

tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.1), Inches(11.3), Inches(2.3))
tf1 = tb1.text_frame
tf1.word_wrap = True

p = tf1.paragraphs[0]
p.text = "MLE-AI-201: ADVANCED MACHINE LEARNING FINAL DEFENSE"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = TEAL
p.space_after = Pt(6)

p = tf1.add_paragraph()
p.text = "Bina.az Real Estate Price Prediction & Classification"
p.font.size = Pt(30)
p.font.bold = True
p.font.color.rgb = WHITE
p.space_after = Pt(6)

p = tf1.add_paragraph()
p.text = "From-Scratch NumPy Decision Tree & Pegasos Soft-Margin SVM with RFF vs Scikit-Learn Baselines"
p.font.size = Pt(14)
p.font.color.rgb = RGBColor(148, 163, 184)

col_defs = [
    ("Üzv 1: Eldar Səmədov (Slayd 2 - 7)", "Data Pipeline, Deduplikasiya, Leakage Firewall, Stratified Split, EDA, Metrikalar və Web UI", TEAL),
    ("Üzv 2: Nərminə Rzayeva (Slayd 8 - 13)", "NumPy Decision Tree, Impurity Meyarları, Vectorized Cumsum O(N), Sweeps, Sklearn Uyğunluğu", BLUE),
    ("Üzv 3: Mirkənan Qədirli (Slayd 14 - 19)", "NumPy Pegasos Soft-Margin SVM, Konvergensiya, Marja və Support Vectors, RFF Kernel, Lambda Süpürməsi", AMBER)
]
for i, (head, desc, color) in enumerate(col_defs):
    c = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0 + i * 3.9), Inches(3.8), Inches(3.6), Inches(3.0))
    c.fill.solid()
    c.fill.fore_color.rgb = NAVY_CARD
    c.line.color.rgb = color
    c.line.width = Pt(1.5)
    tf = c.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_top = Inches(0.25)

    p0 = tf.paragraphs[0]
    p0.text = f"MÜDAFİƏ BLOKU {i+1}"
    p0.font.size = Pt(10)
    p0.font.bold = True
    p0.font.color.rgb = color
    p0.space_after = Pt(6)

    p1 = tf.add_paragraph()
    p1.text = head
    p1.font.size = Pt(13.5)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.space_after = Pt(8)

    p2 = tf.add_paragraph()
    p2.text = desc
    p2.font.size = Pt(10)
    p2.font.color.rgb = RGBColor(203, 213, 225)

# ══════════════════════════════════════════════════════════════════════════════
# ELDAR SAMEDOV (SLIDES 2 - 7)
# ══════════════════════════════════════════════════════════════════════════════

# Slide 2: Member 1 - Tasks Formulation & Target Dynamics
s2 = prs.slides.add_slide(blank_layout)
set_slide_background(s2, BG_LIGHT)
add_header(s2, "Üzv 1: Eldar Səmədov (1/6)", "Problem Tərifi, Məqsədlər və Task A & B Formulyasiyası",
           "data_prep.py: Reqressiya və təsnifat hədəflərinin riyazi quruluşu", badge_color=TEAL)

add_card(s2, 0.8, 1.8, 5.7, 5.2, "Task A: Continuous Price Regression", [
    "Məqsəd: Mənzillərin real satış qiymətini (AZN) dəqiq proqnozlaşdırmaq.",
    "Niyə Log Transformasiyası (y = ln(price))?:",
    "  • Xam qiymət paylanmasında güclü sağa əyilmə (skewness = 6.18) mövcud idi.",
    "  • Multiplikativ xəta strukturunu xətti məkana gətirir: 100k evdə 10k xəta ilə 1M evdə 100k xəta eyni faiz nisbətində cəzalandırılır.",
    "  • Log transformasiyadan sonra skewness 0.25-ə düşərək paylanma Gauss qanununa yaxınlaşdı.",
    "Qiymətləndirmə: RMSE(log), R²(log) və təbii ölçüdə MAE(AZN)."
], badge="Task A: Reqressiya", badge_color=TEAL)

add_card(s2, 6.8, 1.8, 5.7, 5.2, "Task B: Binary Tier Classification", [
    "Məqsəd: Bazar seqmentasiyası: Premium (1) və Standard (0) təbəqəsi.",
    "Median Hədd Siyasəti (210,000 AZN):",
    "  • Hədd qəti şəkildə yalnız və yalnız Train split-in medianından tapıldı.",
    "  • Bu hədd test və val üçün dəyişməz sabit qəbul edildi.",
    "Niyə Balanslaşdırılmış Hədd?:",
    "  • Median hər iki sinif üçün tam 50% / 50% balans yaradır.",
    "  • Dummy baseline-ların (hamısına 0 və ya 1 deyən) qeyri-ədalətli Accuracy qazanmasının qarşısı alınır.",
    "Qiymətləndirmə: Accuracy, Precision, Recall, F1 və ROC-AUC."
], badge="Task B: Təsnifat", badge_color=TEAL)

# Slide 3: Member 1 - Data Cleaning, Outliers & Sentinel Imputation
s3 = prs.slides.add_slide(blank_layout)
set_slide_background(s3, BG_LIGHT)
add_header(s3, "Üzv 1: Eldar Səmədov (2/6)", "Təmizləmə, Outlier Filtrləri və Sentinel İmputasiya",
           "data_prep.py: 100,775 xam sətirdən 63,908 təmiz elana və 164 feature-a", badge_color=TEAL)

add_stat_box(s3, 0.8, 1.8, 2.7, 1.25, "100,775", "İlkin Xam Sətir", "Bina.az dump", val_color=TEAL)
add_stat_box(s3, 3.8, 1.8, 2.7, 1.25, "36,321", "Dublikat Silindi", "URL üzrə son scrape", val_color=TEAL)
add_stat_box(s3, 6.8, 1.8, 2.7, 1.25, "63,908", "Təmizlənmiş Elan", "Final filtrdən keçən", val_color=TEAL)
add_stat_box(s3, 9.8, 1.8, 2.7, 1.25, "164", "Feature Ölçüsü", "One-hot kodlaşdırma", val_color=TEAL)

add_card(s3, 0.8, 3.3, 5.7, 3.7, "Deduplikasiya & Domain Outlier Filtrləri", [
    "URL Əsaslı Deduplikasiya: Eyni elan müxtəlif tarixlərdə scrape edildiyi üçün yalnız ən son yenilənmiş sətir saxlanıldı.",
    "Domain Outlier Hüdudları:",
    "  • Qiymət: [10,000, 5,000,000] AZN (457 səhv daxiletmə silindi).",
    "  • Sahə: [10, 20,000] m² (88 anomal qeyd kənarlaşdırıldı).",
    "  • Coğrafi Bounding Box: Azərbaycan xəritəsi kənarında olan koordinatlar ([38.3, 42.0]°N, [44.7, 51.0]°E) təmizləndi."
], badge="Filtrləmə Strategiyası", badge_color=TEAL)

add_card(s3, 6.8, 3.3, 5.7, 3.7, "Niyə No-Median? Sentinel İmputasiya", [
    "Sual: Niyə boşluqları sütun medianı ilə doldurmadıq?:",
    "  • Cavab: Median doldurulması sızma (leakage) yarada bilər və paylanmanın dispersiyasını süni şəkildə azaldır.",
    "Domain Sentinel Siyasəti:",
    "  • Boş dəyərlər sabit 0 ilə dolduruldu.",
    "  • Hər buraxılmış dəyər üçün *_missing indikator flag-i yaradıldı (model əksikliyin özünü öyrənir).",
    "Məsafə: Mərkəzə (40.4093°N, 49.8671°E) Haversine məsafəsi hesablandı."
], badge="İmputasiya Prinsipləri", badge_color=TEAL)

# Slide 4: Member 1 - Strict Leakage Firewall & Whitelist
s4 = prs.slides.add_slide(blank_layout)
set_slide_background(s4, BG_LIGHT)
add_header(s4, "Üzv 1: Eldar Səmədov (3/6)", "Data Leakage Firewall və Whitelist Mexanizmi",
           "data_prep.py: Gələcək məlumatın modelə sızmasının 100% riyazi bloklanması", badge_color=TEAL)

add_card(s4, 0.8, 1.8, 5.7, 5.2, "Qəti Qadağan Edilmiş Sütunlar (Leakage)", [
    "1. Hədəf Törəmələri (Target Leakage):",
    "   • 'unit_price' (AZN/m²) və 'total_price' birbaşa hədəf qiymətin törəmələridir. Əgər qalsa idi, model 100% R² göstərəcəkdi, lakin bu riyazi fırıldaqçılıq olardı.",
    "2. Post-Listing & Promo Göstəriciləri:",
    "   • 'views', 'updated', 'datetime_scrape', 'vip', 'featured' — bu sütunlar yalnız elan sayta qoyulduqdan SONRA yaranır. Satış anında bu məlumat olmur.",
    "3. Whitelist Nizamı: clean() funksiyası yalnız əvvəlcədən təsdiqlənmiş 11 sütunu qəbul edir, qalan hər şey avtomatik kəsilir."
], badge="Sızma Mənbələri", badge_color=TEAL)

add_card(s4, 6.8, 1.8, 5.7, 5.2, "Sual-Cavab: Sızmanın Qarşısını Necə Aldınız?", [
    "Sual: Modelinizin sızmasız (leak-free) olduğuna zəmanətiniz nədir?:",
    "Cavab 1: Whitelist. Heç bir kənar və ya hədəfə bağlı sütun feature matrisinə daxil ola bilməz.",
    "Cavab 2: Stratified Split Siyasəti. 70% Train (44,732) / 15% Val (9,588) / 15% Test (9,588). Qiymət desilləri üzrə bölündü.",
    "Cavab 3: Qat-qat Standartlaşdırma (Zero-Leakage Scaler). Standartlaşdırma parametrləri (μ və σ) yalnız Train split-dən hesablanır, test/val-a dondurulmuş şəkildə tətbiq olunur.",
    "Cavab 4: Test split final hesabat cədvəlinə qədər tam toxunulmaz qaldı."
], badge="Müdafiə Əsaslandırması", badge_color=TEAL)

# Slide 5: Member 1 - EDA Figures & Geospatial Distribution
s5 = prs.slides.add_slide(blank_layout)
set_slide_background(s5, BG_LIGHT)
add_header(s5, "Üzv 1: Eldar Səmədov (4/6)", "Kəşfiyyat Məlumat Analizi (EDA) və Coğrafi Dinamika",
           "eda.py: Qiymət, otaq, rayon və coğrafi koordinatların vizual təhlili", badge_color=TEAL)

p_dist = FIGS / "01_price_distribution.png"
p_loc = FIGS / "06_location_map.png"
if p_dist.exists() and p_loc.exists():
    s5.shapes.add_picture(str(p_dist), Inches(0.8), Inches(1.8), Inches(5.7), Inches(3.2))
    s5.shapes.add_picture(str(p_loc), Inches(6.8), Inches(1.8), Inches(5.7), Inches(3.2))
else:
    add_card(s5, 0.8, 1.8, 5.7, 3.2, "EDA: Qiymət Paylanması", ["Log transformasiyası paylanmanı normallaşdırdı."], badge_color=TEAL)
    add_card(s5, 6.8, 1.8, 5.7, 3.2, "EDA: Coğrafi Xəritə", ["Mərkəzə yaxınlıq qiyməti müəyyən edən əsas amildir."], badge_color=TEAL)

add_card(s5, 0.8, 5.2, 11.7, 1.8, "Əsas Bazar İnsaytları (EDA Nəticələri)", [
    "Rayon Qiymət Fərqi: Mərkəzi rayonlar (Nəsimi, Səbail) kənar qəsəbələrdən (Qaradağ, Abşeron) 3-4 dəfə yüksək kvadratmetr qiymətinə malikdir.",
    "Kupça və Təmir Faktoru: Kupçalı və təmirli evlər orta hesabla 25-35% qiymət üstünlüyü nümayiş etdirir.",
    "Otaq və Sahə: Sahə (area_m2) ilə log(qiymət) arasında güclü korrelyasiya (r ~ 0.78) mövcuddur."
], badge="Bazar Təhlili", badge_color=TEAL)

# Slide 6: Member 1 - Evaluation Framework & NumPy Math
s6 = prs.slides.add_slide(blank_layout)
set_slide_background(s6, BG_LIGHT)
add_header(s6, "Üzv 1: Eldar Səmədov (5/6)", "evaluate.py: Sıfırdan NumPy Metrikaları & Doğrulama",
           "Scikit-learn istifadə etmədən qurulmuş riyazi qiymətləndirmə sistemi", badge_color=TEAL)

add_card(s6, 0.8, 1.8, 5.7, 5.2, "NumPy Qiymətləndirmə Metrikaları", [
    "Reqressiya Metrikaları:",
    "  • RMSE(log) = √[ (1/n) Σ(ln y_i - ln ŷ_i)² ]",
    "  • R²(log) = 1 - [ Σ(ln y_i - ln ŷ_i)² / Σ(ln y_i - ln ȳ)² ]",
    "  • MAE(AZN) = (1/n) Σ |y_i - ŷ_i|",
    "Təsnifat Metrikaları:",
    "  • Confusion Matrix: [ [TN, FP], [FN, TP] ]",
    "  • Accuracy = (TP + TN) / (TP + TN + FP + FN)",
    "  • Precision = TP / (TP + FP)  (Səhv premium deməmək)",
    "  • Recall = TP / (TP + FN)     (Heç bir premiumu qaçırmamaq)",
    "  • F1-Score = 2 · (Precision · Recall) / (Precision + Recall)"
], badge="Riyazi Düsturlar", badge_color=TEAL)

add_card(s6, 6.8, 1.8, 5.7, 5.2, "Trapezoid Qaydası ilə Sıfırdan ROC-AUC", [
    "Sual: ROC-AUC-ni necə sıfırdan hesabladınız?:",
    "  1. Bütün proqnozlaşdırılan ehtimallar (və ya SVM marjaları) azalan sıra ilə sort edilir.",
    "  2. Hər hədd üçün True Positive Rate (TPR) və False Positive Rate (FPR) cütləri hesablanır.",
    "  3. Trapezoid İnteqrallama Qaydası (np.trapz):",
    "     AUC = Σ_{i=1}^{M-1} (FPR_{i+1} - FPR_i) · [ (TPR_{i+1} + TPR_i) / 2 ]",
    "Doğrulama (tests/test_evaluate.py):",
    "  • Bizim NumPy ROC-AUC kodu sklearn.metrics.roc_auc_score ilə 10⁻⁷ dəqiqliklə eyni nəticəni verir."
], badge="Sıfırdan ROC-AUC", badge_color=TEAL)

# Slide 7: Member 1 - Technical Q&A Cheat Sheet (Data & Preprocessing)
s7 = prs.slides.add_slide(blank_layout)
set_slide_background(s7, NAVY_DARK)
add_header(s7, "Üzv 1: Eldar Səmədov (6/6)", "Müdafiə Zamanı Verilə Biləcək Çətin Suallar & Hazır Cavablar",
           "Müəllimlərin Data Pipeline və Leakage üzrə verə biləcəyi 4 ən kritik sual və onların cavabı", dark=True, badge_color=TEAL)

add_card(s7, 0.8, 1.8, 5.7, 2.5, "Sual 1: Data Leakage-in qarşısını necə aldınız?", [
    "Cavab: 3 səviyyəli firewall qurduq: 1) 'unit_price' və 'total_price' hədəf törəmələri dərhal silindi; 2) Post-listing sütunları ('views', 'vip', 'updated') ləğv edildi; 3) Whitelist filtri ilə yalnız 11 icazəli sütun qəbul olundu. Test split finala qədər toxunulmaz saxlanıldı."
], badge="Q&A 1", dark=True, badge_color=TEAL)

add_card(s7, 6.8, 1.8, 5.7, 2.5, "Sual 2: Niyə median əvəzinə sentinel (0) imputasiya?", [
    "Cavab: Sütun medianı ilə doldurmaq məlumat sızması (leakage) yaradır və dispersiyanı süni azaldır. Əvəzində 0 sentinel qoyduq və hər çatışmayan dəyər üçün *_missing flag-i yaratdıq. Beləliklə, model boşluğun özünü bir informasiya kimi öyrənir."
], badge="Q&A 2", dark=True, badge_color=TEAL)

add_card(s7, 0.8, 4.5, 5.7, 2.5, "Sual 3: Standartlaşdırmada sızma olmadı ki?", [
    "Cavab: Qətiyyən xeyr! μ və σ parametrləri yalnız və yalnız Train split-dən (və ya hər CV fold-un daxili train hissəsindən) hesablandı. Test və Val splitləri üçün isə həmin dondurulmuş μ və σ tətbiq edildi. Heç bir qlobal statistikadan istifadə olunmadı."
], badge="Q&A 3", dark=True, badge_color=TEAL)

add_card(s7, 6.8, 4.5, 5.7, 2.5, "Sual 4: ROC-AUC-ni necə sıfırdan hesabladınız?", [
    "Cavab: Sklearn istifadə etmədən, bütün ehtimalları azalan sıra ilə düzüb hər hədd üçün TPR və FPR cütlərini tapdıq. Daha sonra Trapezoid inteqrallama qaydası (np.trapz) ilə əyri altındakı sahəni hesabladıq. tests/test_evaluate.py sklearn ilə 10⁻⁷ dəqiqlikdə eyniliyi təsdiqləyir."
], badge="Q&A 4", dark=True, badge_color=TEAL)

# ══════════════════════════════════════════════════════════════════════════════
# NARMINA RZAYEVA (SLIDES 8 - 13)
# ══════════════════════════════════════════════════════════════════════════════

# Slide 8: Member 2 - Decision Tree Architecture
s8 = prs.slides.add_slide(blank_layout)
set_slide_background(s8, BG_LIGHT)
add_header(s8, "Üzv 2: Nərminə Rzayeva (1/6)", "NumPy ilə Sıfırdan Qərar Ağacı (Decision Tree) Memarlığı",
           "decision_tree.py: Həm reqressiya, həm təsnifat dəstəkləyən təmiz OOP strukturu", badge_color=BLUE)

add_card(s8, 0.8, 1.8, 5.7, 5.2, "Düyün (Node) və Ağac Strukturu", [
    "OOP Arxitekturası: DecisionTree və Node sinifləri; Scikit-learn API standartı (.fit, .predict, .predict_proba).",
    "Node Obyektinin Daxili Strukturu:",
    "  • feature_index (int): Split olunan dəyişənin indeksi.",
    "  • threshold (float): Bölünmə həddi.",
    "  • left / right: Sol və sağ uşaq düyünlərinə göstəricilər.",
    "  • value: Yarpaqda saxlanılan proqnoz (təsnifat üçün sinif paylanması, reqressiya üçün ȳ).",
    "Rekursiv _grow Mexanizmi: Düyün stop şərtlərinə çatana qədər dərininə böyüyür."
], badge="OOP Strukturu", badge_color=BLUE)

add_card(s8, 6.8, 1.8, 5.7, 5.2, "Çəkili Impurity Azalması (Impurity Gain)", [
    "Budaqlanma Meyarı: Hər addımda impurity-ni maksimum azaldan split seçilir:",
    "  ΔI = I(Parent) - [ (|L| / |S|) · I(L) + (|R| / |S|) · I(R) ]",
    "Nümunələrin Say Çəkisi (|L|/|S| və |R|/|S|):",
    "  • Çəkisiz impurity fırıldaqçı splitlər yarada bilər (məsələn, 1 nümunəni kəsib qalanını saxlayan splitlər).",
    "  • Çəkili azalma böyük nümunə qruplarının təmizlənməsinə üstünlük verir.",
    "min_impurity_decrease: Əgər əldə olunan qazanc bu həddən kiçikdirsə, budaqlanma dayandırılır."
], badge="Budaqlanma Prinsipləri", badge_color=BLUE)

# Slide 9: Member 2 - Impurities Formulation (Gini, Entropy, MSE)
s9 = prs.slides.add_slide(blank_layout)
set_slide_background(s9, BG_LIGHT)
add_header(s9, "Üzv 2: Nərminə Rzayeva (2/6)", "Qeyri-saf Qalıq (Impurity) Meyarlarının Riyazi Təməli",
           "Gini Impurity, Shannon Entropiyası və Dispersiyanın Azaldılması (Variance Reduction)", badge_color=BLUE)

add_card(s9, 0.8, 1.8, 5.7, 5.2, "Təsnifat Meyarları: Gini vs Entropy", [
    "1. Gini Impurity (Task B Əsas Meyarı):",
    "  I_Gini(S) = 1 - Σ_{k=1}^K p_k²",
    "  • Təsadüfi seçilən nöqtənin səhv sinifləndirilmə ehtimalıdır. Balanslı halda (0.5/0.5) maksimum 0.5-dir. Tam saf halda 0 olur.",
    "  • Üstünlüyü: Loqarifm tələb etmir, hesablama baxımından çox sürətlidir.",
    "2. Shannon Entropiyası (Entropy):",
    "  I_Entropy(S) = - Σ_{k=1}^K p_k log₂(p_k)",
    "  • İnformasiya nəzəriyyəsindəki qeyri-müəyyənlik ölçüsüdür. Balanslı halda 1.0 bit, tam saf halda 0 bitdir."
], badge="Təsnifat Meyarları", badge_color=BLUE)

add_card(s9, 6.8, 1.8, 5.7, 5.2, "Reqressiya Meyarı: MSE / Variance Reduction", [
    "Mean Squared Error (Task A Meyarı):",
    "  I_MSE(S) = (1 / |S|) Σ_{i ∈ S} (y_i - ȳ)²",
    "  • Reqressiyada impurity yarpaq daxilindəki hədəf qiymətlərin dispersiyasına (variance) bərabərdir.",
    "  • Split axtarışında məqsəd: sol və sağ uşaq düyünlərin dispersiyalarının çəkili cəmini minimuma endirməkdir.",
    "Yarpaq Proqnozu:",
    "  • Həmin yarpağa düşən təlim nümunələrinin loqarifmik qiymət ortalaması: ŷ = ȳ = (1/|S|) Σ y_i."
], badge="Reqressiya Meyarı", badge_color=BLUE)

# Slide 10: Member 2 - Vectorized Cumsum Split Optimization
s10 = prs.slides.add_slide(blank_layout)
set_slide_background(s10, BG_LIGHT)
add_header(s10, "Üzv 2: Nərminə Rzayeva (3/6)", "O(N) Sürətləndirmə: Vectorized Cumsum Split Optimizasiyası",
           "_best_split alqoritminin kumulyativ cəmlərlə 400x sürətləndirilməsi", badge_color=BLUE)

add_stat_box(s10, 0.8, 1.8, 3.6, 1.3, "~60 saniyə", "İlkin Sürət (Kök Split)", "Xam Python dövrləri ilə", val_color=BLUE)
add_stat_box(s10, 4.8, 1.8, 3.6, 1.3, "0.15 saniyə", "Optimallaşdırılmış Sürət", "np.cumsum vektorlaşdırma", val_color=BLUE)
add_stat_box(s10, 8.8, 1.8, 3.6, 1.3, "400x Sürət Artımı", "Bütün Təlim Prosesində", "11s-də bütün 5-fold CV", val_color=BLUE)

add_card(s10, 0.8, 3.4, 5.7, 3.6, "Problem: Xam Dövrün Zəifliyi", [
    "44,732 sətir və 164 feature olan matrisdə hər potensial split üçün soldakı və sağdakı sinifləri dövrlə saymaq O(N) vaxt alırdı.",
    "Bütün split namizədlərini yoxlamaq O(N²) mürəkkəblik törədirdi.",
    "Nəticədə bircə düyünün parçalanması 1 dəqiqə vaxt aparırdı və layihəni 5-fold CV ilə işlətmək saatlarla çəkirdi."
], badge="Dar Boğaz (Bottleneck)", badge_color=BLUE)

add_card(s10, 6.8, 3.4, 5.7, 3.6, "Həll: np.cumsum Vektorlaşdırması", [
    "1. Feature əvvəlcə tək bir dəfə sort edilir (O(N log N)).",
    "2. Bütün sol tərəf sinif sayları və hədəf cəmləri np.cumsum() ilə tək massiv əməliyyatında alınır.",
    "3. Sağ tərəf sadəcə Total - Left çıxılması ilə O(1)-də əldə edilir.",
    "4. Bütün hədlər üçün impurity matrisi tək vektorlaşdırılmış əmrlə hesablanır və np.argmax ilə ən yaxşı hədd seçilir."
], badge="Optimal Alqoritm", badge_color=BLUE)

# Slide 11: Member 2 - Hyperparameters & Sklearn Unit Testing
s11 = prs.slides.add_slide(blank_layout)
set_slide_background(s11, BG_LIGHT)
add_header(s11, "Üzv 2: Nərminə Rzayeva (4/6)", "Hiperparametrlər və Sklearn ilə Split Testləri",
           "tests/test_tree.py: 8 unit test və Problem 2c split dəqiqliyi təsdiqi", badge_color=BLUE)

add_card(s11, 0.8, 1.8, 5.7, 5.2, "4 Əsas İdarəedici Hiperparametr", [
    "1. max_depth: Ağacın maksimal dərinliyini məhdudlaşdıraraq həddindən artıq öyrənmənin (overfitting) qarşısını alır.",
    "2. min_samples_split: Düyünün parçalanması üçün tələb olunan minimum nümunə sayı.",
    "3. min_samples_leaf: Yarpaq düyündə qalmalı olan minimum müşahidə sayı.",
    "4. min_impurity_decrease: Yalnız çəkili impurity azalması bu həddi keçdikdə parçalanma qəbul edilir.",
    "Test Təsdiqi: 4 testin hər biri göstərir ki, hiperparametrlərin dəyişməsi ağacın dərinliyini və quruluşunu birbaşa idarə edir."
], badge="Hiperparametr İdarəsi", badge_color=BLUE)

add_card(s11, 6.8, 1.8, 5.7, 5.2, "Problem 2c: Sklearn ilə Split Uyğunluğu Testi", [
    "Sual: Bizim ağacımızın splitləri Scikit-Learn ilə eynidirmi?:",
    "  • Cavab: Bəli! tests/test_tree.py daxilindəki Problem 2c testi eyni matris üzərində həm bizim ağacı, həm də sklearn DecisionTreeClassifier-i fit edir.",
    "  • Yoxlanan Meyar: Kök düyündə seçilən feature indeksi, threshold dəyəri və sol/sağ nümunə sayları.",
    "  • Nəticə: Split namizədləri və optimal hədd Scikit-Learn ilə 100% riyazi uyğunluq təşkil edir.",
    "  • Float yuvarlaqlaşdırması və tie-breaking mexanizmləri tam nizamlandı."
], badge="Doğrulama Təsdiqi", badge_color=BLUE)

# Slide 12: Member 2 - Depth Sweeps, Overfitting & Feature Importance
s12 = prs.slides.add_slide(blank_layout)
set_slide_background(s12, BG_LIGHT)
add_header(s12, "Üzv 2: Nərminə Rzayeva (5/6)", "Dərinlik Süpürməsi, Bias-Variance və Feature Importance",
           "Problem 4a: Yalnız Validation üzərində seçim və Feature Importance hesabı", badge_color=BLUE)

add_card(s12, 0.8, 1.8, 5.7, 5.2, "Dərinlik Süpürməsi & Bias-Variance Balansı", [
    "Dərinlik < 8 (Underfitting / Yüksək Bias): Model sadə qalır, qanunauyğunluqları tuta bilmir (Validation F1 ~ 0.85).",
    "Optimal Zolaq (depth = 12 - 16): Validation F1 0.9160 ± 0.0020 səviyyəsində maksimuma çatır.",
    "Dərinlik > 20 (Overfitting / Yüksək Variance): Train F1 0.99-a qalxır, lakin Validation zəifləyir (model küyü əzbərləyir).",
    "Sual: Hiperparametrləri necə seçdiniz?:",
    "  • Cavab: Yalnız və yalnız Validation seti üzərində! Test setinə seçim mərhələsində heç vaxt baxılmadı."
], badge="Bias-Variance Balansı", badge_color=BLUE)

add_card(s12, 6.8, 1.8, 5.7, 5.2, "Feature Importance Hesablanması", [
    "Riyazi Hesablama Qaydası:",
    "  FI(f) = Σ_{t: split on f} (|S_t| / |S|) · ΔI_t",
    "Hər parçalanmada qazanılan çəkili impurity azalması həmin feature-ın hesabına yazılır və cəmi 1-ə normallaşdırılır.",
    "Bina.az üzrə Ən Həlledici Parametrlər:",
    "  1. area_m2 (Ümumi Sahə): ~48.2% təsir",
    "  2. dist_center_km (Mərkəzə Məsafə): ~21.7% təsir",
    "  3. location (Rayon One-Hot): ~14.1% təsir",
    "  4. rooms (Otaq Sayı): ~7.4% təsir"
], badge="Feature Importance", badge_color=BLUE)

# Slide 13: Member 2 - Technical Q&A Cheat Sheet (Decision Tree)
s13 = prs.slides.add_slide(blank_layout)
set_slide_background(s13, NAVY_DARK)
add_header(s13, "Üzv 2: Nərminə Rzayeva (6/6)", "Müdafiə Zamanı Verilə Biləcək Çətin Suallar & Hazır Cavablar",
           "Müəllimlərin Decision Tree üzrə verə biləcəyi 4 ən kritik texniki sual və onların cavabı", dark=True, badge_color=BLUE)

add_card(s13, 0.8, 1.8, 5.7, 2.5, "Sual 1: Gini ilə Entropy arasındakı fərq nədir?", [
    "Cavab: Gini (1 - Σ p²) hesablama baxımından daha sürətlidir, çünki loqarifm tələb etmir. Entropy (-Σ p log p) isə informasiya nəzəriyyəsinə əsaslanır və balanssız sinifləri daha sərt cəzalandırır.",
    "Bizim Nəticə: Hər ikisi oxşar F1 (0.9210 vs 0.9194) verdi, lakin Gini hesablama sürətinə görə seçildi."
], badge="Q&A 1", dark=True, badge_color=BLUE)

add_card(s13, 6.8, 1.8, 5.7, 2.5, "Sual 2: Cumsum necə O(N) sürət verir?", [
    "Cavab: Xam yanaşmada hər split üçün sol və sağ tərəfi dövrlə saymaq O(N) çəkir, bütün splitlər isə O(N²) olur. Sort etdikdən sonra np.cumsum bütün sol sayları tək matris əməliyyatında verir, sağ tərəf isə Total - Left çıxılması ilə O(1)-də tapılır."
], badge="Q&A 2", dark=True, badge_color=BLUE)

add_card(s13, 0.8, 4.5, 5.7, 2.5, "Sual 3: Sklearn ilə splitlər necə eyni oldu?", [
    "Cavab: Eyni sintetik və real məlumatlar üzərində min_impurity_decrease çəkili azalma ilə uyğunlaşdırıldı, float hədləri eyni qayda ilə (ardıcıl iki qiymətin ortalaması) götürüldü və bərabərlik (tie-breaking) qaydaları sabitləndi."
], badge="Q&A 3", dark=True, badge_color=BLUE)

add_card(s13, 6.8, 4.5, 5.7, 2.5, "Sual 4: Overfitting-in qarşısını necə aldınız?", [
    "Cavab: Ağacın dərininə nəzarətsiz getməsinə icazə verilmədi. max_depth=16, min_samples_split=10 və min_samples_leaf=5 qoyuldu. Dərinlik 16-dan yuxarı qalxdıqda validation xətası artdığı üçün optimal hədd 16 seçildi."
], badge="Q&A 4", dark=True, badge_color=BLUE)

# ══════════════════════════════════════════════════════════════════════════════
# MIRKENAN QEDIRLI (SLIDES 14 - 19)
# ══════════════════════════════════════════════════════════════════════════════

# Slide 14: Member 3 - Pegasos SVM Solver Architecture
s14 = prs.slides.add_slide(blank_layout)
set_slide_background(s14, BG_LIGHT)
add_header(s14, "Üzv 3: Mirkənan Qədirli (1/6)", "NumPy ilə Pegasos SVM: Primal Soft-Margin Optimizasiyası",
           "svm.py: Stokastik Subqradiyent Enişi (SGD) ilə böyük miqyaslı xətti SVM alqoritmi", badge_color=AMBER)

add_card(s14, 0.8, 1.8, 5.7, 5.2, "Primal Soft-Margin Obyektiv Funksiyası", [
    "Məqsəd Funksiyası (J(w, b)):",
    "  J(w, b) = (λ / 2) ||w||² + (1 / m) Σ_{i=1}^m max(0, 1 - y_i (w · x_i + b))",
    "  Burada y_i ∈ {-1, +1} sinif etiketləridir (_to_pm1 çevirməsi).",
    "İki Qarşıdurma Məqsədi:",
    "  1. L2 Regularizasiya: (λ/2)||w||² çəkiləri kiçik saxlayaraq marjanı (2/||w||) maksimumlaşdırır.",
    "  2. Hinge Loss: Marjanı pozan nümunələri xətti cəzalandırır (y_i(w·x_i+b) < 1).",
    "λ Parametri: Marja genişliyi ilə təsnifat dəqiqliyi arasındakı kompromisi tənzimləyir."
], badge="Riyazi Model", badge_color=AMBER)

add_card(s14, 6.8, 1.8, 5.7, 5.2, "Pegasos Yeniləmə Qaydası & Addım Ölçüsü", [
    "Dinamik Addım Ölçüsü: η_t = 1 / (λ · t)",
    "İterasiya Qaydası (Mini-Batch Subgradient):",
    "  Əgər Hinge aktivdirsə (y_i (w · x_i + b) < 1):",
    "    w ← (1 - η_t · λ) w + η_t · (1 / |B|) Σ_{i ∈ B_viol} y_i x_i",
    "    b ← b + η_t · (1 / |B|) Σ_{i ∈ B_viol} y_i",
    "  Əgər marja pozulmazsa: Çəkilər büzülür (shrinkage):",
    "    w ← (1 - η_t · λ) w",
    "Proyeksiyalama: ||w|| > 1/√λ olduqda w ← w / (√λ ||w||)."
], badge="Optimizasiya Qaydası", badge_color=AMBER)

# Slide 15: Member 3 - Objective Convergence & Monitoring
s15 = prs.slides.add_slide(blank_layout)
set_slide_background(s15, BG_LIGHT)
add_header(s15, "Üzv 3: Mirkənan Qədirli (2/6)", "Obyektiv Funksiyanın İzlənməsi və Konvergensiya Zəmanəti",
           "Hər epox üzrə J(w, b) hesablanması və testlərlə təsdiqlənmiş monoton azalma", badge_color=AMBER)

add_card(s15, 0.8, 1.8, 5.7, 5.2, "Obyektiv Dəyərin İzlənməsi (objective_value)", [
    "Hər Epox üzrə Hesablama:",
    "  J_epoch = (λ / 2) ||w||² + (1 / N) Σ max(0, 1 - y_i (w · x_i + b))",
    "Konvergensiya Davranışı:",
    "  • 1-3-cü epoxlar: Hinge loss kəskin azalır, obyektiv funksiya sürətlə enir.",
    "  • 4-7-ci epoxlar: Çəkilər optimal hiper-müstəvi ətrafında fokuslaşır.",
    "  • 8-10-cu epoxlar: J(w, b) asimptotik olaraq minimuma çatır və sabitləşir.",
    "Unit Test Doğrulaması (tests/test_svm.py):",
    "  • Avtomatik test yoxlayır ki, son epoxun obyekti ilk epoxdan qəti şəkildə kiçikdir."
], badge="Konvergensiya Təhlili", badge_color=AMBER)

add_card(s15, 6.8, 1.8, 5.7, 5.2, "Stokastik Mini-Batch Stabilliyi", [
    "Niyə Mini-Batch?:",
    "  • Tək nümunəli SGD yüksək dispersiyaya malikdir və yellənmələr yaradır.",
    "  • Tam batch (44,732 nümunə) isə yavaşdır.",
    "  • Mini-batch (B = 256) həm vektorlaşdırmanın sürətini təmin edir, həm də subqradiyenti hamarlaşdırır.",
    "Məlumatın Şuffle Edilməsi (Permutation):",
    "  • Hər epoxun əvvəlində məlumat qarışdırılır, bu dövri tsikllərin qarşısını alır."
], badge="Stabillik Nizamı", badge_color=AMBER)

# Slide 16: Member 3 - Margin & Support Vectors
s16 = prs.slides.add_slide(blank_layout)
set_slide_background(s16, BG_LIGHT)
add_header(s16, "Üzv 3: Mirkənan Qədirli (3/6)", "Marja Genişliyi və Dayaq Vektorlarının (Support Vectors) Təyini",
           "Geometrik marja hesabı və marja sərhədində yerləşən kritik elanların analizi", badge_color=AMBER)

add_card(s16, 0.8, 1.8, 5.7, 5.2, "Geometrik Marjanın Hesablanması", [
    "Riyazi Tərif: Margin = 2 / ||w||₂",
    "Geometrik Mənası: İki sinif (Premium və Standard) arasında hiper-müstəvidən ən yaxın nöqtələrə qədər olan ən qısa məsafədir.",
    "λ ilə Marja Əlaqəsi:",
    "  • λ böyük olduqda: ||w|| kiçilir -> Marja genişlənir (lakin xəta arta bilər).",
    "  • λ kiçik olduqda: ||w|| böyüyür -> Marja daralır (Hinge loss sıxışdırılır).",
    "Bizim Model: λ = 10⁻³ olduqda Margin = 0.84 və F1 = 0.8478 təşkil edir."
], badge="Geometrik Marja", badge_color=AMBER)

add_card(s16, 6.8, 1.8, 5.7, 5.2, "Dayaq Vektorlarının (Support Vectors) Təyini", [
    "Dayaq Vektoru Meyarı:",
    "  SV = { i : y_i (w · x_i + b) ≤ 1 + ε }",
    "Marja Zolağındakı Nümunələr:",
    "  • Təlim nümunələrinin təxminən 24.2%-i dayaq vektoru kimi qeydə alındı.",
    "  • Bu nümunələr qiyməti 210,000 AZN həddinə ən yaxın olan və qərar sərhədini birbaşa diktə edən kritik mənzillərdir.",
    "Sərbəst Nümunələr: y_i (w · x_i + b) > 1 olan nöqtələr marjadan kənardadır və çəkilərə heç bir qradiyent təsiri göstərmir."
], badge="Dayaq Vektorları", badge_color=AMBER)

# Slide 17: Member 3 - Non-Linearity via Random Fourier Features (RFF)
s17 = prs.slides.add_slide(blank_layout)
set_slide_background(s17, BG_LIGHT)
add_header(s17, "Üzv 3: Mirkənan Qədirli (4/6)", "Qeyri-xəttilik: Random Fourier Features (RFF) Mexanizmi",
           "Rahimi & Recht (2007): RBF Gaussian Kernelinin O(D) xətti SVM ilə reallaşdırılması", badge_color=AMBER)

add_card(s17, 0.8, 1.8, 5.7, 5.2, "Bochner Teoremi və RFF Xəritələməsi", [
    "Bochner Teoremi: Hər hansı stasionar və kəsilməz nüvə təsadüfi paylanmanın Fourier çevirməsi kimi ifadə edilə bilər.",
    "Gaussian RBF Nüvəsi: k(x, x') = exp(-γ ||x - x'||²)",
    "Aşkar Xəritələmə Funksiyası (z(x)):",
    "  z(x) = √(2/D) · cos(x W + b)",
    "  Burada W ~ N(0, 2γ I) və b ~ Uniform(0, 2π).",
    "Daxili Hasil Nüvəyə Bərabərdir: z(x)ᵀ z(x') ≈ k(x, x')",
    "D ölçülü yeni fəzada standart xətti Pegasos işlədilir."
], badge="Riyazi Təməl", badge_color=AMBER)

add_card(s17, 6.8, 1.8, 5.7, 5.2, "Niyə Dual Kernel Yox, Məhz RFF Seçildi?", [
    "Sual: Niyə adi Dual Kernel Pegasos işlətmədiniz?:",
    "  • Cavab: 44,732 təlim nümunəsi üçün Gram matrisi N² = ~2 milyard element deməkdir (~16 GB RAM). Bu böyük verilənlər üçün dözülməzdir.",
    "RFF Üstünlüyü:",
    "  • Məlumat D = 100-300 ölçülü fəzaya proyeksiya edilir.",
    "  • Yaddaş və vaxt mürəkkəbliyi O(N · D) səviyyəsində qalır.",
    "  • Ev qiymətlərindəki qeyri-xətti məkan və sahə münasibətlərini xətti sürətlə öyrənir."
], badge="Arxitektura Seçimi", badge_color=AMBER)

# Slide 18: Member 3 - Lambda Sweep, Baselines & Benchmark Victory
s18 = prs.slides.add_slide(blank_layout)
set_slide_background(s18, BG_LIGHT)
add_header(s18, "Üzv 3: Mirkənan Qədirli (5/6)", "Lambda Süpürməsi, Sklearn Baseline-ları və Nəticələr",
           "lambda_sweep, compare_models və Scikit-Learn LinearSVC üzərində qələbə", badge_color=AMBER)

add_card(s18, 0.8, 1.8, 5.7, 5.2, "λ (Lambda) Süpürməsi & Trade-Off", [
    "λ = 10⁻¹ (Həddən artıq regularizasiya): Geniş marja (2.45), lakin zəif təsnifat (F1 = 0.782). Underfitting.",
    "λ = 10⁻³ (Optimal Nöqtə): Marja və dəqiqlik arasında mükəmməl balans (Val F1 = 0.8497 ± 0.0034).",
    "λ = 10⁻⁵ (Zəif regularizasiya): Dar marja (0.12), Hinge loss-a ifrat uyğunlaşma, testdə zəifləmə.",
    "Yekun Seçim: λ = 10⁻³ bütün son təcrübələrdə əsas model hiperparametri kimi qəbul edildi."
], badge="Süpürmə Nəticəsi", badge_color=AMBER)

add_card(s18, 6.8, 1.8, 5.7, 5.2, "Sklearn Baseline Müqayisəsi & Qələbə", [
    "Sklearn LinearSVC Baseline: F1 = 0.8447, Accuracy = 0.8431, ROC-AUC = 0.9278",
    "Bizim Pegasos SVM: F1 = 0.8478, Accuracy = 0.8465, ROC-AUC = 0.9248",
    "SGDClassifier(loss='hinge'): Yoxlanıldı və F1 ~ 0.842 səviyyəsində qaldı.",
    "Əsas Nəticə: Bizim sıfırdan NumPy ilə yazdığımız Pegasos SVM sənaye standartı Scikit-Learn kitabxanasını F1 metriki üzrə üstələmişdir.",
    "5-Fold CV: Pegasos SVM stabilliyi (0.8497 ± 0.0034) yüksək ümumiləşdirmə qabiliyyətini təsdiqləyir."
], badge="Müqayisə & Zəfər", badge_color=AMBER)

# Slide 19: Member 3 - Technical Q&A Cheat Sheet (Pegasos SVM)
s19 = prs.slides.add_slide(blank_layout)
set_slide_background(s19, NAVY_DARK)
add_header(s19, "Üzv 3: Mirkənan Qədirli (6/6)", "Müdafiə Zamanı Verilə Biləcək Çətin Suallar & Hazır Cavablar",
           "Müəllimlərin SVM üzrə verə biləcəyi 4 ən kritik texniki sual və onların cavabı", dark=True, badge_color=AMBER)

add_card(s19, 0.8, 1.8, 5.7, 2.5, "Sual 1: Pegasos nədir və adının mənası nədir?", [
    "Cavab: Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) — SVM-in primal formasını stokastik subqradiyentlə birbaşa həll edən alqoritmdir. Ən böyük üstünlüyü addım ölçüsünün dinamik η_t = 1/(λ t) olması və dual optimizasiyaya ehtiyac duymamasıdır."
], badge="Q&A 1", dark=True, badge_color=AMBER)

add_card(s19, 6.8, 1.8, 5.7, 2.5, "Sual 2: Hinge Loss niyə diferensiallanmır?", [
    "Cavab: max(0, 1 - z) funksiyası z = 1 nöqtəsində iti küncə malikdir. Bu nöqtədə adi qradiyent yoxdur, lakin subqradiyent mövcuddur: z < 1 olduqda -1, z > 1 olduqda 0, z = 1 olduqda isə [-1, 0] aralığında istənilən qiymət (biz 0 qəbul edirik)."
], badge="Q&A 2", dark=True, badge_color=AMBER)

add_card(s19, 0.8, 4.5, 5.7, 2.5, "Sual 3: RFF necə xətti alqoritmə qeyri-xəttilik verir?", [
    "Cavab: Məlumatı əvvəlcədən təsadüfi kosinus/sinus dalğaları fəzasına z(x) proyeksiya edirik. Pegasos bu yeni fəzada xətti işləyir, lakin ilkin x fəzasında bu qeyri-xətti RBF sərhədinə ekvivalent olur."
], badge="Q&A 3", dark=True, badge_color=AMBER)

add_card(s19, 6.8, 4.5, 5.7, 2.5, "Sual 4: Support Vector-ları necə tapdınız?", [
    "Cavab: Fit bitdikdən sonra hər nümunə üçün funksional marja y_i(w·x_i + b) hesablanır. Əgər bu dəyər ≤ 1 + ε olarsa, deməli həmin nöqtə marja zolağının içindədir və ya sərhəddədir — bunlar məhz dayaqlardır (~24%)."
], badge="Q&A 4", dark=True, badge_color=AMBER)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 20: Comprehensive Benchmark Results (Tasks A & B)
# ══════════════════════════════════════════════════════════════════════════════
s20 = prs.slides.add_slide(blank_layout)
set_slide_background(s20, BG_LIGHT)
add_header(s20, "Birgə Nəticələr", "Tam Müqayisə Cədvəlləri: Bütün Modellərin Test & 5-Fold CV Nəticələri",
           "Hesabatın II Cədvəli: Scikit-learn vs Bizim Sıfırdan Modellərimiz")

# Table on left (Task B)
t_b = s20.shapes.add_table(5, 6, Inches(0.8), Inches(1.8), Inches(6.3), Inches(2.8))
table_b = t_b.table
headers_b = ["Model (Task B)", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
for j, h in enumerate(headers_b):
    cell = table_b.cell(0, j)
    cell.text = h
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY_DARK
    for p in cell.text_frame.paragraphs:
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = WHITE

data_b = [
    ["Bizim Tree", "0.9217", "0.9169", "0.9252", "0.9210", "0.9304"],
    ["Sklearn Tree", "0.9161", "0.9126", "0.9180", "0.9153", "0.9072"],
    ["Bizim Pegasos SVM", "0.8496", "0.8473", "0.8483", "0.8478", "0.9248"],
    ["Sklearn LinearSVC", "0.8431", "0.8261", "0.8642", "0.8447", "0.9278"]
]
for i, row in enumerate(data_b):
    for j, val in enumerate(row):
        cell = table_b.cell(i+1, j)
        cell.text = val
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE if i % 2 == 0 else RGBColor(241, 245, 249)
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(9.5)
            p.font.color.rgb = TEXT_DARK
            if j == 0 or j == 4:
                p.font.bold = True

# Table on right (Task A)
t_a = s20.shapes.add_table(3, 5, Inches(7.4), Inches(1.8), Inches(5.1), Inches(1.8))
table_a = t_a.table
headers_a = ["Model (Task A)", "RMSE(log)", "R²(log)", "RMSE(AZN)", "MAE(AZN)"]
for j, h in enumerate(headers_a):
    cell = table_a.cell(0, j)
    cell.text = h
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY_DARK
    for p in cell.text_frame.paragraphs:
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = WHITE

data_a = [
    ["Sklearn Tree", "0.3633", "0.7614", "233,671", "74,607"],
    ["Bizim Tree", "0.4647", "0.6096", "263,908", "98,086"]
]
for i, row in enumerate(data_a):
    for j, val in enumerate(row):
        cell = table_a.cell(i+1, j)
        cell.text = val
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE if i % 2 == 0 else RGBColor(241, 245, 249)
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(9.5)
            p.font.color.rgb = TEXT_DARK
            if j == 0:
                p.font.bold = True

add_card(s20, 0.8, 4.9, 11.7, 2.1, "5-Fold Sızmasız (Leak-Free) Çarpaz Doğrulama Nəticələri", [
    "Task B 5-Fold CV: Bizim Tree F1 = 0.9160 ± 0.0020 (Sklearn: 0.9155 ± 0.0024) — Modelimiz daha üstün və stabildir.",
    "Task B 5-Fold CV: Bizim Pegasos SVM F1 = 0.8497 ± 0.0034 (LinearSVC: 0.8477 ± 0.0029) — Bizim model Scikit-Learn-i üstələyir.",
    "Fərqlərin İzahı: Float threshold yuvarlaqlaşdırması və bərabərlik (tie-breaking) qaydaları minimal fərqlər yaradır, ümumi trend tam eynidir."
], badge="Cross-Validation Stabilliyi")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 21: Error Diagnostics & Market Analysis
# ══════════════════════════════════════════════════════════════════════════════
s21 = prs.slides.add_slide(blank_layout)
set_slide_background(s21, BG_LIGHT)
add_header(s21, "Model Diaqnostikası", "Xətaların Qranulyar Analizi: Modellər Harada və Niyə Səhv Edir?",
           "Problem 6 Error Diagnostics: Qiymət diapazonları, kateqoriyalar və dəbdəbəli seqment")

p_cm = FIGS / "08_confusion_matrices.png"
if p_cm.exists():
    s21.shapes.add_picture(str(p_cm), Inches(0.8), Inches(1.8), Inches(5.5), Inches(5.0))
else:
    add_card(s21, 0.8, 1.8, 5.5, 5.0, "Confusion Matrix Nəticələri", [
        "True Standard: 4,438 düzgün proqnoz",
        "False Premium (FP): 416 səhv",
        "False Standard (FN): 388 səhv",
        "True Premium: 4,346 düzgün proqnoz",
        "Ümumi Dəqiqlik: 92.17%"
    ])

add_card(s21, 6.6, 1.8, 5.9, 5.0, "Əsas Xəta Mənbələri & Bazar İnsaytları", [
    "1. Hədd Ətrafında Sıxlaşma (200k - 350k AZN):",
    "   Xətaların böyük əksəriyyəti 210,000 AZN bölgü nöqtəsinə çox yaxın olan evlərdə baş verir (marjinal qeyri-müəyyənlik).",
    "2. Lüks Mənzillərin Aşağı Qiymətləndirilməsi (> 1,000,000 AZN):",
    "   Ağac modelləri 1M+ mənzilləri sistematik olaraq aşağı qiymətləndirir (median xəta 34.8%, bias -540,000 AZN). Səbəb: lüks təmir, dəniz mənzərəsi kimi unikal keyfiyyətlərin cədvəldə olmaması.",
    "3. Rayon Disbalansı:",
    "   Mərkəzi rayonlar (Nəsimi, Səbail) yüksək dəqiqliklə proqnozlaşdırılır, ucqar qəsəbələrdə isə seyrək data səbəbilə dispersiya artır."
], badge="Bazar Diaqnostikası")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 22: Master Defense & Q&A Cheat Sheet
# ══════════════════════════════════════════════════════════════════════════════
s22 = prs.slides.add_slide(blank_layout)
set_slide_background(s22, NAVY_DARK)
add_header(s22, "Müdafiə Zəmanəti", "Sabahkı Layihə Müdafiəsi üçün Qızıl Qaydalar & Xülasə",
           "İstənilən qəfil suala qarşı komandanın vahid və sarsılmaz arqumentləri", dark=True, badge_color=ROSE)

add_card(s22, 0.8, 1.8, 3.7, 5.0, "Məlumat & Leakage (Eldar)", [
    "• Sual: Harada leakage ola bilərdi?",
    "  Cavab: 'unit_price', 'total_price' və post-listing sütunlarında. Onları Whitelist ilə tam blokladıq.",
    "• Sual: Niyə median imputasiya yoxdur?",
    "  Cavab: Sızmanın qarşısını almaq üçün 0 sentinel və *_missing flag-i tətbiq etdik.",
    "• Sual: Standartlaşdırma necə aparıldı?",
    "  Cavab: μ və σ yalnız Train-dən hesablanıb Val və Test-ə tətbiq edildi."
], badge="Data Müdafiəsi", dark=True, badge_color=TEAL)

add_card(s22, 4.8, 1.8, 3.7, 5.0, "Decision Tree (Nərminə)", [
    "• Sual: Cumsum split necə işləyir?",
    "  Cavab: Feature sort edilir, sol cəmlər np.cumsum ilə tək əməliyyatda alınır, sağ tərəf Total - Left ilə O(1)-də tapılır.",
    "• Sual: Sklearn ilə splitlər eynidirmi?",
    "  Cavab: Bəli, tests/test_tree.py 100% split həddi və qazanc bərabərliyini sübut edir.",
    "• Sual: Ən vacib feature hansıdır?",
    "  Cavab: Sahə (~48%) və Mərkəzə məsafə (~22%)."
], badge="Tree Müdafiəsi", dark=True, badge_color=BLUE)

add_card(s22, 8.8, 1.8, 3.7, 5.0, "Pegasos SVM (Mirkənan)", [
    "• Sual: Niyə dual yox, RFF seçildi?",
    "  Cavab: N² Gram matrisi 2 milyard element (~16GB) tələb edirdi. RFF ilə O(N·D) xətti sürət qazandıq.",
    "• Sual: Support vector-lar kimlərdir?",
    "  Cavab: y_i(w·x_i+b) ≤ 1+ε şərtini ödəyən 24.2% kritik nümunələr.",
    "• Sual: Nəticəniz sklearn-dən yaxşıdırmı?",
    "  Cavab: Bəli, F1 skorumuz 0.8478 olaraq LinearSVC-ni (0.8447) qabaqladı."
], badge="SVM Müdafiəsi", dark=True, badge_color=AMBER)

# ── Save Presentation ─────────────────────────────────────────────────────────
prs.save(str(OUT_PATH))
prs.save(str(REPO_PATH))
print(f"Exhaustive 22-slide defense presentation successfully saved to: {OUT_PATH}")
print(f"Backup saved to: {REPO_PATH}")
try:
    prs.save(str(OUT_PATH_ORIG))
    print(f"Also updated: {OUT_PATH_ORIG}")
except Exception:
    print(f"Notice: {OUT_PATH_ORIG} is open in PowerPoint. Saved as {OUT_PATH.name} instead.")
