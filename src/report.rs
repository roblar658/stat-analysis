use crate::auto::AutoAnalysisReport;
use crate::stats::clustering::ClusteringReport;
use crate::stats::factor::FactorSuitabilityReport;
use crate::stats::logistic::LogisticReport;
use crate::stats::missing::MissingDataReport;
use crate::stats::normality::{
    BootstrapResult, KruskalWallisResult, MannWhitneyResult, NormalityCheck,
};
use crate::stats::regression::LinearRegressionFullReport;
use crate::stats::sphericity::SphericityReport;
use crate::stats::survival::CoxSurvivalReport;
use crate::stats::variance::VarianceHomogeneityReport;
use std::fmt::Write as FmtWrite;
use std::io::{Cursor, Write as IoWrite};
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

#[derive(Clone)]
pub enum ManualCalcResult {
    Normality {
        check: NormalityCheck,
        boot: Option<BootstrapResult>,
        values: Vec<f64>,
    },
    GroupVariance {
        variance: VarianceHomogeneityReport,
        mw: Option<MannWhitneyResult>,
        kw: Option<KruskalWallisResult>,
        groups: Vec<(String, Vec<f64>)>,
    },
    Sphericity(SphericityReport),
    Regression(LinearRegressionFullReport),
    Missing(MissingDataReport),
    Factor(FactorSuitabilityReport),
    Logistic(LogisticReport),
    Survival(CoxSurvivalReport),
    Clustering(ClusteringReport),
}

#[derive(Clone)]
pub struct ManualHistoryEntry {
    pub timestamp: String,
    pub title: String,
    pub parameters: String,
    pub result: ManualCalcResult,
}

pub fn render_markdown_report(rep: &AutoAnalysisReport) -> String {
    render_markdown_report_with_appendix(rep, &[])
}

pub fn render_markdown_report_with_appendix(
    rep: &AutoAnalysisReport,
    manual_entries: &[ManualHistoryEntry],
) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Statistisk Diagnostikk- og Modellrapport");
    let _ = writeln!(md, "- **Fil:** `{}`", rep.filename);
    let _ = writeln!(md, "- **Generert:** {}", rep.generated_at);
    let _ = writeln!(md, "- **Dimensjoner:** {} rader × {} kolonner\n", rep.row_count, rep.col_count);

    let _ = writeln!(md, "## Oppsummering av alle sjekker\n");
    let _ = writeln!(md, "| Kategori | Sjekk | Status | Resultat | Anbefalt tiltak |");
    let _ = writeln!(md, "|---|---|---|---|---|");
    for b in &rep.summary_badges {
        let _ = writeln!(
            md,
            "| {} | {} | **{}** | {} | {} |",
            b.category,
            b.check_name,
            b.status.to_uppercase(),
            b.headline,
            b.action
        );
    }
    md.push('\n');

    if !rep.normality_checks.is_empty() {
        let _ = writeln!(md, "## 1. Normalfordeling (Shapiro-Wilk / Kolmogorov-Smirnov)\n");
        let _ = writeln!(md, "| Variabel | n | Test | Statistikk | p-verdi | Skjevhet | Kurtose | Status |");
        let _ = writeln!(md, "|---|---:|---|---:|---:|---:|---:|---|");
        for nc in &rep.normality_checks {
            let _ = writeln!(
                md,
                "| {} | {} | {} | {:.4} | {:.4} | {:.3} | {:.3} | {} |",
                nc.variable, nc.n, nc.recommended_test, nc.primary_statistic, nc.primary_p_value, nc.skewness, nc.excess_kurtosis, nc.status.to_uppercase()
            );
        }
        md.push('\n');
    }

    if let Some(ref vh) = rep.variance_homogeneity {
        let _ = writeln!(md, "## 2. Varianshomogenitet ({} per {})\n", vh.value_col, vh.group_col);
        let _ = writeln!(md, "- **Levene (median / Brown-Forsythe):** F({}, {}) = {:.3}, p = {:.4}", vh.levene_median.df1, vh.levene_median.df2, vh.levene_median.f_stat, vh.levene_median.p_value);
        let _ = writeln!(md, "- **Levene (gjennomsnitt):** F({}, {}) = {:.3}, p = {:.4}", vh.levene_mean.df1, vh.levene_mean.df2, vh.levene_mean.f_stat, vh.levene_mean.p_value);
        let _ = writeln!(md, "- **Bartletts test:** χ²({}) = {:.3}, p = {:.4}", vh.bartlett.df, vh.bartlett.chi2_stat, vh.bartlett.p_value);
        if let Some(ref wt) = vh.welch_t_test {
            let _ = writeln!(md, "- **Welchs t-test (fallback):** t({:.1}) = {:.3}, p = {:.4}", wt.df, wt.t_stat, wt.p_value);
        }
        let _ = writeln!(md, "- **Welchs ANOVA (fallback):** F({:.1}, {:.1}) = {:.3}, p = {:.4}", vh.welch_anova.df1, vh.welch_anova.df2, vh.welch_anova.f_stat, vh.welch_anova.p_value);
        if let Some(ref fb) = rep.group_fallbacks {
            if let Some(ref mw) = fb.mann_whitney {
                let _ = writeln!(md, "- **Mann-Whitney U (ikke-parametrisk):** U = {:.1}, Z = {:.3}, p = {:.4}", mw.u_stat, mw.z_stat, mw.p_value);
            }
            if let Some(ref kw) = fb.kruskal_wallis {
                let _ = writeln!(md, "- **Kruskal-Wallis H (ikke-parametrisk):** H({}) = {:.3}, p = {:.4}", kw.df, kw.h_stat, kw.p_value);
            }
        }
        let _ = writeln!(md, "- **Anbefaling:** {}\n", vh.recommendation);
    }

    if let Some(ref sp) = rep.sphericity {
        let _ = writeln!(md, "## 3. Sfærisitet — Repeated Measures ANOVA ({})\n", sp.columns.join(", "));
        let _ = writeln!(md, "- **Mauchlys W:** {:.4}, χ²({}) = {:.3}, p = {:.4}", sp.mauchly_w, sp.chi2_df, sp.chi2_stat, sp.mauchly_p_value);
        let _ = writeln!(md, "- **RM-ANOVA F:** {:.3} (ukorrigert p = {:.4})", sp.rm_anova_f, sp.p_uncorrected);
        let _ = writeln!(md, "- **Greenhouse-Geisser:** ε = {:.3}, korrigert p = {:.4}", sp.epsilon_gg, sp.p_greenhouse_geisser);
        let _ = writeln!(md, "- **Huynh-Feldt:** ε = {:.3}, korrigert p = {:.4}", sp.epsilon_hf, sp.p_huynh_feldt);
        let _ = writeln!(md, "- **Anbefaling:** {}\n", sp.recommendation);
    }

    if let Some(ref reg) = rep.linear_regression {
        let _ = writeln!(md, "## 4. Lineær regresjon og forutsetninger (Y = {})\n", reg.target);
        let _ = writeln!(md, "- **Forklart varians:** R² = {:.4}, Justert R² = {:.4} (F = {:.3}, p = {:.4})", reg.r_squared, reg.adj_r_squared, reg.f_stat, reg.f_p_value);
        let _ = writeln!(md, "- **Prediksjonsfeil:** RMSE = {:.4}, MAE = {:.4}", reg.rmse, reg.mae);
        let _ = writeln!(md, "- **Informasjonskriterier:** AIC = {:.2}, BIC = {:.2}", reg.aic, reg.bic);
        let _ = writeln!(md, "- **Residualnormalitet:** {}", reg.residual_normality.interpretation);
        if let Some(ref mc) = reg.multicollinearity {
            let _ = writeln!(md, "- **Multikollinearitet:** {} {}", mc.interpretation, mc.recommendation);
        }
        let _ = writeln!(md, "- **Autokorrelasjon:** {} {}", reg.autocorrelation.interpretation, reg.autocorrelation.recommendation);
        let _ = writeln!(md, "- **Heteroskedastisitet:** {} {}", reg.heteroscedasticity.interpretation, reg.heteroscedasticity.recommendation);
        let _ = writeln!(md, "- **Uteliggere:** {} {}\n", reg.outliers.interpretation, reg.outliers.recommendation);

        let _ = writeln!(md, "### Sammenligning av modeller (AIC / BIC)\n");
        let _ = writeln!(md, "| Modell | Parametere | R² | Just. R² | RMSE | AIC | BIC |");
        let _ = writeln!(md, "|---|---:|---:|---:|---:|---:|---:|");
        for m in &reg.model_comparison {
            let _ = writeln!(md, "| {} | {} | {:.4} | {:.4} | {:.4} | {:.2} | {:.2} |", m.model_name, m.k_params, m.r_squared, m.adj_r_squared, m.rmse, m.aic, m.bic);
        }
        md.push('\n');
    }

    let _ = writeln!(md, "## 5. Manglende data og Littles MCAR-test\n");
    let _ = writeln!(md, "- {}", rep.missing_data.interpretation);
    let _ = writeln!(md, "- **Anbefaling:** {}\n", rep.missing_data.recommendation);

    if let Some(ref fac) = rep.factor_suitability {
        let _ = writeln!(md, "## 6. Egnethet for faktoranalyse / PCA\n");
        let _ = writeln!(md, "- {}", fac.interpretation);
        let _ = writeln!(md, "- **Anbefaling:** {}\n", fac.recommendation);
    }

    if let Some(ref lr) = rep.logistic_regression {
        let _ = writeln!(md, "## 7. Logistisk regresjon ({})\n", lr.target);
        let _ = writeln!(md, "- {}", lr.interpretation);
        let _ = writeln!(md, "- **AIC:** {:.2}, **BIC:** {:.2}, **Nøyaktighet:** {:.1}%, **F1:** {:.3}", lr.aic, lr.bic, lr.confusion_matrix.accuracy * 100.0, lr.confusion_matrix.f1_score);
        let _ = writeln!(md, "- **Anbefaling:** {}\n", lr.recommendation);
    }

    if let Some(ref cox) = rep.cox_survival {
        let _ = writeln!(md, "## 8. Overlevelsesanalyse — Cox PH ({} / {})\n", cox.time_col, cox.event_col);
        let _ = writeln!(md, "- {}", cox.interpretation);
        let _ = writeln!(md, "- **AIC:** {:.2}, **BIC:** {:.2}, **Hendelser:** {}/{}", cox.aic, cox.bic, cox.n_events, cox.n);
        let _ = writeln!(md, "- **Anbefaling:** {}\n", cox.recommendation);
    }

    if let Some(ref cl) = rep.clustering {
        let _ = writeln!(md, "## 9. Klyngeanalyse (Silhouette / Elbow)\n");
        let _ = writeln!(md, "- {}", cl.interpretation);
        let _ = writeln!(md, "- **Anbefaling:** {}\n", cl.recommendation);
    }

    if !manual_entries.is_empty() {
        let _ = writeln!(md, "---\n\n# Appendiks — Manuelle utregninger ({} stk.)\n", manual_entries.len());
        for (idx, entry) in manual_entries.iter().enumerate() {
            let _ = writeln!(md, "## Appendiks A.{}: {} *(kl. {})*", idx + 1, entry.title, entry.timestamp);
            let _ = writeln!(md, "- **Parametere:** {}", entry.parameters);
            match &entry.result {
                ManualCalcResult::Normality { check, boot, .. } => {
                    let _ = writeln!(md, "- **Resultat:** {}", check.interpretation);
                    if let Some(b) = boot {
                        let _ = writeln!(md, "- **Bootstrap 95% KI (B={}):** [{:.4}, {:.4}] (SE = {:.4})", b.resamples, b.ci_low_95, b.ci_high_95, b.bootstrap_se);
                    }
                    let _ = writeln!(md, "- **Anbefaling:** {}\n", check.recommendation);
                }
                ManualCalcResult::GroupVariance { variance, mw, kw, .. } => {
                    let _ = writeln!(md, "- **Resultat:** {}", variance.interpretation);
                    if let Some(m) = mw {
                        let _ = writeln!(md, "- **Mann-Whitney U:** U = {:.1}, p = {:.4}", m.u_stat, m.p_value);
                    }
                    if let Some(k) = kw {
                        let _ = writeln!(md, "- **Kruskal-Wallis H:** H({}) = {:.3}, p = {:.4}", k.df, k.h_stat, k.p_value);
                    }
                    let _ = writeln!(md, "- **Anbefaling:** {}\n", variance.recommendation);
                }
                ManualCalcResult::Sphericity(sp) => {
                    let _ = writeln!(md, "- **Resultat:** {}", sp.interpretation);
                    let _ = writeln!(md, "- **Anbefaling:** {}\n", sp.recommendation);
                }
                ManualCalcResult::Regression(reg) => {
                    let _ = writeln!(md, "- **Tilpasning:** R² = {:.4}, Justert R² = {:.4}, RMSE = {:.4}, MAE = {:.4}, AIC = {:.2}, BIC = {:.2}", reg.r_squared, reg.adj_r_squared, reg.rmse, reg.mae, reg.aic, reg.bic);
                    let _ = writeln!(md, "- **Autokorrelasjon:** {}", reg.autocorrelation.interpretation);
                    let _ = writeln!(md, "- **Heteroskedastisitet:** {}", reg.heteroscedasticity.interpretation);
                    let _ = writeln!(md, "- **Uteliggere:** {}\n", reg.outliers.interpretation);
                }
                ManualCalcResult::Missing(m) => {
                    let _ = writeln!(md, "- **Resultat:** {}", m.interpretation);
                    let _ = writeln!(md, "- **Anbefaling:** {}\n", m.recommendation);
                }
                ManualCalcResult::Factor(fac) => {
                    let _ = writeln!(md, "- **Resultat:** {}", fac.interpretation);
                    let _ = writeln!(md, "- **Anbefaling:** {}\n", fac.recommendation);
                }
                ManualCalcResult::Logistic(lr) => {
                    let _ = writeln!(md, "- **Resultat:** {}", lr.interpretation);
                    let _ = writeln!(md, "- **AIC:** {:.2}, **BIC:** {:.2}, **F1:** {:.3}", lr.aic, lr.bic, lr.confusion_matrix.f1_score);
                    let _ = writeln!(md, "- **Anbefaling:** {}\n", lr.recommendation);
                }
                ManualCalcResult::Survival(cox) => {
                    let _ = writeln!(md, "- **Resultat:** {}", cox.interpretation);
                    let _ = writeln!(md, "- **Anbefaling:** {}\n", cox.recommendation);
                }
                ManualCalcResult::Clustering(cl) => {
                    let _ = writeln!(md, "- **Resultat:** {}", cl.interpretation);
                    let _ = writeln!(md, "- **Anbefaling:** {}\n", cl.recommendation);
                }
            }
        }
    }

    md
}

// =========================================================================
// Pure-Rust Vector PDF 1.4 Generator (A4, WinAnsiEncoding for Norwegian æøå)
// =========================================================================

const PAGE_W: f64 = 595.28;
const PAGE_H: f64 = 841.89;
const MARGIN_L: f64 = 44.0;
const MARGIN_R: f64 = 44.0;
const MARGIN_B: f64 = 48.0;
const CONTENT_W: f64 = PAGE_W - MARGIN_L - MARGIN_R;

/// Encode a UTF-8 string into PDF literal string bytes using WinAnsiEncoding (supports æøåÆØÅ²±×–—•).
fn pdf_encode_winansi(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() + 8);
    out.push(b'(');
    let mut push_byte = |b: u8| {
        match b {
            b'(' | b')' | b'\\' => {
                out.push(b'\\');
                out.push(b);
            }
            0x20..=0x7E => out.push(b),
            _ => {
                // Octal escape for high WinAnsi bytes so stream stays pure 7-bit clean
                out.extend_from_slice(format!("\\{:03o}", b).as_bytes());
            }
        }
    };

    for ch in s.chars() {
        match ch {
            ' '..='~' => push_byte(ch as u8),
            // Latin-1 / WinAnsi 0xA0..=0xFF (includes æ=0xE6, ø=0xF8, å=0xE5, Æ=0xC6, Ø=0xD8, Å=0xC5, ²=0xB2, ±=0xB1, ×=0xD7)
            '\u{00A0}'..='\u{00FF}' => push_byte(ch as u8),
            '\u{2013}' => push_byte(0x96), // en-dash –
            '\u{2014}' => push_byte(0x97), // em-dash —
            '\u{2022}' => push_byte(0x95), // bullet •
            '\u{2026}' => push_byte(0x85), // ellipsis …
            '\u{20AC}' => push_byte(0x80), // €
            '\u{2018}' => push_byte(0x91),
            '\u{2019}' => push_byte(0x92),
            '\u{201C}' => push_byte(0x93),
            '\u{201D}' => push_byte(0x94),
            // Math/Greek symbols cleanly transliterated
            'χ' => { for b in b"chi" { push_byte(*b); } }
            'β' => { for b in b"beta" { push_byte(*b); } }
            'ε' => { for b in b"eps" { push_byte(*b); } }
            'λ' => { for b in b"lambda" { push_byte(*b); } }
            'φ' => { for b in b"phi" { push_byte(*b); } }
            'α' => { for b in b"alpha" { push_byte(*b); } }
            '→' => { for b in b"->" { push_byte(*b); } }
            '≤' => { for b in b"<=" { push_byte(*b); } }
            '≥' => { for b in b">=" { push_byte(*b); } }
            'Ŷ' => { for b in b"Y-hat" { push_byte(*b); } }
            _ => push_byte(b'?'),
        }
    }
    out.push(b')');
    out
}

fn wrap_text(text: &str, max_chars: usize) -> Vec<String> {
    let limit = max_chars.max(4);
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if word.chars().count() > limit {
            if !cur.is_empty() {
                lines.push(cur);
                cur = String::new();
            }
            let mut chunk = String::new();
            for ch in word.chars() {
                chunk.push(ch);
                if chunk.chars().count() >= limit {
                    lines.push(chunk);
                    chunk = String::new();
                }
            }
            if !chunk.is_empty() {
                cur = chunk;
            }
        } else if cur.is_empty() {
            cur.push_str(word);
        } else if cur.chars().count() + 1 + word.chars().count() <= limit {
            cur.push(' ');
            cur.push_str(word);
        } else {
            lines.push(cur);
            cur = word.to_string();
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

struct PdfBuilder {
    pages: Vec<Vec<u8>>,
    cur: Vec<u8>,
    y: f64,
}

impl PdfBuilder {
    fn new() -> Self {
        Self {
            pages: Vec::new(),
            cur: Vec::new(),
            y: PAGE_H - 44.0,
        }
    }

    fn new_page(&mut self) {
        if !self.cur.is_empty() {
            let finished = std::mem::take(&mut self.cur);
            self.pages.push(finished);
        }
        self.y = PAGE_H - 46.0;
    }

    fn ensure_space(&mut self, needed: f64) {
        if self.y - needed < MARGIN_B {
            self.new_page();
        }
    }

    fn cmd(&mut self, s: &str) {
        self.cur.extend_from_slice(s.as_bytes());
        self.cur.push(b'\n');
    }

    fn draw_text(&mut self, x: f64, y: f64, font: &str, size: f64, r: f64, g: f64, b: f64, text: &str) {
        let enc = pdf_encode_winansi(text);
        let prefix = format!("BT /{font} {size:.1} Tf {r:.3} {g:.3} {b:.3} rg {x:.2} {y:.2} Td ");
        self.cur.extend_from_slice(prefix.as_bytes());
        self.cur.extend_from_slice(&enc);
        self.cur.extend_from_slice(b" Tj ET\n");
    }

    fn draw_rect_fill(&mut self, x: f64, y: f64, w: f64, h: f64, r: f64, g: f64, b: f64) {
        self.cmd(&format!("{r:.3} {g:.3} {b:.3} rg {x:.2} {y:.2} {w:.2} {h:.2} re f"));
    }

    fn draw_rect_stroke(&mut self, x: f64, y: f64, w: f64, h: f64, lw: f64, r: f64, g: f64, b: f64) {
        self.cmd(&format!("{lw:.2} w {r:.3} {g:.3} {b:.3} RG {x:.2} {y:.2} {w:.2} {h:.2} re S"));
    }

    fn draw_line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, lw: f64, r: f64, g: f64, b: f64) {
        self.cmd(&format!("{lw:.2} w {r:.3} {g:.3} {b:.3} RG {x1:.2} {y1:.2} m {x2:.2} {y2:.2} l S"));
    }

    fn section_heading(&mut self, title: &str) {
        self.ensure_space(34.0);
        self.y -= 8.0;
        self.draw_text(MARGIN_L, self.y, "F2", 13.0, 0.114, 0.306, 0.847, title);
        self.y -= 5.0;
        self.draw_line(MARGIN_L, self.y, PAGE_W - MARGIN_R, self.y, 0.9, 0.796, 0.835, 0.882);
        self.y -= 13.0;
    }

    fn paragraph(&mut self, text: &str, bold_prefix: Option<&str>) {
        let full = match bold_prefix {
            Some(p) => format!("{}: {}", p, text),
            None => text.to_string(),
        };
        let lines = wrap_text(&full, 96);
        self.ensure_space(lines.len() as f64 * 13.0 + 4.0);
        for (i, line) in lines.iter().enumerate() {
            if i == 0 && bold_prefix.is_some() {
                if let Some(p) = bold_prefix {
                    let prefix_str = format!("{}: ", p);
                    if let Some(rest) = line.strip_prefix(&prefix_str) {
                        self.draw_text(MARGIN_L, self.y, "F2", 9.5, 0.059, 0.090, 0.165, &prefix_str);
                        let est_w = prefix_str.chars().count() as f64 * 5.05;
                        self.draw_text(MARGIN_L + est_w, self.y, "F1", 9.5, 0.200, 0.255, 0.333, rest);
                        self.y -= 13.0;
                        continue;
                    }
                }
            }
            self.draw_text(MARGIN_L, self.y, "F1", 9.5, 0.200, 0.255, 0.333, line);
            self.y -= 13.0;
        }
        self.y -= 2.0;
    }

    fn status_rgb(status: &str) -> ((f64, f64, f64), (f64, f64, f64), &'static str) {
        match status {
            "ok" => ((0.863, 0.988, 0.906), (0.078, 0.325, 0.176), "OK"),
            "warning" => ((0.996, 0.953, 0.780), (0.471, 0.208, 0.059), "ADVARSEL"),
            "fail" => ((0.996, 0.886, 0.886), (0.498, 0.114, 0.114), "BRUTT"),
            _ => ((0.886, 0.910, 0.941), (0.200, 0.255, 0.333), "N/A"),
        }
    }

    fn draw_table_header(&mut self, headers: &[&str], col_widths: &[f64]) {
        let line_h = 10.5;
        let wrapped_h: Vec<Vec<String>> = headers
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                let w = *col_widths.get(i).unwrap_or(&60.0);
                let max_c = ((w - 8.0) / 4.8).floor().max(5.0) as usize;
                wrap_text(h, max_c)
            })
            .collect();
        let max_lines = wrapped_h.iter().map(|v| v.len()).max().unwrap_or(1);
        let hdr_h = max_lines as f64 * line_h + 7.0;

        self.ensure_space(hdr_h + 18.0);
        let box_bottom = self.y - hdr_h + 4.0;
        self.draw_rect_fill(MARGIN_L, box_bottom, CONTENT_W, hdr_h, 0.925, 0.949, 0.976);
        let mut cx = MARGIN_L + 4.0;
        for (i, lines) in wrapped_h.iter().enumerate() {
            for (li, line) in lines.iter().enumerate() {
                let ly = self.y - 6.5 - (li as f64) * line_h;
                self.draw_text(cx, ly, "F2", 8.2, 0.059, 0.090, 0.165, line);
            }
            cx += *col_widths.get(i).unwrap_or(&60.0);
        }
        self.draw_line(MARGIN_L, box_bottom, PAGE_W - MARGIN_R, box_bottom, 0.8, 0.720, 0.776, 0.843);
        self.y -= hdr_h;
    }

    fn table(&mut self, headers: &[&str], col_widths: &[f64], rows: &[Vec<String>]) {
        let line_h = 10.5;
        self.draw_table_header(headers, col_widths);

        for (ri, row) in rows.iter().enumerate() {
            // Wrap every cell completely — NO truncation with "..."!
            let wrapped_cells: Vec<Vec<String>> = row
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    let w = *col_widths.get(i).unwrap_or(&60.0);
                    let max_c = ((w - 8.0) / 4.45).floor().max(5.0) as usize;
                    wrap_text(cell, max_c)
                })
                .collect();
            let max_lines = wrapped_cells.iter().map(|v| v.len()).max().unwrap_or(1);
            let row_h = (max_lines as f64) * line_h + 7.0;

            if self.y - row_h < MARGIN_B {
                self.new_page();
                self.draw_table_header(headers, col_widths);
            }

            let box_bottom = self.y - row_h + 4.0;
            if ri % 2 == 1 {
                self.draw_rect_fill(MARGIN_L, box_bottom, CONTENT_W, row_h, 0.976, 0.984, 0.992);
            }

            let mut rcx = MARGIN_L + 4.0;
            for (i, lines) in wrapped_cells.iter().enumerate() {
                let w = *col_widths.get(i).unwrap_or(&60.0);
                let raw_cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
                let upper = raw_cell.trim().to_ascii_uppercase();
                if upper == "OK" || upper == "ADVARSEL" || upper == "BRUTT" || upper == "FAIL" || upper == "WARNING" {
                    let st = match upper.as_str() {
                        "OK" => "ok",
                        "ADVARSEL" | "WARNING" => "warning",
                        _ => "fail",
                    };
                    let (bg, fg, lbl) = Self::status_rgb(st);
                    let badge_w = (w - 8.0).clamp(32.0, 50.0);
                    let badge_y = self.y - 11.0;
                    self.draw_rect_fill(rcx - 1.0, badge_y, badge_w, 12.0, bg.0, bg.1, bg.2);
                    self.draw_text(rcx + 3.0, badge_y + 3.0, "F2", 7.4, fg.0, fg.1, fg.2, lbl);
                } else {
                    for (li, line) in lines.iter().enumerate() {
                        let ly = self.y - 7.0 - (li as f64) * line_h;
                        self.draw_text(rcx, ly, "F1", 8.1, 0.118, 0.161, 0.231, line);
                    }
                }
                rcx += w;
            }
            self.draw_line(MARGIN_L, box_bottom, PAGE_W - MARGIN_R, box_bottom, 0.4, 0.860, 0.890, 0.925);
            self.y -= row_h;
        }
        self.y -= 8.0;
    }

    /// Vector scatter/line chart embedded directly on the PDF page
    fn draw_mini_chart_box(
        &mut self,
        x: f64,
        y_top: f64,
        w: f64,
        h: f64,
        title: &str,
        x_label: &str,
        y_label: &str,
    ) -> (f64, f64, f64, f64) {
        self.draw_rect_fill(x, y_top - h, w, h, 0.992, 0.995, 0.998);
        self.draw_rect_stroke(x, y_top - h, w, h, 0.7, 0.796, 0.835, 0.882);
        self.draw_text(x + 8.0, y_top - 13.0, "F2", 9.0, 0.059, 0.090, 0.165, title);
        let px = x + 34.0;
        let py = y_top - h + 22.0;
        let pw = w - 46.0;
        let ph = h - 42.0;
        self.draw_line(px, py, px + pw, py, 0.7, 0.200, 0.255, 0.333);
        self.draw_line(px, py, px, py + ph, 0.7, 0.200, 0.255, 0.333);
        self.draw_text(px + pw / 2.0 - 25.0, y_top - h + 6.0, "F1", 7.5, 0.35, 0.42, 0.50, x_label);
        self.draw_text(x + 4.0, py + ph + 3.0, "F1", 7.2, 0.35, 0.42, 0.50, y_label);
        (px, py, pw, ph)
    }

    fn finish(mut self) -> Vec<u8> {
        if !self.cur.is_empty() {
            let last = std::mem::take(&mut self.cur);
            self.pages.push(last);
        }
        let total_pages = self.pages.len().max(1);
        for (i, page_buf) in self.pages.iter_mut().enumerate() {
            // Footer line & page number
            let footer_line = format!(
                "0.5 w 0.80 0.84 0.88 RG {:.2} 34.0 m {:.2} 34.0 l S\n",
                MARGIN_L,
                PAGE_W - MARGIN_R
            );
            page_buf.extend_from_slice(footer_line.as_bytes());
            let left_txt = pdf_encode_winansi("stat-analysis — Statistisk Diagnostikk & Modellkvalitet");
            page_buf.extend_from_slice(format!("BT /F1 8.0 Tf 0.40 0.45 0.55 rg {:.2} 22.0 Td ", MARGIN_L).as_bytes());
            page_buf.extend_from_slice(&left_txt);
            page_buf.extend_from_slice(b" Tj ET\n");

            let right_txt = pdf_encode_winansi(&format!("Side {} av {}", i + 1, total_pages));
            page_buf.extend_from_slice(format!("BT /F1 8.0 Tf 0.40 0.45 0.55 rg {:.2} 22.0 Td ", PAGE_W - MARGIN_R - 48.0).as_bytes());
            page_buf.extend_from_slice(&right_txt);
            page_buf.extend_from_slice(b" Tj ET\n");
        }

        // Assemble PDF objects:
        // 1: Catalog, 2: Pages, 3: Font F1 (Helvetica), 4: Font F2 (Helvetica-Bold)
        // For each page i: (5 + 2*i) = Page object, (6 + 2*i) = Content stream object
        let mut pdf: Vec<u8> = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");

        let mut offsets: Vec<usize> = vec![0]; // 1-based indexing

        let mut push_obj = |body: &[u8]| {
            offsets.push(pdf.len());
            let obj_num = offsets.len() - 1;
            pdf.extend_from_slice(format!("{} 0 obj\n", obj_num).as_bytes());
            pdf.extend_from_slice(body);
            pdf.extend_from_slice(b"\nendobj\n");
        };

        // 1: Catalog
        push_obj(b"<< /Type /Catalog /Pages 2 0 R >>");

        // 2: Pages
        let mut kids = String::new();
        for i in 0..total_pages {
            let _ = write!(kids, "{} 0 R ", 5 + 2 * i);
        }
        let pages_obj = format!("<< /Type /Pages /Kids [ {}] /Count {} >>", kids, total_pages);
        push_obj(pages_obj.as_bytes());

        // 3 & 4: Fonts with WinAnsiEncoding
        push_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
        push_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>");

        // Pages & Streams
        for (i, stream) in self.pages.iter().enumerate() {
            let content_id = 6 + 2 * i;
            let page_obj = format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {:.2} {:.2}] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {} 0 R >>",
                PAGE_W, PAGE_H, content_id
            );
            push_obj(page_obj.as_bytes());

            let mut content_obj = Vec::with_capacity(stream.len() + 64);
            content_obj.extend_from_slice(format!("<< /Length {} >>\nstream\n", stream.len()).as_bytes());
            content_obj.extend_from_slice(stream);
            content_obj.extend_from_slice(b"endstream");
            push_obj(&content_obj);
        }

        // Cross-reference table
        let xref_pos = pdf.len();
        pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for &off in &offsets[1..] {
            pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
        }

        let trailer = format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        );
        pdf.extend_from_slice(trailer.as_bytes());
        pdf
    }
}

/// Generate a genuine multi-page vector PDF report (A4) in pure Rust.
pub fn render_pdf_report(rep: &AutoAnalysisReport) -> Vec<u8> {
    render_pdf_report_with_appendix(rep, &[])
}

/// Generate a multi-page vector PDF report (A4) including all manual calculations as an Appendix section.
pub fn render_pdf_report_with_appendix(
    rep: &AutoAnalysisReport,
    manual_entries: &[ManualHistoryEntry],
) -> Vec<u8> {
    let mut pdf = PdfBuilder::new();

    // Header Banner
    pdf.draw_rect_fill(MARGIN_L, pdf.y - 54.0, CONTENT_W, 56.0, 0.114, 0.306, 0.847);
    pdf.draw_text(MARGIN_L + 14.0, pdf.y - 22.0, "F2", 18.0, 1.0, 1.0, 1.0, "stat-analysis — Statistisk Diagnostikkrapport");
    pdf.draw_text(
        MARGIN_L + 14.0,
        pdf.y - 40.0,
        "F1",
        9.5,
        0.859,
        0.918,
        0.996,
        &format!(
            "Fil: {}  •  {} rader × {} kolonner  •  Generert: {}",
            rep.filename, rep.row_count, rep.col_count, rep.generated_at
        ),
    );
    pdf.y -= 70.0;

    // Summary Table
    pdf.section_heading("Oppsummering av alle sjekker");
    let sum_rows: Vec<Vec<String>> = rep
        .summary_badges
        .iter()
        .map(|b| {
            vec![
                b.check_name.clone(),
                b.status.to_uppercase(),
                b.headline.clone(),
                b.action.clone(),
            ]
        })
        .collect();
    pdf.table(
        &["Sjekk / Test", "Status", "Resultat", "Anbefalt tiltak"],
        &[132.0, 56.0, 142.0, CONTENT_W - 330.0],
        &sum_rows,
    );

    // 1. Normality
    if !rep.normality_checks.is_empty() {
        pdf.section_heading("1. Normalfordeling (Shapiro-Wilk / Kolmogorov-Smirnov)");
        let rows: Vec<Vec<String>> = rep
            .normality_checks
            .iter()
            .map(|n| {
                vec![
                    n.variable.clone(),
                    format!("{}", n.n),
                    n.recommended_test.clone(),
                    format!("{:.4}", n.primary_statistic),
                    format!("{:.4}", n.primary_p_value),
                    format!("{:.3}", n.skewness),
                    format!("{:.3}", n.excess_kurtosis),
                    n.status.to_uppercase(),
                ]
            })
            .collect();
        pdf.table(
            &["Variabel", "n", "Test", "Stat.", "p-verdi", "Skjevhet", "Kurtose", "Status"],
            &[86.0, 30.0, 128.0, 48.0, 48.0, 52.0, 52.0, CONTENT_W - 444.0],
            &rows,
        );
        if !rep.normality_bootstraps.is_empty() {
            let brows: Vec<Vec<String>> = rep
                .normality_bootstraps
                .iter()
                .map(|b| vec![b.target.clone(), format!("{}", b.resamples), format!("{:.4}", b.observed_mean), format!("{:.4}", b.bootstrap_se), format!("[{:.4}, {:.4}]", b.ci_low_95, b.ci_high_95)])
                .collect();
            pdf.table(&["Bootstrap (ikke-normal)", "B", "Snitt", "SE", "95% KI"], &[140.0, 48.0, 82.0, 82.0, CONTENT_W - 352.0], &brows);
        }
    }

    // 2. Variance Homogeneity
    if let Some(ref vh) = rep.variance_homogeneity {
        pdf.section_heading(&format!("2. Varianshomogenitet ({} per {})", vh.value_col, vh.group_col));
        pdf.paragraph(&vh.interpretation, Some("Resultat"));
        pdf.paragraph(&vh.recommendation, Some("Anbefalt tiltak"));
        let mut rows = vec![
            vec!["Levene (median / Brown-Forsythe)".into(), format!("F = {:.3}", vh.levene_median.f_stat), format!("{}, {}", vh.levene_median.df1, vh.levene_median.df2), format!("{:.4}", vh.levene_median.p_value)],
            vec!["Levene (gjennomsnitt)".into(), format!("F = {:.3}", vh.levene_mean.f_stat), format!("{}, {}", vh.levene_mean.df1, vh.levene_mean.df2), format!("{:.4}", vh.levene_mean.p_value)],
            vec!["Bartletts test".into(), format!("chi2 = {:.3}", vh.bartlett.chi2_stat), format!("{}", vh.bartlett.df), format!("{:.4}", vh.bartlett.p_value)],
            vec!["Welchs ANOVA (fallback)".into(), format!("F = {:.3}", vh.welch_anova.f_stat), format!("{:.1}, {:.1}", vh.welch_anova.df1, vh.welch_anova.df2), format!("{:.4}", vh.welch_anova.p_value)],
        ];
        if let Some(ref wt) = vh.welch_t_test {
            rows.push(vec!["Welchs t-test (fallback)".into(), format!("t = {:.3}", wt.t_stat), format!("{:.1}", wt.df), format!("{:.4}", wt.p_value)]);
        }
        if let Some(ref fb) = rep.group_fallbacks {
            if let Some(ref mw) = fb.mann_whitney {
                rows.push(vec!["Mann-Whitney U (ikke-parametrisk)".into(), format!("U = {:.1} (Z={:.2})", mw.u_stat, mw.z_stat), "-".into(), format!("{:.4}", mw.p_value)]);
            }
            if let Some(ref kw) = fb.kruskal_wallis {
                rows.push(vec!["Kruskal-Wallis H (ikke-parametrisk)".into(), format!("H = {:.3}", kw.h_stat), format!("{}", kw.df), format!("{:.4}", kw.p_value)]);
            }
        }
        pdf.table(&["Test", "Statistikk", "df", "p-verdi"], &[210.0, 120.0, 85.0, CONTENT_W - 415.0], &rows);
        if !rep.extra_group_checks.is_empty() {
            let xrows: Vec<Vec<String>> = rep
                .extra_group_checks
                .iter()
                .map(|(evh, efb)| {
                    let np = if let Some(ref mw) = efb.mann_whitney {
                        format!("MW p={:.4}", mw.p_value)
                    } else if let Some(ref kw) = efb.kruskal_wallis {
                        format!("KW p={:.4}", kw.p_value)
                    } else {
                        "-".into()
                    };
                    vec![
                        evh.value_col.clone(),
                        format!("{:.4}", evh.levene_median.p_value),
                        format!("{:.4}", evh.bartlett.p_value),
                        format!("{:.4}", evh.welch_anova.p_value),
                        np,
                        evh.status.to_uppercase(),
                    ]
                })
                .collect();
            pdf.table(
                &["Øvrige utfall per gruppe", "Levene p", "Bartlett p", "Welch p", "Ikke-param.", "Status"],
                &[132.0, 68.0, 68.0, 68.0, 102.0, CONTENT_W - 438.0],
                &xrows,
            );
        }
    }

    // 3. Sphericity
    if let Some(ref sp) = rep.sphericity {
        pdf.section_heading(&format!("3. Sfærisitet — Repeated Measures ANOVA ({})", sp.columns.join(", ")));
        pdf.paragraph(&sp.interpretation, Some("Resultat"));
        pdf.paragraph(&sp.recommendation, Some("Anbefalt tiltak"));
        let rows = vec![
            vec!["Mauchlys test".into(), format!("W = {:.4}", sp.mauchly_w), format!("p = {:.4}", sp.mauchly_p_value)],
            vec!["RM-ANOVA (ukorrigert)".into(), format!("F = {:.3}", sp.rm_anova_f), format!("p = {:.4}", sp.p_uncorrected)],
            vec!["Greenhouse-Geisser".into(), format!("eps = {:.3}", sp.epsilon_gg), format!("p = {:.4}", sp.p_greenhouse_geisser)],
            vec!["Huynh-Feldt".into(), format!("eps = {:.3}", sp.epsilon_hf), format!("p = {:.4}", sp.p_huynh_feldt)],
        ];
        pdf.table(&["Metode", "Estimat / Statistikk", "p-verdi"], &[200.0, 160.0, CONTENT_W - 360.0], &rows);
    }

    // 4. Linear Regression + Diagnostics + Vector Residual & ACF Charts
    if let Some(ref reg) = rep.linear_regression {
        pdf.section_heading(&format!("4. Lineær regresjon, forutsetninger og modellkvalitet (Y = {})", reg.target));
        pdf.paragraph(
            &format!(
                "R² = {:.4}, Justert R² = {:.4}, RMSE = {:.4}, MAE = {:.4}, F = {:.2} (p = {:.4}), AIC = {:.2}, BIC = {:.2}.",
                reg.r_squared, reg.adj_r_squared, reg.rmse, reg.mae, reg.f_stat, reg.f_p_value, reg.aic, reg.bic
            ),
            Some("Modelltilpasning"),
        );

        let coef_rows: Vec<Vec<String>> = reg
            .coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let h3 = &reg.heteroscedasticity.hc3_coefs[i];
                let nw = &reg.autocorrelation.newey_west_coefs[i];
                vec![
                    c.name.clone(),
                    format!("{:.4}", c.estimate),
                    format!("{:.4} ({:.4})", c.std_error, c.p_value),
                    format!("{:.4} ({:.4})", h3.std_error, h3.p_value),
                    format!("{:.4} ({:.4})", nw.std_error, nw.p_value),
                ]
            })
            .collect();
        pdf.table(
            &["Variabel", "Estimat", "OLS SE (p)", "HC3 SE (p)", "Newey-West SE (p)"],
            &[120.0, 72.0, 105.0, 105.0, CONTENT_W - 402.0],
            &coef_rows,
        );

        if let Some(ref mc) = reg.multicollinearity {
            pdf.paragraph(&format!("{} {}", mc.interpretation, mc.recommendation), Some("Multikollinearitet (VIF)"));
        }
        pdf.paragraph(&format!("{} {}", reg.autocorrelation.interpretation, reg.autocorrelation.recommendation), Some("Autokorrelasjon"));
        pdf.paragraph(&format!("{} {}", reg.heteroscedasticity.interpretation, reg.heteroscedasticity.recommendation), Some("Heteroskedastisitet"));
        pdf.paragraph(&format!("{} {}", reg.outliers.interpretation, reg.outliers.recommendation), Some("Uteliggere & innflytelse"));

        // Model comparison AIC/BIC
        let cmp_rows: Vec<Vec<String>> = reg
            .model_comparison
            .iter()
            .map(|m| {
                vec![
                    m.model_name.clone(),
                    format!("{}", m.k_params),
                    format!("{:.4}", m.r_squared),
                    format!("{:.4}", m.adj_r_squared),
                    format!("{:.3}", m.rmse),
                    format!("{:.2}", m.aic),
                    format!("{:.2}", m.bic),
                ]
            })
            .collect();
        pdf.table(
            &["Modell (AIC/BIC)", "k", "R²", "Just. R²", "RMSE", "AIC", "BIC"],
            &[175.0, 30.0, 56.0, 60.0, 62.0, 62.0, CONTENT_W - 445.0],
            &cmp_rows,
        );

        // Vector Charts side-by-side: Residuals vs Fitted + ACF
        pdf.ensure_space(165.0);
        let chart_h = 148.0;
        let chart_w = (CONTENT_W - 12.0) / 2.0;
        let y_top = pdf.y;

        // Left: Residuals vs Fitted
        let (px, py, pw, ph) = pdf.draw_mini_chart_box(
            MARGIN_L, y_top, chart_w, chart_h, "Residualer mot predikert verdi", "Predikert", "Residual",
        );
        if !reg.fitted.is_empty() {
            let min_x = reg.fitted.iter().copied().fold(f64::INFINITY, f64::min);
            let max_x = reg.fitted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let max_y = reg.residuals.iter().map(|v| v.abs()).fold(1e-6, f64::max) * 1.1;
            let sx = (max_x - min_x).max(1e-9);
            let y0 = py + ph / 2.0;
            pdf.draw_line(px, y0, px + pw, y0, 0.7, 0.86, 0.15, 0.15);
            for (&fx, &ry) in reg.fitted.iter().zip(reg.residuals.iter()) {
                let cx = px + ((fx - min_x) / sx) * pw;
                let cy = py + ((ry + max_y) / (2.0 * max_y)) * ph;
                pdf.draw_rect_fill(cx - 1.3, cy - 1.3, 2.6, 2.6, 0.114, 0.306, 0.847);
            }
        }

        // Right: ACF plot
        let rx = MARGIN_L + chart_w + 12.0;
        let (apx, apy, apw, aph) = pdf.draw_mini_chart_box(
            rx, y_top, chart_w, chart_h, "ACF-plott (Residual-autokorrelasjon)", "Lag", "ACF",
        );
        let ay0 = apy + aph / 2.0;
        let cb = reg.autocorrelation.acf_conf_bound.min(0.95);
        pdf.draw_line(apx, ay0, apx + apw, ay0, 0.6, 0.3, 0.35, 0.45);
        pdf.draw_line(apx, apy + ((cb + 1.0) / 2.0) * aph, apx + apw, apy + ((cb + 1.0) / 2.0) * aph, 0.6, 0.86, 0.15, 0.15);
        pdf.draw_line(apx, apy + ((-cb + 1.0) / 2.0) * aph, apx + apw, apy + ((-cb + 1.0) / 2.0) * aph, 0.6, 0.86, 0.15, 0.15);
        let n_acf = reg.autocorrelation.acf_lags.len().max(1);
        let slot = apw / n_acf as f64;
        for (idx, &(_lag, val)) in reg.autocorrelation.acf_lags.iter().enumerate() {
            let bx = apx + (idx as f64 + 0.25) * slot;
            let bw = (slot * 0.5).clamp(2.0, 9.0);
            let yv = apy + ((val.clamp(-1.0, 1.0) + 1.0) / 2.0) * aph;
            let (r, g, b) = if val.abs() > cb { (0.86, 0.15, 0.15) } else { (0.114, 0.306, 0.847) };
            let bot = ay0.min(yv);
            let bh = (ay0 - yv).abs().max(1.0);
            pdf.draw_rect_fill(bx, bot, bw, bh, r, g, b);
        }
        pdf.y -= chart_h + 14.0;
    }

    // 5. Missing Data
    pdf.section_heading("5. Manglende data og Littles MCAR-test");
    pdf.paragraph(&rep.missing_data.interpretation, Some("Resultat"));
    pdf.paragraph(&rep.missing_data.recommendation, Some("Anbefalt tiltak"));
    if !rep.missing_data.mice_summary.is_empty() {
        let rows: Vec<Vec<String>> = rep
            .missing_data
            .mice_summary
            .iter()
            .map(|x| {
                vec![
                    x.column.clone(),
                    format!("{:.4}", x.complete_case_mean),
                    format!("{:.4}", x.fiml_em_mean),
                    format!("{:.4}", x.mice_pooled_mean),
                    format!("{:.4}", x.mice_pooled_se),
                ]
            })
            .collect();
        pdf.table(
            &["Kolonne", "Komplett snitt", "FIML (EM) snitt", "MICE snitt", "MICE SE"],
            &[130.0, 95.0, 95.0, 95.0, CONTENT_W - 415.0],
            &rows,
        );
    }

    // 6. Factor Suitability
    if let Some(ref fac) = rep.factor_suitability {
        pdf.section_heading("6. Egnethet for faktoranalyse / PCA (KMO & Bartlett)");
        pdf.paragraph(&fac.interpretation, Some("Resultat"));
        pdf.paragraph(&fac.recommendation, Some("Anbefalt tiltak"));
    }

    // 7. Logistic Regression + Vector ROC Curve
    if let Some(ref lr) = rep.logistic_regression {
        pdf.section_heading(&format!("7. Logistisk regresjon — Hosmer-Lemeshow & ROC/AUC ({})", lr.target));
        pdf.paragraph(&lr.interpretation, Some("Resultat"));
        pdf.paragraph(&lr.recommendation, Some("Anbefalt tiltak"));
        pdf.paragraph(
            &format!(
                "AIC = {:.2}, BIC = {:.2}, Nøyaktighet = {:.1}%, Sensitivitet = {:.1}%, Spesifisitet = {:.1}%, F1 = {:.3}.",
                lr.aic,
                lr.bic,
                lr.confusion_matrix.accuracy * 100.0,
                lr.confusion_matrix.sensitivity * 100.0,
                lr.confusion_matrix.specificity * 100.0,
                lr.confusion_matrix.f1_score
            ),
            Some("Klassifiseringsmål"),
        );

        pdf.ensure_space(160.0);
        let ch = 145.0;
        let cw = (CONTENT_W - 12.0) / 2.0;
        let yt = pdf.y;
        let (px, py, pw, ph) = pdf.draw_mini_chart_box(
            MARGIN_L, yt, cw, ch, &format!("ROC-kurve (AUC = {:.3})", lr.auc), "FPR", "TPR",
        );
        pdf.draw_line(px, py, px + pw, py + ph, 0.6, 0.6, 0.65, 0.72);
        for w in lr.roc_points.windows(2) {
            pdf.draw_line(
                px + w[0].fpr.clamp(0.0, 1.0) * pw,
                py + w[0].tpr.clamp(0.0, 1.0) * ph,
                px + w[1].fpr.clamp(0.0, 1.0) * pw,
                py + w[1].tpr.clamp(0.0, 1.0) * ph,
                1.4, 0.114, 0.306, 0.847,
            );
        }

        let rx = MARGIN_L + cw + 12.0;
        let (cpx, cpy, cpw, cph) = pdf.draw_mini_chart_box(
            rx, yt, cw, ch, "Hosmer-Lemeshow kalibrering", "Forventet", "Observert",
        );
        pdf.draw_line(cpx, cpy, cpx + cpw, cpy + cph, 0.6, 0.08, 0.55, 0.35);
        for w in lr.hosmer_lemeshow.bins.windows(2) {
            pdf.draw_line(
                cpx + w[0].expected_rate.clamp(0.0, 1.0) * cpw,
                cpy + w[0].observed_rate.clamp(0.0, 1.0) * cph,
                cpx + w[1].expected_rate.clamp(0.0, 1.0) * cpw,
                cpy + w[1].observed_rate.clamp(0.0, 1.0) * cph,
                1.2, 0.114, 0.306, 0.847,
            );
        }
        pdf.y -= ch + 14.0;
    }

    // 8. Cox Survival
    if let Some(ref cox) = rep.cox_survival {
        pdf.section_heading(&format!("8. Overlevelsesanalyse — Cox PH & Schoenfeld ({} / {})", cox.time_col, cox.event_col));
        pdf.paragraph(&cox.interpretation, Some("Resultat"));
        pdf.paragraph(&cox.recommendation, Some("Anbefalt tiltak"));
        let rows: Vec<Vec<String>> = cox
            .coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let st = &cox.schoenfeld_tests[i];
                vec![
                    c.name.clone(),
                    format!("{:.3}", c.hazard_ratio),
                    format!("{:.4}", c.p_value),
                    format!("{:.3}", st.rho_time),
                    format!("{:.4}", st.p_value),
                    if st.ph_met { "OK".into() } else { "BRUTT".into() },
                ]
            })
            .collect();
        pdf.table(
            &["Kovariat", "Hazard Ratio", "Wald p", "Schoenfeld r", "Schoenfeld p", "PH"],
            &[130.0, 80.0, 75.0, 80.0, 80.0, CONTENT_W - 445.0],
            &rows,
        );
    }

    // 9. Clustering
    if let Some(ref cl) = rep.clustering {
        pdf.section_heading("9. Klyngeanalyse (Silhouette-skår & Elbow-metoden)");
        pdf.paragraph(&cl.interpretation, Some("Resultat"));
        pdf.paragraph(&cl.recommendation, Some("Anbefalt tiltak"));
        let rows: Vec<Vec<String>> = cl
            .k_evals
            .iter()
            .map(|e| {
                vec![
                    format!("k = {}", e.k),
                    format!("{:.2}", e.wcss),
                    if e.k >= 2 { format!("{:.4}", e.silhouette) } else { "-".into() },
                ]
            })
            .collect();
        pdf.table(&["Antall klynger", "WCSS (Elbow)", "Silhouette-skår"], &[140.0, 175.0, CONTENT_W - 315.0], &rows);
    }

    // =========================================================================
    // APPENDIKS: Alle manuelle utregninger fra Manuell testbenk
    // =========================================================================
    if !manual_entries.is_empty() {
        pdf.new_page();
        pdf.draw_rect_fill(MARGIN_L, pdf.y - 48.0, CONTENT_W, 50.0, 0.118, 0.161, 0.231);
        pdf.draw_text(
            MARGIN_L + 14.0,
            pdf.y - 20.0,
            "F2",
            16.0,
            1.0,
            1.0,
            1.0,
            "APPENDIKS — Manuelle utregninger",
        );
        pdf.draw_text(
            MARGIN_L + 14.0,
            pdf.y - 36.0,
            "F1",
            9.2,
            0.800,
            0.855,
            0.930,
            &format!(
                "Alle egendefinerte tester kjørt i Manuell testbenk ({} stk.) for {}",
                manual_entries.len(),
                rep.filename
            ),
        );
        pdf.y -= 64.0;

        for (idx, entry) in manual_entries.iter().enumerate() {
            pdf.section_heading(&format!(
                "Appendiks A.{}: {} (kl. {})",
                idx + 1,
                entry.title,
                entry.timestamp
            ));
            pdf.paragraph(&entry.parameters, Some("Valgte kolonner og parametere"));

            match &entry.result {
                ManualCalcResult::Normality { check, boot, .. } => {
                    pdf.paragraph(&check.interpretation, Some("Resultat"));
                    pdf.paragraph(&check.recommendation, Some("Anbefalt tiltak"));
                    let mut rows = vec![
                        vec!["Anbefalt test".into(), check.recommended_test.clone(), format!("{:.4}", check.primary_statistic), format!("{:.4}", check.primary_p_value), check.status.to_uppercase()],
                    ];
                    if let Some(ref sw) = check.shapiro_wilk {
                        rows.push(vec!["Shapiro-Wilk".into(), format!("n = {}", sw.n), format!("W = {:.4}", sw.w), format!("{:.4}", sw.p_value), if sw.p_value >= 0.05 { "OK".into() } else { "BRUTT".into() }]);
                    }
                    rows.push(vec!["Kolmogorov-Smirnov (Lilliefors)".into(), format!("n = {}", check.ks_test.n), format!("D = {:.4}", check.ks_test.d_stat), format!("{:.4}", check.ks_test.p_value_lilliefors), if check.ks_test.p_value_lilliefors >= 0.05 { "OK".into() } else { "BRUTT".into() }]);
                    rows.push(vec!["Skjevhet / Eksess-kurtose".into(), format!("Snitt {:.3}, SD {:.3}", check.mean, check.std_dev), format!("{:.3} / {:.3}", check.skewness, check.excess_kurtosis), "-".into(), check.status.to_uppercase()]);
                    if let Some(ref b) = boot {
                        rows.push(vec![format!("Bootstrap (B = {})", b.resamples), format!("Snitt {:.4}, SE {:.4}", b.observed_mean, b.bootstrap_se), format!("[{:.4}, {:.4}]", b.ci_low_95, b.ci_high_95), "95% KI".into(), "OK".into()]);
                    }
                    pdf.table(&["Test / Mål", "Detaljer", "Verdi / KI", "p-verdi", "Status"], &[138.0, 128.0, 118.0, 62.0, CONTENT_W - 446.0], &rows);
                }
                ManualCalcResult::GroupVariance { variance, mw, kw, .. } => {
                    pdf.paragraph(&variance.interpretation, Some("Resultat"));
                    pdf.paragraph(&variance.recommendation, Some("Anbefalt tiltak"));
                    let mut rows = vec![
                        vec!["Levene (median / Brown-Forsythe)".into(), format!("F = {:.3}", variance.levene_median.f_stat), format!("{}, {}", variance.levene_median.df1, variance.levene_median.df2), format!("{:.4}", variance.levene_median.p_value)],
                        vec!["Levene (gjennomsnitt)".into(), format!("F = {:.3}", variance.levene_mean.f_stat), format!("{}, {}", variance.levene_mean.df1, variance.levene_mean.df2), format!("{:.4}", variance.levene_mean.p_value)],
                        vec!["Bartletts test".into(), format!("chi2 = {:.3}", variance.bartlett.chi2_stat), format!("{}", variance.bartlett.df), format!("{:.4}", variance.bartlett.p_value)],
                        vec!["Welchs ANOVA".into(), format!("F = {:.3}", variance.welch_anova.f_stat), format!("{:.1}, {:.1}", variance.welch_anova.df1, variance.welch_anova.df2), format!("{:.4}", variance.welch_anova.p_value)],
                    ];
                    if let Some(ref wt) = variance.welch_t_test {
                        rows.push(vec!["Welchs t-test".into(), format!("t = {:.3}", wt.t_stat), format!("{:.1}", wt.df), format!("{:.4}", wt.p_value)]);
                    }
                    if let Some(ref m) = mw {
                        rows.push(vec!["Mann-Whitney U".into(), format!("U = {:.1} (Z = {:.2})", m.u_stat, m.z_stat), "-".into(), format!("{:.4}", m.p_value)]);
                    }
                    if let Some(ref k) = kw {
                        rows.push(vec!["Kruskal-Wallis H".into(), format!("H = {:.3}", k.h_stat), format!("{}", k.df), format!("{:.4}", k.p_value)]);
                    }
                    pdf.table(&["Test", "Statistikk", "df", "p-verdi"], &[210.0, 125.0, 82.0, CONTENT_W - 417.0], &rows);

                    let grp_rows: Vec<Vec<String>> = variance
                        .group_summaries
                        .iter()
                        .map(|g| vec![g.name.clone(), format!("{}", g.n), format!("{:.3}", g.mean), format!("{:.3}", g.median), format!("{:.3}", g.std_dev), format!("{:.3}", g.variance)])
                        .collect();
                    pdf.table(&["Gruppe", "n", "Snitt", "Median", "Std.avvik", "Varians"], &[130.0, 45.0, 82.0, 82.0, 82.0, CONTENT_W - 421.0], &grp_rows);
                }
                ManualCalcResult::Sphericity(sp) => {
                    pdf.paragraph(&sp.interpretation, Some("Resultat"));
                    pdf.paragraph(&sp.recommendation, Some("Anbefalt tiltak"));
                    let rows = vec![
                        vec!["Mauchlys sfærisitetstest".into(), format!("W = {:.4}, chi2({}) = {:.3}", sp.mauchly_w, sp.chi2_df, sp.chi2_stat), format!("{:.4}", sp.mauchly_p_value)],
                        vec!["RM-ANOVA (ukorrigert)".into(), format!("F({:.1}, {:.1}) = {:.3}", sp.df1_uncorrected, sp.df2_uncorrected, sp.rm_anova_f), format!("{:.4}", sp.p_uncorrected)],
                        vec!["Greenhouse-Geisser".into(), format!("eps = {:.3}", sp.epsilon_gg), format!("{:.4}", sp.p_greenhouse_geisser)],
                        vec!["Huynh-Feldt".into(), format!("eps = {:.3}", sp.epsilon_hf), format!("{:.4}", sp.p_huynh_feldt)],
                    ];
                    pdf.table(&["Metode", "Statistikk / Epsilon", "p-verdi"], &[185.0, 210.0, CONTENT_W - 395.0], &rows);
                }
                ManualCalcResult::Regression(reg) => {
                    pdf.paragraph(
                        &format!(
                            "n = {}, R² = {:.4}, Justert R² = {:.4}, RMSE = {:.4}, MAE = {:.4}, F = {:.2} (p = {:.4}), AIC = {:.2}, BIC = {:.2}.",
                            reg.n, reg.r_squared, reg.adj_r_squared, reg.rmse, reg.mae, reg.f_stat, reg.f_p_value, reg.aic, reg.bic
                        ),
                        Some("Modelltilpasning"),
                    );
                    let coef_rows: Vec<Vec<String>> = reg
                        .coefficients
                        .iter()
                        .enumerate()
                        .map(|(i, c)| {
                            let h3 = &reg.heteroscedasticity.hc3_coefs[i];
                            let nw = &reg.autocorrelation.newey_west_coefs[i];
                            vec![
                                c.name.clone(),
                                format!("{:.4}", c.estimate),
                                format!("{:.4} ({:.4})", c.std_error, c.p_value),
                                format!("{:.4} ({:.4})", h3.std_error, h3.p_value),
                                format!("{:.4} ({:.4})", nw.std_error, nw.p_value),
                            ]
                        })
                        .collect();
                    pdf.table(&["Variabel", "Estimat", "OLS SE (p)", "HC3 SE (p)", "Newey-West SE (p)"], &[120.0, 72.0, 105.0, 105.0, CONTENT_W - 402.0], &coef_rows);

                    if let Some(ref mc) = reg.multicollinearity {
                        let vif_rows: Vec<Vec<String>> = mc
                            .vifs
                            .iter()
                            .map(|v| vec![v.variable.clone(), format!("{:.2}", v.vif), format!("{:.4}", v.r_squared_aux), v.severity.to_uppercase()])
                            .collect();
                        pdf.table(&["Prediktor (VIF)", "VIF", "Hjelpe-R²", "Status"], &[180.0, 85.0, 110.0, CONTENT_W - 375.0], &vif_rows);
                    }
                    pdf.paragraph(&format!("{} {}", reg.autocorrelation.interpretation, reg.autocorrelation.recommendation), Some("Autokorrelasjon"));
                    pdf.paragraph(&format!("{} {}", reg.heteroscedasticity.interpretation, reg.heteroscedasticity.recommendation), Some("Heteroskedastisitet"));
                    pdf.paragraph(&format!("{} {}", reg.outliers.interpretation, reg.outliers.recommendation), Some("Uteliggere & innflytelse"));

                    let cmp_rows: Vec<Vec<String>> = reg
                        .model_comparison
                        .iter()
                        .map(|m| vec![m.model_name.clone(), format!("{}", m.k_params), format!("{:.4}", m.r_squared), format!("{:.4}", m.adj_r_squared), format!("{:.3}", m.rmse), format!("{:.2}", m.aic), format!("{:.2}", m.bic)])
                        .collect();
                    pdf.table(&["Modell (AIC/BIC)", "k", "R²", "Just. R²", "RMSE", "AIC", "BIC"], &[175.0, 30.0, 56.0, 60.0, 62.0, 62.0, CONTENT_W - 445.0], &cmp_rows);

                    // Embedded vector Residual & ACF charts
                    pdf.ensure_space(160.0);
                    let chart_h = 142.0;
                    let chart_w = (CONTENT_W - 12.0) / 2.0;
                    let y_top = pdf.y;
                    let (px, py, pw, ph) = pdf.draw_mini_chart_box(MARGIN_L, y_top, chart_w, chart_h, "Residualer mot predikert", "Predikert", "Residual");
                    if !reg.fitted.is_empty() {
                        let min_x = reg.fitted.iter().copied().fold(f64::INFINITY, f64::min);
                        let max_x = reg.fitted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let max_y = reg.residuals.iter().map(|v| v.abs()).fold(1e-6, f64::max) * 1.1;
                        let sx = (max_x - min_x).max(1e-9);
                        let y0 = py + ph / 2.0;
                        pdf.draw_line(px, y0, px + pw, y0, 0.7, 0.86, 0.15, 0.15);
                        for (&fx, &ry) in reg.fitted.iter().zip(reg.residuals.iter()) {
                            let cx = px + ((fx - min_x) / sx) * pw;
                            let cy = py + ((ry + max_y) / (2.0 * max_y)) * ph;
                            pdf.draw_rect_fill(cx - 1.3, cy - 1.3, 2.6, 2.6, 0.114, 0.306, 0.847);
                        }
                    }
                    let rx = MARGIN_L + chart_w + 12.0;
                    let (apx, apy, apw, aph) = pdf.draw_mini_chart_box(rx, y_top, chart_w, chart_h, "ACF-plott (Residualer)", "Lag", "ACF");
                    let ay0 = apy + aph / 2.0;
                    let cb = reg.autocorrelation.acf_conf_bound.min(0.95);
                    pdf.draw_line(apx, ay0, apx + apw, ay0, 0.6, 0.3, 0.35, 0.45);
                    pdf.draw_line(apx, apy + ((cb + 1.0) / 2.0) * aph, apx + apw, apy + ((cb + 1.0) / 2.0) * aph, 0.6, 0.86, 0.15, 0.15);
                    pdf.draw_line(apx, apy + ((-cb + 1.0) / 2.0) * aph, apx + apw, apy + ((-cb + 1.0) / 2.0) * aph, 0.6, 0.86, 0.15, 0.15);
                    let n_acf = reg.autocorrelation.acf_lags.len().max(1);
                    let slot = apw / n_acf as f64;
                    for (i_lag, &(_lag, val)) in reg.autocorrelation.acf_lags.iter().enumerate() {
                        let bx = apx + (i_lag as f64 + 0.25) * slot;
                        let bw = (slot * 0.5).clamp(2.0, 9.0);
                        let yv = apy + ((val.clamp(-1.0, 1.0) + 1.0) / 2.0) * aph;
                        let (r, g, b) = if val.abs() > cb { (0.86, 0.15, 0.15) } else { (0.114, 0.306, 0.847) };
                        pdf.draw_rect_fill(bx, ay0.min(yv), bw, (ay0 - yv).abs().max(1.0), r, g, b);
                    }
                    pdf.y -= chart_h + 14.0;
                }
                ManualCalcResult::Missing(m) => {
                    pdf.paragraph(&m.interpretation, Some("Resultat"));
                    pdf.paragraph(&m.recommendation, Some("Anbefalt tiltak"));
                    if !m.mice_summary.is_empty() {
                        let rows: Vec<Vec<String>> = m
                            .mice_summary
                            .iter()
                            .map(|x| vec![x.column.clone(), format!("{:.4}", x.complete_case_mean), format!("{:.4}", x.fiml_em_mean), format!("{:.4}", x.mice_pooled_mean), format!("{:.4}", x.mice_pooled_se)])
                            .collect();
                        pdf.table(&["Kolonne", "Komplett snitt", "FIML (EM) snitt", "MICE snitt", "MICE SE"], &[130.0, 95.0, 95.0, 95.0, CONTENT_W - 415.0], &rows);
                    }
                }
                ManualCalcResult::Factor(fac) => {
                    pdf.paragraph(&fac.interpretation, Some("Resultat"));
                    pdf.paragraph(&fac.recommendation, Some("Anbefalt tiltak"));
                    let rows: Vec<Vec<String>> = fac
                        .per_variable_msa
                        .iter()
                        .map(|v| vec![v.variable.clone(), format!("{:.3}", v.msa), format!("{:.3}", v.max_abs_corr)])
                        .collect();
                    pdf.table(&["Variabel", "Individuell MSA", "Maks |r| mot andre"], &[200.0, 150.0, CONTENT_W - 350.0], &rows);
                }
                ManualCalcResult::Logistic(lr) => {
                    pdf.paragraph(&lr.interpretation, Some("Resultat"));
                    pdf.paragraph(&lr.recommendation, Some("Anbefalt tiltak"));
                    let coef_rows: Vec<Vec<String>> = lr
                        .coefficients
                        .iter()
                        .map(|c| vec![c.name.clone(), format!("{:.4}", c.estimate), format!("{:.3}", c.odds_ratio), format!("{:.4}", c.std_error), format!("{:.2}", c.z_stat), format!("{:.4}", c.p_value)])
                        .collect();
                    pdf.table(&["Variabel", "Log-odds", "Odds Ratio", "SE", "z", "p-verdi"], &[140.0, 75.0, 78.0, 75.0, 62.0, CONTENT_W - 430.0], &coef_rows);

                    pdf.ensure_space(155.0);
                    let ch = 138.0;
                    let cw = (CONTENT_W - 12.0) / 2.0;
                    let yt = pdf.y;
                    let (px, py, pw, ph) = pdf.draw_mini_chart_box(MARGIN_L, yt, cw, ch, &format!("ROC-kurve (AUC = {:.3})", lr.auc), "FPR", "TPR");
                    pdf.draw_line(px, py, px + pw, py + ph, 0.6, 0.6, 0.65, 0.72);
                    for w in lr.roc_points.windows(2) {
                        pdf.draw_line(px + w[0].fpr.clamp(0.0, 1.0) * pw, py + w[0].tpr.clamp(0.0, 1.0) * ph, px + w[1].fpr.clamp(0.0, 1.0) * pw, py + w[1].tpr.clamp(0.0, 1.0) * ph, 1.4, 0.114, 0.306, 0.847);
                    }
                    let rx = MARGIN_L + cw + 12.0;
                    let (cpx, cpy, cpw, cph) = pdf.draw_mini_chart_box(rx, yt, cw, ch, "Hosmer-Lemeshow kalibrering", "Forventet", "Observert");
                    pdf.draw_line(cpx, cpy, cpx + cpw, cpy + cph, 0.6, 0.08, 0.55, 0.35);
                    for w in lr.hosmer_lemeshow.bins.windows(2) {
                        pdf.draw_line(cpx + w[0].expected_rate.clamp(0.0, 1.0) * cpw, cpy + w[0].observed_rate.clamp(0.0, 1.0) * cph, cpx + w[1].expected_rate.clamp(0.0, 1.0) * cpw, cpy + w[1].observed_rate.clamp(0.0, 1.0) * cph, 1.2, 0.114, 0.306, 0.847);
                    }
                    pdf.y -= ch + 14.0;
                }
                ManualCalcResult::Survival(cox) => {
                    pdf.paragraph(&cox.interpretation, Some("Resultat"));
                    pdf.paragraph(&cox.recommendation, Some("Anbefalt tiltak"));
                    let rows: Vec<Vec<String>> = cox
                        .coefficients
                        .iter()
                        .enumerate()
                        .map(|(i, c)| {
                            let st = &cox.schoenfeld_tests[i];
                            vec![c.name.clone(), format!("{:.3}", c.hazard_ratio), format!("{:.4}", c.p_value), format!("{:.3}", st.rho_time), format!("{:.4}", st.p_value), if st.ph_met { "OK".into() } else { "BRUTT".into() }]
                        })
                        .collect();
                    pdf.table(&["Kovariat", "Hazard Ratio", "Wald p", "Schoenfeld r", "Schoenfeld p", "PH"], &[130.0, 80.0, 75.0, 80.0, 80.0, CONTENT_W - 445.0], &rows);
                }
                ManualCalcResult::Clustering(cl) => {
                    pdf.paragraph(&cl.interpretation, Some("Resultat"));
                    pdf.paragraph(&cl.recommendation, Some("Anbefalt tiltak"));
                    let rows: Vec<Vec<String>> = cl
                        .k_evals
                        .iter()
                        .map(|e| vec![format!("k = {}", e.k), format!("{:.2}", e.wcss), if e.k >= 2 { format!("{:.4}", e.silhouette) } else { "-".into() }])
                        .collect();
                    pdf.table(&["Antall klynger", "WCSS (Elbow)", "Silhouette-skår"], &[140.0, 175.0, CONTENT_W - 315.0], &rows);
                }
            }
        }
    }

    pdf.finish()
}

/// Bundle PDF report, Markdown report, JSON data, and all individual SVG visualizations into a ZIP archive.
pub fn build_zip_bundle(rep: &AutoAnalysisReport) -> Result<Vec<u8>, String> {
    build_zip_bundle_with_appendix(rep, &[])
}

pub fn build_zip_bundle_with_appendix(
    rep: &AutoAnalysisReport,
    manual_entries: &[ManualHistoryEntry],
) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut zip = ZipWriter::new(&mut cursor);
        let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

        let pdf_bytes = render_pdf_report_with_appendix(rep, manual_entries);
        zip.start_file("rapport.pdf", opts).map_err(|e| e.to_string())?;
        zip.write_all(&pdf_bytes).map_err(|e| e.to_string())?;

        let md = render_markdown_report_with_appendix(rep, manual_entries);
        zip.start_file("rapport.md", opts).map_err(|e| e.to_string())?;
        zip.write_all(md.as_bytes()).map_err(|e| e.to_string())?;

        let json = serde_json::to_string_pretty(rep).map_err(|e| e.to_string())?;
        zip.start_file("resultater.json", opts).map_err(|e| e.to_string())?;
        zip.write_all(json.as_bytes()).map_err(|e| e.to_string())?;

        for ch in &rep.charts {
            let path = format!("visualiseringer/{}", ch.filename);
            zip.start_file(&path, opts).map_err(|e| e.to_string())?;
            zip.write_all(ch.svg.as_bytes()).map_err(|e| e.to_string())?;
        }

        zip.finish().map_err(|e| e.to_string())?;
    }
    Ok(cursor.into_inner())
}
