use crate::stats::clustering::{ClusterKEval, ClusterPoint2D};
use crate::stats::logistic::{HosmerLemeshowBin, RocPoint};
use crate::stats::missing::ColumnMissingInfo;
use crate::stats::utils::{mean, normal_inv_cdf, quantile, std_dev};
use serde::{Deserialize, Serialize};
use std::fmt::Write;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedChart {
    pub id: String,
    pub filename: String,
    pub title: String,
    pub category: String,
    pub description: String,
    pub svg: String,
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn svg_header(w: f64, h: f64, title: &str) -> String {
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background:#ffffff; border-radius:8px;">
<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff" rx="8"/>
<text x="{tx}" y="26" text-anchor="middle" font-size="14" font-weight="600" fill="#1e293b">{title}</text>
"##,
        w = w,
        h = h,
        tx = w / 2.0,
        title = esc(title)
    )
}

fn nice_ticks(min_v: f64, max_v: f64, count: usize) -> Vec<f64> {
    if !min_v.is_finite() || !max_v.is_finite() || (max_v - min_v).abs() < 1e-12 {
        return vec![min_v];
    }
    let step = (max_v - min_v) / (count.saturating_sub(1).max(1) as f64);
    (0..count).map(|i| min_v + i as f64 * step).collect()
}

fn fmt_num(v: f64) -> String {
    if v.abs() >= 1000.0 || (v.abs() > 0.0 && v.abs() < 0.01) {
        format!("{:.2e}", v)
    } else if v.abs() >= 100.0 {
        format!("{:.1}", v)
    } else {
        format!("{:.2}", v)
    }
}

/// Histogram with overlaid Normal density curve
pub fn svg_histogram(title: &str, var_name: &str, data: &[f64]) -> String {
    let w = 560.0;
    let h = 340.0;
    let ml = 58.0;
    let mr = 24.0;
    let mt = 44.0;
    let mb = 50.0;
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut out = svg_header(w, h, title);
    if data.len() < 2 {
        out.push_str("</svg>");
        return out;
    }

    let min_x = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_x = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (max_x - min_x).max(1e-9);
    let n = data.len();
    let bins = ((n as f64).sqrt().ceil() as usize).clamp(6, 22);
    let bin_w = span / bins as f64;

    let mut counts = vec![0usize; bins];
    for &v in data {
        let mut idx = ((v - min_x) / bin_w).floor() as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1;
    }

    // Convert to density so normal PDF overlays directly
    let densities: Vec<f64> = counts
        .iter()
        .map(|&c| c as f64 / (n as f64 * bin_w))
        .collect();
    let m = mean(data);
    let s = std_dev(data).max(1e-9);
    let norm_peak = 1.0 / (s * (2.0 * std::f64::consts::PI).sqrt());
    let max_y = densities
        .iter()
        .copied()
        .fold(norm_peak, f64::max)
        * 1.12;

    // Grid & axes
    for t in nice_ticks(0.0, max_y, 5) {
        let py = mt + ph - (t / max_y) * ph;
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="#e2e8f0" stroke-width="1"/>
<text x="{lx}" y="{ty:.1}" text-anchor="end" font-size="10" fill="#64748b">{lbl}</text>"##,
            x2 = ml + pw,
            lx = ml - 6.0,
            ty = py + 3.0,
            lbl = fmt_num(t)
        );
    }

    // Bars
    let bar_px = pw / bins as f64;
    for (i, &d) in densities.iter().enumerate() {
        let bx = ml + i as f64 * bar_px;
        let bh = (d / max_y) * ph;
        let by = mt + ph - bh;
        let _ = writeln!(
            out,
            r##"<rect x="{bx:.1}" y="{by:.1}" width="{bw:.1}" height="{bh:.1}" fill="#3b82f6" fill-opacity="0.72" stroke="#1d4ed8" stroke-width="0.8"/>"##,
            bw = (bar_px - 1.5).max(1.0)
        );
    }

    // Normal curve polyline
    let mut pts = String::new();
    for step in 0..=80 {
        let x_val = min_x + (step as f64 / 80.0) * span;
        let z = (x_val - m) / s;
        let pdf = (-0.5 * z * z).exp() / (s * (2.0 * std::f64::consts::PI).sqrt());
        let px = ml + ((x_val - min_x) / span) * pw;
        let py = mt + ph - (pdf / max_y) * ph;
        let _ = write!(pts, "{:.1},{:.1} ", px, py);
    }
    let _ = writeln!(
        out,
        r##"<polyline fill="none" stroke="#ef4444" stroke-width="2.2" points="{pts}"/>"##
    );

    // Axes lines & x ticks
    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{y0}" x2="{x2}" y2="{y0}" stroke="#334155" stroke-width="1.2"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{y0}" stroke="#334155" stroke-width="1.2"/>"##,
        y0 = mt + ph,
        x2 = ml + pw
    );
    for t in nice_ticks(min_x, max_x, 5) {
        let px = ml + ((t - min_x) / span) * pw;
        let _ = writeln!(
            out,
            r##"<text x="{px:.1}" y="{ty}" text-anchor="middle" font-size="10" fill="#475569">{lbl}</text>"##,
            ty = mt + ph + 16.0,
            lbl = fmt_num(t)
        );
    }
    let _ = writeln!(
        out,
        r##"<text x="{cx}" y="{y}" text-anchor="middle" font-size="11" fill="#334155">{lbl}</text>
<text x="{lx}" y="{mt}" text-anchor="end" font-size="10" fill="#ef4444">— Normalfordeling</text>"##,
        cx = ml + pw / 2.0,
        y = h - 10.0,
        lbl = esc(var_name),
        lx = ml + pw
    );

    out.push_str("</svg>");
    out
}

/// Normal Q-Q Plot
pub fn svg_qq_plot(title: &str, data: &[f64]) -> String {
    let w = 520.0;
    let h = 340.0;
    let ml = 60.0;
    let mr = 24.0;
    let mt = 44.0;
    let mb = 48.0;
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut out = svg_header(w, h, title);
    let n = data.len();
    if n < 3 {
        out.push_str("</svg>");
        return out;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let theo: Vec<f64> = (0..n)
        .map(|i| normal_inv_cdf(((i + 1) as f64 - 0.5) / n as f64))
        .collect();

    let min_x = theo[0].min(-2.5);
    let max_x = theo[n - 1].max(2.5);
    let m = mean(&sorted);
    let s = std_dev(&sorted).max(1e-9);
    let min_y = sorted[0].min(m + min_x * s);
    let max_y = sorted[n - 1].max(m + max_x * s);
    let sx = (max_x - min_x).max(1e-9);
    let sy = (max_y - min_y).max(1e-9);

    let map_x = |v: f64| ml + ((v - min_x) / sx) * pw;
    let map_y = |v: f64| mt + ph - ((v - min_y) / sy) * ph;

    // Grid
    for t in nice_ticks(min_y, max_y, 5) {
        let py = map_y(t);
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="#e2e8f0" stroke-width="1"/>
<text x="{lx}" y="{ty:.1}" text-anchor="end" font-size="10" fill="#64748b">{lbl}</text>"##,
            x2 = ml + pw,
            lx = ml - 6.0,
            ty = py + 3.0,
            lbl = fmt_num(t)
        );
    }

    // Reference line y = m + s*x
    let _ = writeln!(
        out,
        r##"<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}" stroke="#ef4444" stroke-width="1.8" stroke-dasharray="5,4"/>"##,
        x1 = map_x(min_x),
        y1 = map_y(m + s * min_x),
        x2 = map_x(max_x),
        y2 = map_y(m + s * max_x)
    );

    // Points (subsample if > 500 for snappy SVG)
    let step = (n / 500).max(1);
    for i in (0..n).step_by(step) {
        let _ = writeln!(
            out,
            r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="3" fill="#2563eb" fill-opacity="0.75"/>"##,
            cx = map_x(theo[i]),
            cy = map_y(sorted[i])
        );
    }

    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{y0}" x2="{x2}" y2="{y0}" stroke="#334155" stroke-width="1.2"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{y0}" stroke="#334155" stroke-width="1.2"/>"##,
        y0 = mt + ph,
        x2 = ml + pw
    );
    for t in nice_ticks(min_x, max_x, 5) {
        let px = map_x(t);
        let _ = writeln!(
            out,
            r##"<text x="{px:.1}" y="{ty}" text-anchor="middle" font-size="10" fill="#475569">{lbl}</text>"##,
            ty = mt + ph + 16.0,
            lbl = fmt_num(t)
        );
    }
    let _ = writeln!(
        out,
        r##"<text x="{cx}" y="{y}" text-anchor="middle" font-size="11" fill="#334155">Teoretiske normal-kvantiler (Z)</text>
<text x="16" y="{cy}" text-anchor="middle" font-size="11" fill="#334155" transform="rotate(-90 16 {cy})">Observerte kvantiler</text>"##,
        cx = ml + pw / 2.0,
        y = h - 10.0,
        cy = mt + ph / 2.0
    );

    out.push_str("</svg>");
    out
}

/// Boxplot for multiple groups or variables
pub fn svg_boxplot(title: &str, y_label: &str, groups: &[(String, Vec<f64>)]) -> String {
    let w = (groups.len() as f64 * 90.0 + 130.0).clamp(480.0, 760.0);
    let h = 350.0;
    let ml = 62.0;
    let mr = 24.0;
    let mt = 44.0;
    let mb = 56.0;
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut out = svg_header(w, h, title);
    let all_vals: Vec<f64> = groups.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    if all_vals.is_empty() {
        out.push_str("</svg>");
        return out;
    }

    let mut min_y = all_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let mut max_y = all_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pad = (max_y - min_y).max(1e-6) * 0.08;
    min_y -= pad;
    max_y += pad;
    let sy = max_y - min_y;
    let map_y = |v: f64| mt + ph - ((v - min_y) / sy) * ph;

    for t in nice_ticks(min_y, max_y, 5) {
        let py = map_y(t);
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="#e2e8f0" stroke-width="1"/>
<text x="{lx}" y="{ty:.1}" text-anchor="end" font-size="10" fill="#64748b">{lbl}</text>"##,
            x2 = ml + pw,
            lx = ml - 6.0,
            ty = py + 3.0,
            lbl = fmt_num(t)
        );
    }

    let palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"];
    let slot_w = pw / groups.len().max(1) as f64;
    let box_w = (slot_w * 0.45).clamp(18.0, 56.0);

    for (idx, (name, vals)) in groups.iter().enumerate() {
        if vals.is_empty() {
            continue;
        }
        let cx = ml + (idx as f64 + 0.5) * slot_w;
        let q1 = quantile(vals, 0.25);
        let med = quantile(vals, 0.5);
        let q3 = quantile(vals, 0.75);
        let iqr = q3 - q1;
        let lo_fence = q1 - 1.5 * iqr;
        let hi_fence = q3 + 1.5 * iqr;
        let whisk_lo = vals
            .iter()
            .copied()
            .filter(|&v| v >= lo_fence)
            .fold(q1, f64::min);
        let whisk_hi = vals
            .iter()
            .copied()
            .filter(|&v| v <= hi_fence)
            .fold(q3, f64::max);

        let col = palette[idx % palette.len()];
        let y_q1 = map_y(q1);
        let y_q3 = map_y(q3);
        let y_med = map_y(med);
        let y_lo = map_y(whisk_lo);
        let y_hi = map_y(whisk_hi);

        let _ = writeln!(
            out,
            r##"<line x1="{cx:.1}" y1="{y_hi:.1}" x2="{cx:.1}" y2="{y_lo:.1}" stroke="#334155" stroke-width="1.3"/>
<line x1="{wx1:.1}" y1="{y_hi:.1}" x2="{wx2:.1}" y2="{y_hi:.1}" stroke="#334155" stroke-width="1.3"/>
<line x1="{wx1:.1}" y1="{y_lo:.1}" x2="{wx2:.1}" y2="{y_lo:.1}" stroke="#334155" stroke-width="1.3"/>
<rect x="{bx:.1}" y="{y_q3:.1}" width="{box_w:.1}" height="{bh:.1}" fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="1.8" rx="2"/>
<line x1="{bx:.1}" y1="{y_med:.1}" x2="{bx2:.1}" y2="{y_med:.1}" stroke="#0f172a" stroke-width="2.2"/>"##,
            wx1 = cx - box_w * 0.3,
            wx2 = cx + box_w * 0.3,
            bx = cx - box_w / 2.0,
            bx2 = cx + box_w / 2.0,
            bh = (y_q1 - y_q3).abs().max(2.0)
        );

        // Outliers
        for &v in vals {
            if v < lo_fence || v > hi_fence {
                let _ = writeln!(
                    out,
                    r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="3.2" fill="#ef4444" fill-opacity="0.8"/>"##,
                    cy = map_y(v)
                );
            }
        }

        let short_name: String = name.chars().take(14).collect();
        let _ = writeln!(
            out,
            r##"<text x="{cx:.1}" y="{ty}" text-anchor="middle" font-size="10.5" fill="#334155">{lbl}</text>"##,
            ty = mt + ph + 18.0,
            lbl = esc(&short_name)
        );
    }

    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{y0}" x2="{x2}" y2="{y0}" stroke="#334155" stroke-width="1.2"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{y0}" stroke="#334155" stroke-width="1.2"/>
<text x="16" y="{cy}" text-anchor="middle" font-size="11" fill="#334155" transform="rotate(-90 16 {cy})">{lbl}</text>"##,
        y0 = mt + ph,
        x2 = ml + pw,
        cy = mt + ph / 2.0,
        lbl = esc(y_label)
    );

    out.push_str("</svg>");
    out
}

/// Correlation Matrix Heatmap (highlights |r| > 0.8)
pub fn svg_corr_heatmap(title: &str, labels: &[String], corr: &[Vec<f64>]) -> String {
    let p = labels.len();
    let cell = (420.0 / p.max(1) as f64).clamp(36.0, 62.0);
    let ml = 110.0;
    let mt = 56.0;
    let mr = 30.0;
    let mb = 90.0;
    let w = ml + p as f64 * cell + mr;
    let h = mt + p as f64 * cell + mb;

    let mut out = svg_header(w, h, title);
    for i in 0..p {
        let short_i: String = labels[i].chars().take(13).collect();
        let ry = mt + (i as f64 + 0.5) * cell;
        let _ = writeln!(
            out,
            r##"<text x="{lx}" y="{ry:.1}" text-anchor="end" dominant-baseline="middle" font-size="10.5" fill="#334155">{lbl}</text>"##,
            lx = ml - 8.0,
            lbl = esc(&short_i)
        );
        let cx = ml + (i as f64 + 0.5) * cell;
        let cy = mt + p as f64 * cell + 12.0;
        let _ = writeln!(
            out,
            r##"<text x="{cx:.1}" y="{cy:.1}" text-anchor="end" font-size="10.5" fill="#334155" transform="rotate(-35 {cx:.1} {cy:.1})">{lbl}</text>"##,
            lbl = esc(&short_i)
        );

        for j in 0..p {
            let r = corr[i][j].clamp(-1.0, 1.0);
            let (cr, cg, cb) = if r >= 0.0 {
                // white -> crimson/red-blue? Let's do blue for positive, red for negative
                let t = r;
                ((255.0 * (1.0 - 0.82 * t)) as u8, (255.0 * (1.0 - 0.55 * t)) as u8, 255u8)
            } else {
                let t = -r;
                (255u8, (255.0 * (1.0 - 0.72 * t)) as u8, (255.0 * (1.0 - 0.72 * t)) as u8)
            };
            let x = ml + j as f64 * cell;
            let y = mt + i as f64 * cell;
            let high = i != j && r.abs() > 0.8;
            let stroke = if high { "#dc2626" } else { "#ffffff" };
            let sw = if high { 2.2 } else { 1.0 };
            let txt_col = if r.abs() > 0.55 { "#ffffff" } else { "#0f172a" };
            let _ = writeln!(
                out,
                r##"<rect x="{x:.1}" y="{y:.1}" width="{cell:.1}" height="{cell:.1}" fill="rgb({cr},{cg},{cb})" stroke="{stroke}" stroke-width="{sw}"/>
<text x="{tx:.1}" y="{ty:.1}" text-anchor="middle" dominant-baseline="middle" font-size="10" font-weight="{fw}" fill="{txt_col}">{r:.2}</text>"##,
                tx = x + cell / 2.0,
                ty = y + cell / 2.0,
                fw = if high { "700" } else { "500" }
            );
        }
    }
    out.push_str("</svg>");
    out
}

/// ACF Plot (Autocorrelation Function)
pub fn svg_acf_plot(title: &str, acf: &[(usize, f64)], conf_bound: f64) -> String {
    let w = 560.0;
    let h = 320.0;
    let ml = 56.0;
    let mr = 24.0;
    let mt = 44.0;
    let mb = 48.0;
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut out = svg_header(w, h, title);
    let map_y = |v: f64| mt + ph - ((v + 1.0) / 2.0) * ph;

    for &t in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
        let py = map_y(t);
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="{col}" stroke-width="{sw}"/>
<text x="{lx}" y="{ty:.1}" text-anchor="end" font-size="10" fill="#64748b">{t:.1}</text>"##,
            x2 = ml + pw,
            col = if t == 0.0 { "#334155" } else { "#e2e8f0" },
            sw = if t == 0.0 { 1.2 } else { 1.0 },
            lx = ml - 6.0,
            ty = py + 3.0
        );
    }

    // Confidence bands
    let cb = conf_bound.min(0.95);
    for &b in &[cb, -cb] {
        let py = map_y(b);
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="#ef4444" stroke-width="1.3" stroke-dasharray="5,4"/>"##,
            x2 = ml + pw
        );
    }

    let n = acf.len().max(1);
    let slot = pw / n as f64;
    let bar_w = (slot * 0.45).clamp(6.0, 22.0);
    let y0 = map_y(0.0);

    for (idx, &(lag, val)) in acf.iter().enumerate() {
        let cx = ml + (idx as f64 + 0.5) * slot;
        let yv = map_y(val.clamp(-1.0, 1.0));
        let top = y0.min(yv);
        let bh = (y0 - yv).abs().max(1.0);
        let col = if val.abs() > conf_bound {
            "#ef4444"
        } else {
            "#2563eb"
        };
        let _ = writeln!(
            out,
            r##"<rect x="{bx:.1}" y="{top:.1}" width="{bar_w:.1}" height="{bh:.1}" fill="{col}" rx="1.5"/>
<text x="{cx:.1}" y="{ty}" text-anchor="middle" font-size="9.5" fill="#475569">{lag}</text>"##,
            bx = cx - bar_w / 2.0,
            ty = mt + ph + 16.0
        );
    }

    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<text x="{cx}" y="{y}" text-anchor="middle" font-size="11" fill="#334155">Lag (tidsforsinkelse)</text>
<text x="{lx}" y="{mt}" text-anchor="end" font-size="10" fill="#ef4444">--- ±95% konfidensbånd (±{conf_bound:.2})</text>"##,
        yb = mt + ph,
        cx = ml + pw / 2.0,
        y = h - 10.0,
        lx = ml + pw
    );

    out.push_str("</svg>");
    out
}

/// Residuals vs Fitted scatter plot
pub fn svg_residuals_vs_fitted(title: &str, fitted: &[f64], residuals: &[f64]) -> String {
    let w = 560.0;
    let h = 340.0;
    let ml = 62.0;
    let mr = 24.0;
    let mt = 44.0;
    let mb = 48.0;
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut out = svg_header(w, h, title);
    if fitted.is_empty() {
        out.push_str("</svg>");
        return out;
    }

    let min_x = fitted.iter().copied().fold(f64::INFINITY, f64::min);
    let max_x = fitted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_abs_y = residuals
        .iter()
        .map(|v| v.abs())
        .fold(1e-6, f64::max)
        * 1.15;
    let min_y = -max_abs_y;
    let max_y = max_abs_y;
    let sx = (max_x - min_x).max(1e-9);
    let sy = (max_y - min_y).max(1e-9);

    let map_x = |v: f64| ml + ((v - min_x) / sx) * pw;
    let map_y = |v: f64| mt + ph - ((v - min_y) / sy) * ph;

    for t in nice_ticks(min_y, max_y, 5) {
        let py = map_y(t);
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="#e2e8f0" stroke-width="1"/>
<text x="{lx}" y="{ty:.1}" text-anchor="end" font-size="10" fill="#64748b">{lbl}</text>"##,
            x2 = ml + pw,
            lx = ml - 6.0,
            ty = py + 3.0,
            lbl = fmt_num(t)
        );
    }

    // Zero reference line
    let y0 = map_y(0.0);
    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{y0:.1}" x2="{x2}" y2="{y0:.1}" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="5,4"/>"##,
        x2 = ml + pw
    );

    for (&fx, &ry) in fitted.iter().zip(residuals.iter()) {
        let _ = writeln!(
            out,
            r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="3.2" fill="#2563eb" fill-opacity="0.65"/>"##,
            cx = map_x(fx),
            cy = map_y(ry)
        );
    }

    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{yb}" x2="{x2}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>"##,
        yb = mt + ph,
        x2 = ml + pw
    );
    for t in nice_ticks(min_x, max_x, 5) {
        let _ = writeln!(
            out,
            r##"<text x="{px:.1}" y="{ty}" text-anchor="middle" font-size="10" fill="#475569">{lbl}</text>"##,
            px = map_x(t),
            ty = mt + ph + 16.0,
            lbl = fmt_num(t)
        );
    }
    let _ = writeln!(
        out,
        r##"<text x="{cx}" y="{y}" text-anchor="middle" font-size="11" fill="#334155">Predikerte verdier (Ŷ)</text>
<text x="16" y="{cy}" text-anchor="middle" font-size="11" fill="#334155" transform="rotate(-90 16 {cy})">Residualer (e)</text>"##,
        cx = ml + pw / 2.0,
        y = h - 10.0,
        cy = mt + ph / 2.0
    );

    out.push_str("</svg>");
    out
}

/// Residuals vs Leverage plot (sized/colored by Cook's Distance)
pub fn svg_leverage_cooks(
    title: &str,
    leverage: &[f64],
    std_resids: &[f64],
    cooks_d: &[f64],
    cooks_thresh: f64,
) -> String {
    let w = 560.0;
    let h = 340.0;
    let ml = 58.0;
    let mr = 24.0;
    let mt = 44.0;
    let mb = 48.0;
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut out = svg_header(w, h, title);
    if leverage.is_empty() {
        out.push_str("</svg>");
        return out;
    }

    let max_x = leverage.iter().copied().fold(0.1, f64::max) * 1.12;
    let max_abs_y = std_resids
        .iter()
        .map(|v| v.abs())
        .fold(2.5, f64::max)
        * 1.15;
    let map_x = |v: f64| ml + (v / max_x) * pw;
    let map_y = |v: f64| mt + ph - ((v + max_abs_y) / (2.0 * max_abs_y)) * ph;

    for t in nice_ticks(-max_abs_y, max_abs_y, 5) {
        let py = map_y(t);
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="#e2e8f0" stroke-width="1"/>
<text x="{lx}" y="{ty:.1}" text-anchor="end" font-size="10" fill="#64748b">{lbl}</text>"##,
            x2 = ml + pw,
            lx = ml - 6.0,
            ty = py + 3.0,
            lbl = fmt_num(t)
        );
    }

    let y0 = map_y(0.0);
    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{y0:.1}" x2="{x2}" y2="{y0:.1}" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,4"/>"##,
        x2 = ml + pw
    );

    for i in 0..leverage.len() {
        let cx = map_x(leverage[i]);
        let cy = map_y(std_resids[i]);
        let d = cooks_d[i];
        let flagged = d > cooks_thresh;
        let col = if d > 1.0 {
            "#dc2626"
        } else if flagged {
            "#f59e0b"
        } else {
            "#2563eb"
        };
        let r = (3.0 + (d / cooks_thresh.max(1e-6)).sqrt() * 2.2).clamp(3.0, 9.0);
        let _ = writeln!(
            out,
            r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="{r:.1}" fill="{col}" fill-opacity="0.72" stroke="#1e293b" stroke-width="0.5"/>"##
        );
        if flagged {
            let _ = writeln!(
                out,
                r##"<text x="{tx:.1}" y="{ty:.1}" font-size="9" font-weight="600" fill="#991b1b">#{idx}</text>"##,
                tx = cx + r + 2.0,
                ty = cy - 2.0,
                idx = i + 1
            );
        }
    }

    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{yb}" x2="{x2}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>"##,
        yb = mt + ph,
        x2 = ml + pw
    );
    for t in nice_ticks(0.0, max_x, 5) {
        let _ = writeln!(
            out,
            r##"<text x="{px:.1}" y="{ty}" text-anchor="middle" font-size="10" fill="#475569">{lbl}</text>"##,
            px = map_x(t),
            ty = mt + ph + 16.0,
            lbl = fmt_num(t)
        );
    }
    let _ = writeln!(
        out,
        r##"<text x="{cx}" y="{y}" text-anchor="middle" font-size="11" fill="#334155">Leverage (h_ii) — oransje/rød: Cook's D &gt; 4/n ({cooks_thresh:.3})</text>
<text x="16" y="{cy}" text-anchor="middle" font-size="11" fill="#334155" transform="rotate(-90 16 {cy})">Standardiserte residualer</text>"##,
        cx = ml + pw / 2.0,
        y = h - 10.0,
        cy = mt + ph / 2.0
    );

    out.push_str("</svg>");
    out
}

/// Missingness horizontal bar chart per column
pub fn svg_missingness(title: &str, cols: &[ColumnMissingInfo]) -> String {
    let w = 580.0;
    let bar_h = 22.0;
    let gap = 8.0;
    let mt = 44.0;
    let mb = 36.0;
    let ml = 140.0;
    let mr = 64.0;
    let h = (mt + mb + cols.len().max(1) as f64 * (bar_h + gap)).clamp(180.0, 620.0);
    let pw = w - ml - mr;

    let mut out = svg_header(w, h, title);
    let max_pct = cols
        .iter()
        .map(|c| c.missing_pct)
        .fold(10.0_f64, f64::max)
        .min(100.0);

    for (i, c) in cols.iter().enumerate() {
        let y = mt + i as f64 * (bar_h + gap);
        let bw = (c.missing_pct / max_pct) * pw;
        let fill = if c.missing_pct == 0.0 {
            "#10b981"
        } else if c.missing_pct < 5.0 {
            "#f59e0b"
        } else {
            "#ef4444"
        };
        let short_lbl: String = c.column.chars().take(18).collect();
        let _ = writeln!(
            out,
            r##"<text x="{lx}" y="{ty:.1}" text-anchor="end" dominant-baseline="middle" font-size="10.5" fill="#334155">{lbl}</text>
<rect x="{ml}" y="{y:.1}" width="{pw:.1}" height="{bar_h:.1}" fill="#f1f5f9" rx="3"/>
<rect x="{ml}" y="{y:.1}" width="{bw:.1}" height="{bar_h:.1}" fill="{fill}" rx="3"/>
<text x="{tx:.1}" y="{ty:.1}" dominant-baseline="middle" font-size="10" fill="#334155">{pct:.1}% ({cnt})</text>"##,
            lx = ml - 8.0,
            ty = y + bar_h / 2.0,
            lbl = esc(&short_lbl),
            bw = bw.max(if c.missing_count > 0 { 3.0 } else { 0.0 }),
            tx = ml + pw + 6.0,
            pct = c.missing_pct,
            cnt = c.missing_count
        );
    }

    out.push_str("</svg>");
    out
}

/// PCA Scree plot (Eigenvalues vs Component with Kaiser = 1.0 line)
pub fn svg_scree_plot(title: &str, eigenvalues: &[f64]) -> String {
    let w = 520.0;
    let h = 320.0;
    let ml = 56.0;
    let mr = 24.0;
    let mt = 44.0;
    let mb = 48.0;
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut out = svg_header(w, h, title);
    if eigenvalues.is_empty() {
        out.push_str("</svg>");
        return out;
    }

    let max_y = eigenvalues.iter().copied().fold(1.5, f64::max) * 1.15;
    let map_y = |v: f64| mt + ph - (v / max_y) * ph;
    let n = eigenvalues.len();
    let map_x = |i: usize| {
        if n <= 1 {
            ml + pw / 2.0
        } else {
            ml + (i as f64 / (n - 1) as f64) * pw
        }
    };

    for t in nice_ticks(0.0, max_y, 5) {
        let py = map_y(t);
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="#e2e8f0" stroke-width="1"/>
<text x="{lx}" y="{ty:.1}" text-anchor="end" font-size="10" fill="#64748b">{lbl}</text>"##,
            x2 = ml + pw,
            lx = ml - 6.0,
            ty = py + 3.0,
            lbl = fmt_num(t)
        );
    }

    // Kaiser criterion line (y = 1.0)
    let yk = map_y(1.0);
    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{yk:.1}" x2="{x2}" y2="{yk:.1}" stroke="#ef4444" stroke-width="1.4" stroke-dasharray="5,4"/>
<text x="{x2}" y="{yk_lbl:.1}" text-anchor="end" font-size="9.5" fill="#ef4444">Kaiser = 1.0</text>"##,
        x2 = ml + pw,
        yk_lbl = yk - 4.0
    );

    let mut pts = String::new();
    for (i, &ev) in eigenvalues.iter().enumerate() {
        let _ = write!(pts, "{:.1},{:.1} ", map_x(i), map_y(ev));
    }
    let _ = writeln!(
        out,
        r##"<polyline fill="none" stroke="#2563eb" stroke-width="2.2" points="{pts}"/>"##
    );

    for (i, &ev) in eigenvalues.iter().enumerate() {
        let cx = map_x(i);
        let cy = map_y(ev);
        let _ = writeln!(
            out,
            r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="4.5" fill="#2563eb"/>
<text x="{cx:.1}" y="{ty}" text-anchor="middle" font-size="10" fill="#475569">PC{idx}</text>"##,
            ty = mt + ph + 16.0,
            idx = i + 1
        );
    }

    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{yb}" x2="{x2}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<text x="16" y="{cy}" text-anchor="middle" font-size="11" fill="#334155" transform="rotate(-90 16 {cy})">Egenverdi</text>"##,
        yb = mt + ph,
        x2 = ml + pw,
        cy = mt + ph / 2.0
    );

    out.push_str("</svg>");
    out
}

/// Repeated-measures condition profile plot
pub fn svg_sphericity_profile(title: &str, labels: &[String], means: &[f64]) -> String {
    let w = 520.0;
    let h = 300.0;
    let ml = 60.0;
    let mr = 24.0;
    let mt = 44.0;
    let mb = 50.0;
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut out = svg_header(w, h, title);
    if means.is_empty() {
        out.push_str("</svg>");
        return out;
    }

    let min_v = means.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pad = (max_v - min_v).max(1.0) * 0.2;
    let lo = min_v - pad;
    let hi = max_v + pad;
    let map_y = |v: f64| mt + ph - ((v - lo) / (hi - lo)) * ph;
    let n = means.len();
    let map_x = |i: usize| {
        if n <= 1 {
            ml + pw / 2.0
        } else {
            ml + (i as f64 / (n - 1) as f64) * pw
        }
    };

    for t in nice_ticks(lo, hi, 5) {
        let py = map_y(t);
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="#e2e8f0" stroke-width="1"/>
<text x="{lx}" y="{ty:.1}" text-anchor="end" font-size="10" fill="#64748b">{lbl}</text>"##,
            x2 = ml + pw,
            lx = ml - 6.0,
            ty = py + 3.0,
            lbl = fmt_num(t)
        );
    }

    let mut pts = String::new();
    for (i, &m) in means.iter().enumerate() {
        let _ = write!(pts, "{:.1},{:.1} ", map_x(i), map_y(m));
    }
    let _ = writeln!(
        out,
        r##"<polyline fill="none" stroke="#8b5cf6" stroke-width="2.5" points="{pts}"/>"##
    );
    for (i, &m) in means.iter().enumerate() {
        let cx = map_x(i);
        let cy = map_y(m);
        let short_lbl: String = labels[i].chars().take(12).collect();
        let _ = writeln!(
            out,
            r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="5" fill="#7c3aed"/>
<text x="{cx:.1}" y="{vy:.1}" text-anchor="middle" font-size="9.5" fill="#334155">{m:.2}</text>
<text x="{cx:.1}" y="{ty}" text-anchor="middle" font-size="10" fill="#475569">{lbl}</text>"##,
            vy = cy - 9.0,
            ty = mt + ph + 18.0,
            lbl = esc(&short_lbl)
        );
    }

    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{yb}" x2="{x2}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>"##,
        yb = mt + ph,
        x2 = ml + pw
    );

    out.push_str("</svg>");
    out
}

/// ROC Curve with AUC shading
pub fn svg_roc_curve(title: &str, points: &[RocPoint], auc: f64) -> String {
    let w = 480.0;
    let h = 360.0;
    let ml = 58.0;
    let mr = 26.0;
    let mt = 44.0;
    let mb = 50.0;
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut out = svg_header(w, h, title);
    let map_x = |v: f64| ml + v.clamp(0.0, 1.0) * pw;
    let map_y = |v: f64| mt + ph - v.clamp(0.0, 1.0) * ph;

    for &t in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let py = map_y(t);
        let px = map_x(t);
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="#e2e8f0" stroke-width="1"/>
<line x1="{px:.1}" y1="{mt}" x2="{px:.1}" y2="{yb}" stroke="#e2e8f0" stroke-width="1"/>
<text x="{lx}" y="{ty:.1}" text-anchor="end" font-size="10" fill="#64748b">{t:.2}</text>
<text x="{px:.1}" y="{by}" text-anchor="middle" font-size="10" fill="#64748b">{t:.2}</text>"##,
            x2 = ml + pw,
            yb = mt + ph,
            lx = ml - 6.0,
            ty = py + 3.0,
            by = mt + ph + 16.0
        );
    }

    // Diagonal random baseline
    let _ = writeln!(
        out,
        r##"<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,4"/>"##,
        x1 = map_x(0.0),
        y1 = map_y(0.0),
        x2 = map_x(1.0),
        y2 = map_y(1.0)
    );

    // Polygon fill under ROC
    let mut poly = format!("{:.1},{:.1} ", map_x(0.0), map_y(0.0));
    let mut line = String::new();
    for pt in points {
        let px = map_x(pt.fpr);
        let py = map_y(pt.tpr);
        let _ = write!(poly, "{:.1},{:.1} ", px, py);
        let _ = write!(line, "{:.1},{:.1} ", px, py);
    }
    let _ = write!(poly, "{:.1},{:.1}", map_x(1.0), map_y(0.0));

    let _ = writeln!(
        out,
        r##"<polygon points="{poly}" fill="#3b82f6" fill-opacity="0.16"/>
<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{line}"/>
<rect x="{bx}" y="{by}" width="128" height="28" fill="#ffffff" fill-opacity="0.9" stroke="#cbd5e1" rx="4"/>
<text x="{btx}" y="{bty}" text-anchor="middle" font-size="12" font-weight="700" fill="#1e40af">AUC = {auc:.3}</text>
<line x1="{ml}" y1="{yb}" x2="{x2}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<text x="{cx}" y="{y}" text-anchor="middle" font-size="11" fill="#334155">1 - Spesifisitet (FPR)</text>
<text x="16" y="{cy}" text-anchor="middle" font-size="11" fill="#334155" transform="rotate(-90 16 {cy})">Sensitivitet (TPR)</text>"##,
        bx = ml + pw - 138.0,
        by = mt + ph - 38.0,
        btx = ml + pw - 74.0,
        bty = mt + ph - 20.0,
        yb = mt + ph,
        x2 = ml + pw,
        cx = ml + pw / 2.0,
        y = h - 10.0,
        cy = mt + ph / 2.0
    );

    out.push_str("</svg>");
    out
}

/// Hosmer-Lemeshow Calibration plot (Expected vs Observed rate per group)
pub fn svg_calibration_plot(title: &str, bins: &[HosmerLemeshowBin]) -> String {
    let w = 480.0;
    let h = 340.0;
    let ml = 58.0;
    let mr = 26.0;
    let mt = 44.0;
    let mb = 50.0;
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut out = svg_header(w, h, title);
    let map_x = |v: f64| ml + v.clamp(0.0, 1.0) * pw;
    let map_y = |v: f64| mt + ph - v.clamp(0.0, 1.0) * ph;

    for &t in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let py = map_y(t);
        let px = map_x(t);
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="#e2e8f0" stroke-width="1"/>
<text x="{lx}" y="{ty:.1}" text-anchor="end" font-size="10" fill="#64748b">{t:.2}</text>
<text x="{px:.1}" y="{by}" text-anchor="middle" font-size="10" fill="#64748b">{t:.2}</text>"##,
            x2 = ml + pw,
            lx = ml - 6.0,
            ty = py + 3.0,
            by = mt + ph + 16.0
        );
    }

    // Ideal calibration line y = x
    let _ = writeln!(
        out,
        r##"<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}" stroke="#10b981" stroke-width="1.6" stroke-dasharray="5,4"/>"##,
        x1 = map_x(0.0),
        y1 = map_y(0.0),
        x2 = map_x(1.0),
        y2 = map_y(1.0)
    );

    let mut pts = String::new();
    for b in bins {
        let _ = write!(pts, "{:.1},{:.1} ", map_x(b.expected_rate), map_y(b.observed_rate));
    }
    let _ = writeln!(
        out,
        r##"<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{pts}"/>"##
    );
    for b in bins {
        let _ = writeln!(
            out,
            r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="4.5" fill="#2563eb"/>"##,
            cx = map_x(b.expected_rate),
            cy = map_y(b.observed_rate)
        );
    }

    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{yb}" x2="{x2}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<text x="{cx}" y="{y}" text-anchor="middle" font-size="11" fill="#334155">Predikert sannsynlighet (forventet)</text>
<text x="16" y="{cy}" text-anchor="middle" font-size="11" fill="#334155" transform="rotate(-90 16 {cy})">Observert andel hendelser</text>"##,
        yb = mt + ph,
        x2 = ml + pw,
        cx = ml + pw / 2.0,
        y = h - 10.0,
        cy = mt + ph / 2.0
    );

    out.push_str("</svg>");
    out
}

/// Schoenfeld residuals vs time plot
pub fn svg_schoenfeld(title: &str, var_name: &str, points: &[(f64, f64)], beta: f64) -> String {
    let w = 540.0;
    let h = 320.0;
    let ml = 62.0;
    let mr = 24.0;
    let mt = 44.0;
    let mb = 48.0;
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut out = svg_header(w, h, title);
    if points.is_empty() {
        out.push_str("</svg>");
        return out;
    }

    let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let min_y = points.iter().map(|p| p.1).fold(beta, f64::min);
    let max_y = points.iter().map(|p| p.1).fold(beta, f64::max);
    let pad_y = (max_y - min_y).max(0.5) * 0.15;
    let lo_y = min_y - pad_y;
    let hi_y = max_y + pad_y;
    let sx = (max_x - min_x).max(1e-9);
    let sy = (hi_y - lo_y).max(1e-9);

    let map_x = |v: f64| ml + ((v - min_x) / sx) * pw;
    let map_y = |v: f64| mt + ph - ((v - lo_y) / sy) * ph;

    for t in nice_ticks(lo_y, hi_y, 5) {
        let py = map_y(t);
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="#e2e8f0" stroke-width="1"/>
<text x="{lx}" y="{ty:.1}" text-anchor="end" font-size="10" fill="#64748b">{lbl}</text>"##,
            x2 = ml + pw,
            lx = ml - 6.0,
            ty = py + 3.0,
            lbl = fmt_num(t)
        );
    }

    // Constant beta horizontal line (PH null hypothesis)
    let yb = map_y(beta);
    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{yb:.1}" x2="{x2}" y2="{yb:.1}" stroke="#10b981" stroke-width="1.6" stroke-dasharray="5,4"/>"##,
        x2 = ml + pw
    );

    // OLS trend line of Schoenfeld residuals vs time
    let mx = points.iter().map(|p| p.0).sum::<f64>() / points.len() as f64;
    let my = points.iter().map(|p| p.1).sum::<f64>() / points.len() as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for &(tx, ty) in points {
        num += (tx - mx) * (ty - my);
        den += (tx - mx).powi(2);
    }
    let slope = if den > 1e-12 { num / den } else { 0.0 };
    let intercept = my - slope * mx;
    let _ = writeln!(
        out,
        r##"<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}" stroke="#ef4444" stroke-width="2"/>"##,
        x1 = map_x(min_x),
        y1 = map_y(intercept + slope * min_x),
        x2 = map_x(max_x),
        y2 = map_y(intercept + slope * max_x)
    );

    for &(tx, ty) in points {
        let _ = writeln!(
            out,
            r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="3.4" fill="#2563eb" fill-opacity="0.7"/>"##,
            cx = map_x(tx),
            cy = map_y(ty)
        );
    }

    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{ybot}" x2="{x2}" y2="{ybot}" stroke="#334155" stroke-width="1.2"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{ybot}" stroke="#334155" stroke-width="1.2"/>"##,
        ybot = mt + ph,
        x2 = ml + pw
    );
    for t in nice_ticks(min_x, max_x, 5) {
        let _ = writeln!(
            out,
            r##"<text x="{px:.1}" y="{ty}" text-anchor="middle" font-size="10" fill="#475569">{lbl}</text>"##,
            px = map_x(t),
            ty = mt + ph + 16.0,
            lbl = fmt_num(t)
        );
    }
    let _ = writeln!(
        out,
        r##"<text x="{cx}" y="{y}" text-anchor="middle" font-size="11" fill="#334155">Tid (grønn: konstant β, rød: trend over tid)</text>
<text x="16" y="{cy}" text-anchor="middle" font-size="11" fill="#334155" transform="rotate(-90 16 {cy})">Beta(t) for {lbl}</text>"##,
        cx = ml + pw / 2.0,
        y = h - 10.0,
        cy = mt + ph / 2.0,
        lbl = esc(var_name)
    );

    out.push_str("</svg>");
    out
}

/// Side-by-side Elbow (WCSS) and Silhouette score vs k
pub fn svg_elbow_silhouette(
    title: &str,
    evals: &[ClusterKEval],
    elbow_k: usize,
    best_k: usize,
) -> String {
    let w = 680.0;
    let h = 320.0;
    let mut out = svg_header(w, h, title);
    if evals.len() < 2 {
        out.push_str("</svg>");
        return out;
    }

    // Left panel: Elbow (WCSS)
    {
        let ml = 60.0;
        let pw = 255.0;
        let mt = 52.0;
        let ph = 205.0;
        let max_w = evals.iter().map(|e| e.wcss).fold(1.0, f64::max) * 1.08;
        let min_k = evals.first().unwrap().k as f64;
        let max_k = evals.last().unwrap().k as f64;
        let map_x = |k: usize| ml + ((k as f64 - min_k) / (max_k - min_k).max(1.0)) * pw;
        let map_y = |v: f64| mt + ph - (v / max_w) * ph;

        let _ = writeln!(
            out,
            r##"<text x="{cx}" y="44" text-anchor="middle" font-size="11.5" font-weight="600" fill="#334155">Elbow-metoden (WCSS, anbefalt k={elbow_k})</text>"##,
            cx = ml + pw / 2.0
        );
        for t in nice_ticks(0.0, max_w, 4) {
            let py = map_y(t);
            let _ = writeln!(
                out,
                r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="#e2e8f0" stroke-width="1"/>
<text x="{lx}" y="{ty:.1}" text-anchor="end" font-size="9.5" fill="#64748b">{lbl}</text>"##,
                x2 = ml + pw,
                lx = ml - 5.0,
                ty = py + 3.0,
                lbl = fmt_num(t)
            );
        }
        let mut pts = String::new();
        for ev in evals {
            let _ = write!(pts, "{:.1},{:.1} ", map_x(ev.k), map_y(ev.wcss));
        }
        let _ = writeln!(
            out,
            r##"<polyline fill="none" stroke="#2563eb" stroke-width="2.2" points="{pts}"/>"##
        );
        for ev in evals {
            let cx = map_x(ev.k);
            let cy = map_y(ev.wcss);
            let fill = if ev.k == elbow_k { "#ef4444" } else { "#2563eb" };
            let r = if ev.k == elbow_k { 5.5 } else { 3.8 };
            let _ = writeln!(
                out,
                r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="{r}" fill="{fill}"/>
<text x="{cx:.1}" y="{ty}" text-anchor="middle" font-size="9.5" fill="#475569">{k}</text>"##,
                ty = mt + ph + 16.0,
                k = ev.k
            );
        }
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{yb}" x2="{x2}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<text x="{cx}" y="{y}" text-anchor="middle" font-size="10.5" fill="#334155">Antall klynger (k)</text>"##,
            yb = mt + ph,
            x2 = ml + pw,
            cx = ml + pw / 2.0,
            y = h - 12.0
        );
    }

    // Right panel: Silhouette
    {
        let ml = 395.0;
        let pw = 255.0;
        let mt = 52.0;
        let ph = 205.0;
        let sil_evals: Vec<&ClusterKEval> = evals.iter().filter(|e| e.k >= 2).collect();
        let min_k = sil_evals.first().map(|e| e.k).unwrap_or(2) as f64;
        let max_k = sil_evals.last().map(|e| e.k).unwrap_or(3) as f64;
        let map_x = |k: usize| ml + ((k as f64 - min_k) / (max_k - min_k).max(1.0)) * pw;
        let map_y = |v: f64| mt + ph - v.clamp(0.0, 1.0) * ph;

        let _ = writeln!(
            out,
            r##"<text x="{cx}" y="44" text-anchor="middle" font-size="11.5" font-weight="600" fill="#334155">Silhouette-skår (beste k={best_k})</text>"##,
            cx = ml + pw / 2.0
        );
        for &t in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let py = map_y(t);
            let _ = writeln!(
                out,
                r##"<line x1="{ml}" y1="{py:.1}" x2="{x2}" y2="{py:.1}" stroke="#e2e8f0" stroke-width="1"/>
<text x="{lx}" y="{ty:.1}" text-anchor="end" font-size="9.5" fill="#64748b">{t:.2}</text>"##,
                x2 = ml + pw,
                lx = ml - 5.0,
                ty = py + 3.0
            );
        }
        let mut pts = String::new();
        for ev in &sil_evals {
            let _ = write!(pts, "{:.1},{:.1} ", map_x(ev.k), map_y(ev.silhouette));
        }
        let _ = writeln!(
            out,
            r##"<polyline fill="none" stroke="#10b981" stroke-width="2.2" points="{pts}"/>"##
        );
        for ev in &sil_evals {
            let cx = map_x(ev.k);
            let cy = map_y(ev.silhouette);
            let fill = if ev.k == best_k { "#ef4444" } else { "#10b981" };
            let r = if ev.k == best_k { 5.5 } else { 3.8 };
            let _ = writeln!(
                out,
                r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="{r}" fill="{fill}"/>
<text x="{cx:.1}" y="{ty}" text-anchor="middle" font-size="9.5" fill="#475569">{k}</text>"##,
                ty = mt + ph + 16.0,
                k = ev.k
            );
        }
        let _ = writeln!(
            out,
            r##"<line x1="{ml}" y1="{yb}" x2="{x2}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<text x="{cx}" y="{y}" text-anchor="middle" font-size="10.5" fill="#334155">Antall klynger (k)</text>"##,
            yb = mt + ph,
            x2 = ml + pw,
            cx = ml + pw / 2.0,
            y = h - 12.0
        );
    }

    out.push_str("</svg>");
    out
}

/// 2D PCA Cluster scatter plot
pub fn svg_clusters_2d(title: &str, pts: &[ClusterPoint2D]) -> String {
    let w = 520.0;
    let h = 340.0;
    let ml = 56.0;
    let mr = 24.0;
    let mt = 44.0;
    let mb = 48.0;
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut out = svg_header(w, h, title);
    if pts.is_empty() {
        out.push_str("</svg>");
        return out;
    }

    let min_x = pts.iter().map(|p| p.pc1).fold(f64::INFINITY, f64::min);
    let max_x = pts.iter().map(|p| p.pc1).fold(f64::NEG_INFINITY, f64::max);
    let min_y = pts.iter().map(|p| p.pc2).fold(f64::INFINITY, f64::min);
    let max_y = pts.iter().map(|p| p.pc2).fold(f64::NEG_INFINITY, f64::max);
    let sx = (max_x - min_x).max(1e-9);
    let sy = (max_y - min_y).max(1e-9);
    let map_x = |v: f64| ml + ((v - min_x) / sx) * pw;
    let map_y = |v: f64| mt + ph - ((v - min_y) / sy) * ph;

    let palette = [
        "#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16",
    ];
    for p in pts {
        let col = palette[(p.cluster.saturating_sub(1)) % palette.len()];
        let _ = writeln!(
            out,
            r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="3.8" fill="{col}" fill-opacity="0.78"/>"##,
            cx = map_x(p.pc1),
            cy = map_y(p.pc2)
        );
    }

    let _ = writeln!(
        out,
        r##"<line x1="{ml}" y1="{yb}" x2="{x2}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{yb}" stroke="#334155" stroke-width="1.2"/>
<text x="{cx}" y="{y}" text-anchor="middle" font-size="11" fill="#334155">Hovedkomponent 1 (PC1)</text>
<text x="16" y="{cy}" text-anchor="middle" font-size="11" fill="#334155" transform="rotate(-90 16 {cy})">Hovedkomponent 2 (PC2)</text>"##,
        yb = mt + ph,
        x2 = ml + pw,
        cx = ml + pw / 2.0,
        y = h - 10.0,
        cy = mt + ph / 2.0
    );

    out.push_str("</svg>");
    out
}
