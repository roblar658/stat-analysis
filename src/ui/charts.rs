use crate::stats::clustering::{ClusterKEval, ClusterPoint2D};
use crate::stats::logistic::{HosmerLemeshowBin, RocPoint};
use crate::stats::missing::ColumnMissingInfo;
use crate::stats::utils::{mean, normal_inv_cdf, quantile, std_dev};
use egui::{Align2, Button, Color32, FontId, Frame, Margin, Pos2, Rect, RichText, Stroke, Ui, Vec2};
use egui_plot::{Bar, BarChart, HLine, Legend, Line, LineStyle, Plot, PlotPoints, Points};

// High-contrast light palette (WCAG AA/AAA compliant)
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(15, 23, 42); // #0F172A
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(51, 65, 85); // #334155
pub const SURFACE_CARD: Color32 = Color32::WHITE;
pub const SURFACE_SUBTLE: Color32 = Color32::from_rgb(248, 250, 252); // #F8FAFC
pub const BORDER_DEFAULT: Color32 = Color32::from_rgb(203, 213, 225); // #CBD5E1
pub const PRIMARY_BLUE: Color32 = Color32::from_rgb(29, 78, 216); // #1D4ED8

pub fn status_colors(status: &str) -> (Color32, Color32, Color32) {
    // (bg, fg, border)
    match status {
        "ok" => (
            Color32::from_rgb(220, 252, 231),
            Color32::from_rgb(20, 83, 45),
            Color32::from_rgb(134, 239, 172),
        ),
        "warning" => (
            Color32::from_rgb(254, 243, 199),
            Color32::from_rgb(120, 53, 15),
            Color32::from_rgb(252, 211, 77),
        ),
        "fail" => (
            Color32::from_rgb(254, 226, 226),
            Color32::from_rgb(127, 29, 29),
            Color32::from_rgb(252, 165, 165),
        ),
        _ => (
            Color32::from_rgb(226, 232, 240),
            Color32::from_rgb(51, 65, 85),
            Color32::from_rgb(203, 213, 225),
        ),
    }
}

pub fn status_dot_color(status: &str) -> Color32 {
    match status {
        "ok" => Color32::from_rgb(21, 128, 61),
        "warning" => Color32::from_rgb(180, 83, 9),
        "fail" => Color32::from_rgb(185, 28, 28),
        _ => Color32::from_rgb(100, 116, 139),
    }
}

pub fn status_label(status: &str) -> &'static str {
    crate::i18n::status_label(status)
}

pub fn status_badge(ui: &mut Ui, status: &str) {
    let (bg, fg, border) = status_colors(status);
    Frame::new()
        .fill(bg)
        .stroke(Stroke::new(1.0_f32, border))
        .corner_radius(10.0_f32)
        .inner_margin(Margin::symmetric(8, 2))
        .show(ui, |ui| {
            ui.label(RichText::new(status_label(status)).color(fg).strong().size(11.0));
        });
}

pub fn primary_button(ui: &mut Ui, text: &str) -> egui::Response {
    ui.add(
        Button::new(RichText::new(text).color(Color32::WHITE).strong())
            .fill(PRIMARY_BLUE)
            .stroke(Stroke::new(1.0_f32, Color32::from_rgb(30, 64, 175))),
    )
}

pub fn secondary_button(ui: &mut Ui, text: &str) -> egui::Response {
    ui.add(
        Button::new(RichText::new(text).color(TEXT_PRIMARY).strong())
            .fill(Color32::from_rgb(241, 245, 249))
            .stroke(Stroke::new(1.0_f32, Color32::from_rgb(148, 163, 184))),
    )
}

pub fn card_frame() -> Frame {
    Frame::new()
        .fill(SURFACE_CARD)
        .stroke(Stroke::new(1.0_f32, BORDER_DEFAULT))
        .corner_radius(10.0_f32)
        .inner_margin(Margin::same(16))
}

pub fn bold_text(text: &str, size: f32, color: Color32) -> RichText {
    RichText::new(text)
        .font(FontId::new(size, egui::FontFamily::Name("bold".into())))
        .color(color)
}

/// Clean horizontal-scrolling table wrapper that disables word-splitting inside cells
/// so column headers and values stay crisp on a single line.
pub fn show_table(ui: &mut Ui, id: &str, add_rows: impl FnOnce(&mut Ui)) {
    egui::ScrollArea::horizontal().id_salt(format!("{}_scroll", id)).show(ui, |ui| {
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        egui::Grid::new(id)
            .striped(true)
            .spacing([20.0, 7.0])
            .min_col_width(48.0)
            .show(ui, add_rows);
    });
}

pub fn table_header_row(ui: &mut Ui, headers: &[&str]) {
    for &h in headers {
        ui.label(bold_text(h, 12.5, TEXT_PRIMARY));
    }
    ui.end_row();
}

/// Render a full-width card constrained strictly to available width so nothing overflows.
pub fn show_card(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui)) {
    let max_w = ui.available_width().max(260.0);
    card_frame().show(ui, |ui| {
        ui.set_max_width((max_w - 34.0).max(220.0));
        ui.set_min_width((max_w - 34.0).max(220.0));
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
        add_contents(ui);
    });
}

/// Responsive two-column helper: side-by-side on wide screens, stacked vertically when narrow.
pub fn responsive_two_col(
    ui: &mut Ui,
    left: impl FnOnce(&mut Ui),
    right: impl FnOnce(&mut Ui),
) {
    let avail = ui.available_width();
    if avail >= 740.0 {
        let gap = 14.0_f32;
        let col_w = ((avail - gap) / 2.0).floor().max(300.0);
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(
                Vec2::new(col_w, 0.0),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    ui.set_max_width(col_w);
                    left(ui);
                },
            );
            ui.add_space(gap);
            ui.allocate_ui_with_layout(
                Vec2::new(col_w, 0.0),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    ui.set_max_width(col_w);
                    right(ui);
                },
            );
        });
    } else {
        left(ui);
        ui.add_space(12.0);
        right(ui);
    }
}

pub fn callout_box(ui: &mut Ui, status: &str, interpretation: &str, recommendation: &str) {
    let (_, _, border) = status_colors(status);
    let max_w = ui.available_width().max(200.0);
    Frame::new()
        .fill(SURFACE_SUBTLE)
        .stroke(Stroke::new(1.5_f32, border))
        .corner_radius(8.0_f32)
        .inner_margin(Margin::same(12))
        .show(ui, |ui| {
            ui.set_max_width((max_w - 26.0).max(180.0));
            ui.set_min_width((max_w - 26.0).max(180.0));
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
            ui.horizontal_wrapped(|ui| {
                status_badge(ui, status);
                ui.add_space(4.0);
                ui.label(RichText::new(interpretation).color(TEXT_PRIMARY).size(13.5));
            });
            if !recommendation.is_empty() {
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new(crate::i18n::tr("Anbefalt tiltak:", "Recommended action:"))
                            .color(TEXT_PRIMARY)
                            .strong()
                            .size(13.0),
                    );
                    ui.label(RichText::new(recommendation).color(TEXT_SECONDARY).size(13.0));
                });
            }
        });
}

pub fn kpi_card(ui: &mut Ui, title: &str, value: &str, sub: Option<&str>) {
    Frame::new()
        .fill(SURFACE_SUBTLE)
        .stroke(Stroke::new(1.0_f32, BORDER_DEFAULT))
        .corner_radius(8.0_f32)
        .inner_margin(Margin::symmetric(14, 10))
        .show(ui, |ui| {
            ui.set_min_width(132.0);
            ui.set_max_width(240.0);
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(title.to_uppercase())
                        .size(10.5)
                        .strong()
                        .color(TEXT_SECONDARY),
                );
                ui.add_space(2.0);
                ui.label(RichText::new(value).size(18.0).strong().color(TEXT_PRIMARY));
                if let Some(s) = sub {
                    ui.label(RichText::new(s).size(12.0).color(TEXT_SECONDARY));
                }
            });
        });
}

pub fn save_svg_button(ui: &mut Ui, default_filename: &str, svg_builder: impl FnOnce() -> String) {
    if ui
        .add(
            Button::new(RichText::new(crate::i18n::tr("Lagre SVG...", "Save SVG...")).size(12.0).color(TEXT_PRIMARY))
                .fill(Color32::from_rgb(241, 245, 249))
                .stroke(Stroke::new(1.0_f32, Color32::from_rgb(148, 163, 184))),
        )
        .clicked()
    {
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name(default_filename)
            .add_filter(crate::i18n::tr("SVG vektorgrafikk", "SVG vector graphics"), &["svg"])
            .save_file()
        {
            let svg = svg_builder();
            let _ = std::fs::write(path, svg);
        }
    }
}

/// Interactive Histogram with overlaid Normal density curve
pub fn plot_histogram(ui: &mut Ui, id: &str, data: &[f64], height: f32) {
    if data.len() < 2 {
        ui.label("For få observasjoner for histogram.");
        return;
    }
    let min_x = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_x = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (max_x - min_x).max(1e-9);
    let n = data.len();
    let bins = ((n as f64).sqrt().ceil() as usize).clamp(6, 24);
    let bin_w = span / bins as f64;

    let mut counts = vec![0usize; bins];
    for &v in data {
        let mut idx = ((v - min_x) / bin_w).floor() as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1;
    }

    let bars: Vec<Bar> = counts
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let x_center = min_x + (i as f64 + 0.5) * bin_w;
            let density = c as f64 / (n as f64 * bin_w);
            Bar::new(x_center, density)
                .width(bin_w * 0.92)
                .fill(Color32::from_rgba_unmultiplied(37, 99, 235, 185))
                .stroke(Stroke::new(1.0_f32, Color32::from_rgb(29, 78, 216)))
        })
        .collect();

    let m = mean(data);
    let s = std_dev(data).max(1e-9);
    let norm_pts: Vec<[f64; 2]> = (0..=90)
        .map(|i| {
            let x = min_x + (i as f64 / 90.0) * span;
            let z = (x - m) / s;
            let pdf = (-0.5 * z * z).exp() / (s * (2.0 * std::f64::consts::PI).sqrt());
            [x, pdf]
        })
        .collect();

    Plot::new(id)
        .height(height)
        .width(ui.available_width().max(220.0))
        .legend(Legend::default())
        .x_axis_label("Verdi")
        .y_axis_label("Tetthet")
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(BarChart::new(bars).name("Histogram"));
            plot_ui.line(
                Line::new(PlotPoints::new(norm_pts))
                    .name("Normalfordeling")
                    .color(Color32::from_rgb(220, 38, 38))
                    .width(2.2_f32),
            );
        });
}

/// Interactive Normal Q-Q Plot
pub fn plot_qq(ui: &mut Ui, id: &str, data: &[f64], height: f32) {
    let n = data.len();
    if n < 3 {
        ui.label("For få observasjoner for Q-Q-plott.");
        return;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pts: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let z = normal_inv_cdf(((i + 1) as f64 - 0.5) / n as f64);
            [z, sorted[i]]
        })
        .collect();
    let m = mean(&sorted);
    let s = std_dev(&sorted).max(1e-9);
    let z0 = pts.first().unwrap()[0].min(-2.5);
    let z1 = pts.last().unwrap()[0].max(2.5);
    let ref_line = vec![[z0, m + s * z0], [z1, m + s * z1]];

    Plot::new(id)
        .height(height)
        .width(ui.available_width().max(220.0))
        .legend(Legend::default())
        .x_axis_label("Teoretiske normal-kvantiler (Z)")
        .y_axis_label("Observerte kvantiler")
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(PlotPoints::new(ref_line))
                    .name("Normal-linje")
                    .color(Color32::from_rgb(220, 38, 38))
                    .style(LineStyle::dashed_dense())
                    .width(1.8_f32),
            );
            plot_ui.points(
                Points::new(PlotPoints::new(pts))
                    .name("Observasjoner")
                    .radius(3.2_f32)
                    .color(PRIMARY_BLUE),
            );
        });
}

/// Custom high-precision native Painter Boxplot per group (strictly clipped to available width)
pub fn plot_boxplot(ui: &mut Ui, groups: &[(String, Vec<f64>)], height: f32) {
    let all_vals: Vec<f64> = groups.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    if all_vals.is_empty() {
        ui.label("Ingen data for boksplott.");
        return;
    }
    let avail_w = ui.available_width().max(260.0);
    let (rect, _resp) = ui.allocate_exact_size(Vec2::new(avail_w, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 6.0_f32, SURFACE_SUBTLE);
    painter.rect_stroke(rect, 6.0_f32, Stroke::new(1.0_f32, BORDER_DEFAULT), egui::StrokeKind::Inside);

    let ml = 58.0_f32;
    let mr = 18.0_f32;
    let mt = 18.0_f32;
    let mb = 36.0_f32;
    let pw = (rect.width() - ml - mr).max(60.0);
    let ph = (rect.height() - mt - mb).max(60.0);

    let mut min_y = all_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let mut max_y = all_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pad = (max_y - min_y).max(1e-6) * 0.1;
    min_y -= pad;
    max_y += pad;
    let span_y = max_y - min_y;

    let map_y = |v: f64| rect.top() + mt + ph - (((v - min_y) / span_y) as f32) * ph;

    for i in 0..=4 {
        let v = min_y + (i as f64 / 4.0) * span_y;
        let py = map_y(v);
        painter.line_segment(
            [Pos2::new(rect.left() + ml, py), Pos2::new(rect.right() - mr, py)],
            Stroke::new(1.0_f32, Color32::from_rgb(226, 232, 240)),
        );
        painter.text(
            Pos2::new(rect.left() + ml - 6.0, py),
            Align2::RIGHT_CENTER,
            format!("{:.1}", v),
            FontId::proportional(11.0),
            TEXT_SECONDARY,
        );
    }

    let palette = [
        Color32::from_rgb(29, 78, 216),
        Color32::from_rgb(4, 120, 87),
        Color32::from_rgb(180, 83, 9),
        Color32::from_rgb(109, 40, 217),
        Color32::from_rgb(190, 24, 93),
        Color32::from_rgb(14, 116, 144),
    ];

    let slot_w = pw / groups.len().max(1) as f32;
    let box_w = (slot_w * 0.44).clamp(18.0, 68.0);
    let max_chars = ((slot_w / 7.2) as usize).clamp(6, 18);

    for (idx, (name, vals)) in groups.iter().enumerate() {
        if vals.is_empty() {
            continue;
        }
        let cx = rect.left() + ml + (idx as f32 + 0.5) * slot_w;
        let q1 = quantile(vals, 0.25);
        let med = quantile(vals, 0.5);
        let q3 = quantile(vals, 0.75);
        let iqr = q3 - q1;
        let lo_f = q1 - 1.5 * iqr;
        let hi_f = q3 + 1.5 * iqr;
        let w_lo = vals.iter().copied().filter(|&v| v >= lo_f).fold(q1, f64::min);
        let w_hi = vals.iter().copied().filter(|&v| v <= hi_f).fold(q3, f64::max);

        let col = palette[idx % palette.len()];
        let y_q1 = map_y(q1);
        let y_q3 = map_y(q3);
        let y_med = map_y(med);
        let y_lo = map_y(w_lo);
        let y_hi = map_y(w_hi);

        painter.line_segment([Pos2::new(cx, y_hi), Pos2::new(cx, y_lo)], Stroke::new(1.5_f32, TEXT_SECONDARY));
        painter.line_segment(
            [Pos2::new(cx - box_w * 0.3, y_hi), Pos2::new(cx + box_w * 0.3, y_hi)],
            Stroke::new(1.5_f32, TEXT_SECONDARY),
        );
        painter.line_segment(
            [Pos2::new(cx - box_w * 0.3, y_lo), Pos2::new(cx + box_w * 0.3, y_lo)],
            Stroke::new(1.5_f32, TEXT_SECONDARY),
        );

        let box_rect = Rect::from_min_max(
            Pos2::new(cx - box_w / 2.0, y_q3.min(y_q1)),
            Pos2::new(cx + box_w / 2.0, y_q1.max(y_q3).max(y_q3 + 2.0)),
        );
        painter.rect_filled(box_rect, 3.0_f32, col.linear_multiply(0.25));
        painter.rect_stroke(box_rect, 3.0_f32, Stroke::new(1.8_f32, col), egui::StrokeKind::Inside);

        painter.line_segment(
            [Pos2::new(cx - box_w / 2.0, y_med), Pos2::new(cx + box_w / 2.0, y_med)],
            Stroke::new(2.5_f32, TEXT_PRIMARY),
        );

        for &v in vals {
            if v < lo_f || v > hi_f {
                painter.circle_filled(Pos2::new(cx, map_y(v)), 3.6_f32, Color32::from_rgb(220, 38, 38));
            }
        }

        let short: String = if name.chars().count() > max_chars {
            format!("{}…", name.chars().take(max_chars - 1).collect::<String>())
        } else {
            name.clone()
        };
        painter.text(
            Pos2::new(cx, rect.bottom() - 15.0),
            Align2::CENTER_CENTER,
            short,
            FontId::proportional(11.5),
            TEXT_PRIMARY,
        );
    }
}

/// Custom native Painter Correlation Matrix Heatmap wrapped in horizontal ScrollArea so it NEVER overflows
pub fn plot_corr_heatmap(ui: &mut Ui, id: &str, labels: &[String], corr: &[Vec<f64>]) {
    let p = labels.len();
    if p == 0 {
        return;
    }
    egui::ScrollArea::horizontal().id_salt(id).show(ui, |ui| {
        let avail = ui.available_width();
        let cell = ((avail - 135.0) / p as f32).clamp(44.0, 62.0);
        let ml = 120.0_f32;
        let mt = 28.0_f32;
        let w = ml + p as f32 * cell + 16.0;
        let h = mt + p as f32 * cell + 34.0;

        let (rect, _resp) = ui.allocate_exact_size(Vec2::new(w, h), egui::Sense::hover());
        let painter = ui.painter_at(rect);

        for i in 0..p {
            let short: String = if labels[i].chars().count() > 14 {
                format!("{}…", labels[i].chars().take(13).collect::<String>())
            } else {
                labels[i].clone()
            };
            painter.text(
                Pos2::new(rect.left() + ml - 8.0, rect.top() + mt + (i as f32 + 0.5) * cell),
                Align2::RIGHT_CENTER,
                &short,
                FontId::proportional(11.5),
                TEXT_PRIMARY,
            );
            let short_top: String = if labels[i].chars().count() > 7 {
                format!("{}…", labels[i].chars().take(6).collect::<String>())
            } else {
                labels[i].clone()
            };
            painter.text(
                Pos2::new(rect.left() + ml + (i as f32 + 0.5) * cell, rect.top() + mt - 10.0),
                Align2::CENTER_CENTER,
                &short_top,
                FontId::proportional(11.0),
                TEXT_PRIMARY,
            );

            for j in 0..p {
                let r = corr[i][j].clamp(-1.0, 1.0);
                let fill = if r >= 0.0 {
                    let t = r as f32;
                    Color32::from_rgb(
                        (255.0 * (1.0 - 0.80 * t)) as u8,
                        (255.0 * (1.0 - 0.55 * t)) as u8,
                        255,
                    )
                } else {
                    let t = (-r) as f32;
                    Color32::from_rgb(
                        255,
                        (255.0 * (1.0 - 0.72 * t)) as u8,
                        (255.0 * (1.0 - 0.72 * t)) as u8,
                    )
                };
                let cell_rect = Rect::from_min_size(
                    Pos2::new(rect.left() + ml + j as f32 * cell, rect.top() + mt + i as f32 * cell),
                    Vec2::splat(cell),
                );
                painter.rect_filled(cell_rect, 2.0_f32, fill);
                let high = i != j && r.abs() > 0.8;
                painter.rect_stroke(
                    cell_rect,
                    2.0_f32,
                    Stroke::new(
                        if high { 2.4_f32 } else { 1.0_f32 },
                        if high {
                            Color32::from_rgb(185, 28, 28)
                        } else {
                            Color32::WHITE
                        },
                    ),
                    egui::StrokeKind::Inside,
                );
                let fg = if r.abs() > 0.50 {
                    Color32::WHITE
                } else {
                    TEXT_PRIMARY
                };
                painter.text(
                    cell_rect.center(),
                    Align2::CENTER_CENTER,
                    format!("{:.2}", r),
                    FontId::proportional(11.5),
                    fg,
                );
            }
        }
        painter.text(
            Pos2::new(rect.left() + ml, rect.bottom() - 12.0),
            Align2::LEFT_CENTER,
            "Rød ramme markerer sterk kollinearitet (|r| > 0.80)",
            FontId::proportional(11.5),
            Color32::from_rgb(185, 28, 28),
        );
    });
}

/// Interactive ACF bar plot
pub fn plot_acf(ui: &mut Ui, id: &str, acf: &[(usize, f64)], conf_bound: f64, height: f32) {
    let bars: Vec<Bar> = acf
        .iter()
        .map(|&(lag, val)| {
            let col = if val.abs() > conf_bound {
                Color32::from_rgb(220, 38, 38)
            } else {
                PRIMARY_BLUE
            };
            Bar::new(lag as f64, val).width(0.45).fill(col)
        })
        .collect();

    Plot::new(id)
        .height(height)
        .width(ui.available_width().max(220.0))
        .legend(Legend::default())
        .x_axis_label("Lag (tidsforsinkelse)")
        .y_axis_label("Autokorrelasjon (ACF)")
        .include_y(-1.0)
        .include_y(1.0)
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(BarChart::new(bars).name("ACF"));
            plot_ui.hline(
                HLine::new(conf_bound)
                    .name("+95% grense")
                    .color(Color32::from_rgb(220, 38, 38))
                    .style(LineStyle::dashed_dense()),
            );
            plot_ui.hline(
                HLine::new(-conf_bound)
                    .name("-95% grense")
                    .color(Color32::from_rgb(220, 38, 38))
                    .style(LineStyle::dashed_dense()),
            );
            plot_ui.hline(HLine::new(0.0).color(TEXT_SECONDARY));
        });
}

/// Interactive Residuals vs Fitted scatter
pub fn plot_residuals_vs_fitted(ui: &mut Ui, id: &str, fitted: &[f64], residuals: &[f64], height: f32) {
    let pts: Vec<[f64; 2]> = fitted
        .iter()
        .copied()
        .zip(residuals.iter().copied())
        .map(|(f, r)| [f, r])
        .collect();

    Plot::new(id)
        .height(height)
        .width(ui.available_width().max(220.0))
        .legend(Legend::default())
        .x_axis_label("Predikerte verdier (Ŷ)")
        .y_axis_label("Residualer (e)")
        .show(ui, |plot_ui| {
            plot_ui.hline(
                HLine::new(0.0)
                    .name("Null-linje")
                    .color(Color32::from_rgb(220, 38, 38))
                    .style(LineStyle::dashed_dense()),
            );
            plot_ui.points(
                Points::new(PlotPoints::new(pts))
                    .name("Residualer")
                    .radius(3.4_f32)
                    .color(PRIMARY_BLUE),
            );
        });
}

/// Interactive Leverage vs Standardized Residuals colored by Cook's D
pub fn plot_leverage_cooks(
    ui: &mut Ui,
    id: &str,
    leverage: &[f64],
    std_resids: &[f64],
    cooks_d: &[f64],
    thresh: f64,
    height: f32,
) {
    let mut normal_pts = Vec::new();
    let mut warn_pts = Vec::new();
    let mut high_pts = Vec::new();
    for i in 0..leverage.len() {
        let p = [leverage[i], std_resids[i]];
        if cooks_d[i] > 1.0 {
            high_pts.push(p);
        } else if cooks_d[i] > thresh {
            warn_pts.push(p);
        } else {
            normal_pts.push(p);
        }
    }

    Plot::new(id)
        .height(height)
        .width(ui.available_width().max(220.0))
        .legend(Legend::default())
        .x_axis_label("Leverage (h_ii)")
        .y_axis_label("Standardiserte residualer")
        .show(ui, |plot_ui| {
            plot_ui.hline(HLine::new(0.0).color(TEXT_SECONDARY).style(LineStyle::dashed_dense()));
            plot_ui.points(
                Points::new(PlotPoints::new(normal_pts))
                    .name("Normal (Cook's D <= 4/n)")
                    .radius(3.2_f32)
                    .color(PRIMARY_BLUE),
            );
            if !warn_pts.is_empty() {
                plot_ui.points(
                    Points::new(PlotPoints::new(warn_pts))
                        .name("Innflytelsesrik (D > 4/n)")
                        .radius(5.2_f32)
                        .color(Color32::from_rgb(217, 119, 6)),
                );
            }
            if !high_pts.is_empty() {
                plot_ui.points(
                    Points::new(PlotPoints::new(high_pts))
                        .name("Kritisk uteligger (D > 1)")
                        .radius(6.5_f32)
                        .color(Color32::from_rgb(220, 38, 38)),
                );
            }
        });
}

/// Interactive Missingness progress bar table/chart (bounded cleanly to available width, no word splitting)
pub fn plot_missingness_bars(ui: &mut Ui, id: &str, cols: &[ColumnMissingInfo]) {
    let bar_w = (ui.available_width() - 260.0).clamp(120.0, 340.0);
    show_table(ui, id, |ui| {
        for c in cols {
            ui.label(bold_text(&c.column, 13.0, TEXT_PRIMARY));
            let frac = (c.missing_pct / 100.0).clamp(0.0, 1.0) as f32;
            let col = if c.missing_count == 0 {
                Color32::from_rgb(22, 163, 74)
            } else if c.missing_pct < 5.0 {
                Color32::from_rgb(217, 119, 6)
            } else {
                Color32::from_rgb(220, 38, 38)
            };
            ui.add(
                egui::ProgressBar::new(frac)
                    .desired_width(bar_w)
                    .fill(col)
                    .text(bold_text(&format!("{:.1}%", c.missing_pct), 12.0, TEXT_PRIMARY)),
            );
            ui.label(RichText::new(format!("{} mangler", c.missing_count)).color(TEXT_SECONDARY));
            ui.end_row();
        }
    });
}

/// Interactive PCA Scree Plot
pub fn plot_scree(ui: &mut Ui, id: &str, eigenvalues: &[f64], height: f32) {
    let pts: Vec<[f64; 2]> = eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &e)| [(i + 1) as f64, e])
        .collect();
    Plot::new(id)
        .height(height)
        .width(ui.available_width().max(220.0))
        .legend(Legend::default())
        .x_axis_label("Komponent #")
        .y_axis_label("Egenverdi")
        .include_y(0.0)
        .show(ui, |plot_ui| {
            plot_ui.hline(
                HLine::new(1.0)
                    .name("Kaiser-kriterium (= 1.0)")
                    .color(Color32::from_rgb(220, 38, 38))
                    .style(LineStyle::dashed_dense()),
            );
            plot_ui.line(
                Line::new(PlotPoints::new(pts.clone()))
                    .name("Egenverdi")
                    .color(PRIMARY_BLUE)
                    .width(2.2_f32),
            );
            plot_ui.points(
                Points::new(PlotPoints::new(pts))
                    .radius(4.5_f32)
                    .color(PRIMARY_BLUE),
            );
        });
}

/// Repeated-measures profile plot
pub fn plot_sphericity_profile(ui: &mut Ui, id: &str, means: &[f64], height: f32) {
    let pts: Vec<[f64; 2]> = means
        .iter()
        .enumerate()
        .map(|(i, &m)| [(i + 1) as f64, m])
        .collect();
    Plot::new(id)
        .height(height)
        .width(ui.available_width().max(220.0))
        .legend(Legend::default())
        .x_axis_label("Måling #")
        .y_axis_label("Gjennomsnitt")
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(PlotPoints::new(pts.clone()))
                    .name("Gjennomsnitt over tid/betingelse")
                    .color(Color32::from_rgb(109, 40, 217))
                    .width(2.4_f32),
            );
            plot_ui.points(
                Points::new(PlotPoints::new(pts))
                    .radius(5.0_f32)
                    .color(Color32::from_rgb(109, 40, 217)),
            );
        });
}

/// Interactive ROC Curve
pub fn plot_roc(ui: &mut Ui, id: &str, points: &[RocPoint], auc: f64, height: f32) {
    let curve: Vec<[f64; 2]> = points.iter().map(|p| [p.fpr, p.tpr]).collect();
    let diag = vec![[0.0, 0.0], [1.0, 1.0]];
    Plot::new(id)
        .height(height)
        .width(ui.available_width().max(220.0))
        .legend(Legend::default())
        .x_axis_label("1 - Spesifisitet (FPR)")
        .y_axis_label("Sensitivitet (TPR)")
        .include_x(0.0)
        .include_x(1.0)
        .include_y(0.0)
        .include_y(1.0)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(PlotPoints::new(diag))
                    .name("Tilfeldig (AUC = 0.50)")
                    .color(TEXT_SECONDARY)
                    .style(LineStyle::dashed_dense()),
            );
            plot_ui.line(
                Line::new(PlotPoints::new(curve))
                    .name(format!("ROC-kurve (AUC = {:.3})", auc))
                    .color(PRIMARY_BLUE)
                    .width(2.6_f32),
            );
        });
}

/// Interactive Hosmer-Lemeshow Calibration plot
pub fn plot_calibration(ui: &mut Ui, id: &str, bins: &[HosmerLemeshowBin], height: f32) {
    let pts: Vec<[f64; 2]> = bins
        .iter()
        .map(|b| [b.expected_rate, b.observed_rate])
        .collect();
    let diag = vec![[0.0, 0.0], [1.0, 1.0]];
    Plot::new(id)
        .height(height)
        .width(ui.available_width().max(220.0))
        .legend(Legend::default())
        .x_axis_label("Predikert sannsynlighet")
        .y_axis_label("Observert andel hendelser")
        .include_x(0.0)
        .include_x(1.0)
        .include_y(0.0)
        .include_y(1.0)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(PlotPoints::new(diag))
                    .name("Ideell kalibrering")
                    .color(Color32::from_rgb(4, 120, 87))
                    .style(LineStyle::dashed_dense()),
            );
            plot_ui.line(
                Line::new(PlotPoints::new(pts.clone()))
                    .name("Hosmer-Lemeshow grupper")
                    .color(PRIMARY_BLUE)
                    .width(2.0_f32),
            );
            plot_ui.points(
                Points::new(PlotPoints::new(pts))
                    .radius(4.8_f32)
                    .color(PRIMARY_BLUE),
            );
        });
}

/// Interactive Schoenfeld residuals vs time plot
pub fn plot_schoenfeld(ui: &mut Ui, id: &str, points: &[(f64, f64)], beta: f64, height: f32) {
    if points.is_empty() {
        return;
    }
    let pts: Vec<[f64; 2]> = points.iter().map(|&(t, y)| [t, y]).collect();
    let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
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
    let trend = vec![
        [min_x, intercept + slope * min_x],
        [max_x, intercept + slope * max_x],
    ];

    Plot::new(id)
        .height(height)
        .width(ui.available_width().max(220.0))
        .legend(Legend::default())
        .x_axis_label("Tid")
        .y_axis_label("Skalert Schoenfeld Beta(t)")
        .show(ui, |plot_ui| {
            plot_ui.hline(
                HLine::new(beta)
                    .name(format!("Konstant β = {:.3}", beta))
                    .color(Color32::from_rgb(4, 120, 87))
                    .style(LineStyle::dashed_dense()),
            );
            plot_ui.line(
                Line::new(PlotPoints::new(trend))
                    .name("Trend over tid")
                    .color(Color32::from_rgb(220, 38, 38))
                    .width(2.0_f32),
            );
            plot_ui.points(
                Points::new(PlotPoints::new(pts))
                    .name("Hendelser")
                    .radius(3.4_f32)
                    .color(PRIMARY_BLUE),
            );
        });
}

/// Responsive Elbow (WCSS) & Silhouette vs k
pub fn plot_elbow_and_silhouette(ui: &mut Ui, id_prefix: &str, evals: &[ClusterKEval], height: f32) {
    let wcss_pts: Vec<[f64; 2]> = evals.iter().map(|e| [e.k as f64, e.wcss]).collect();
    let sil_pts: Vec<[f64; 2]> = evals
        .iter()
        .filter(|e| e.k >= 2)
        .map(|e| [e.k as f64, e.silhouette])
        .collect();

    responsive_two_col(
        ui,
        |ui| {
            ui.label(RichText::new("Elbow-metoden (WCSS)").strong().color(TEXT_PRIMARY));
            Plot::new(format!("{}_elbow", id_prefix))
                .height(height)
                .width(ui.available_width().max(220.0))
                .x_axis_label("Antall klynger (k)")
                .y_axis_label("WCSS (Inertia)")
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(PlotPoints::new(wcss_pts.clone()))
                            .color(PRIMARY_BLUE)
                            .width(2.2_f32),
                    );
                    plot_ui.points(
                        Points::new(PlotPoints::new(wcss_pts))
                            .radius(4.5_f32)
                            .color(PRIMARY_BLUE),
                    );
                });
        },
        |ui| {
            ui.label(RichText::new("Silhouette-skår (høyere er bedre)").strong().color(TEXT_PRIMARY));
            Plot::new(format!("{}_sil", id_prefix))
                .height(height)
                .width(ui.available_width().max(220.0))
                .x_axis_label("Antall klynger (k)")
                .y_axis_label("Gj.snitt Silhouette")
                .include_y(0.0)
                .include_y(1.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(PlotPoints::new(sil_pts.clone()))
                            .color(Color32::from_rgb(4, 120, 87))
                            .width(2.2_f32),
                    );
                    plot_ui.points(
                        Points::new(PlotPoints::new(sil_pts))
                            .radius(4.5_f32)
                            .color(Color32::from_rgb(4, 120, 87)),
                    );
                });
        },
    );
}

/// Interactive 2D PCA Cluster scatter plot
pub fn plot_clusters_2d(ui: &mut Ui, id: &str, pts: &[ClusterPoint2D], height: f32) {
    let max_c = pts.iter().map(|p| p.cluster).max().unwrap_or(1);
    let palette = [
        Color32::from_rgb(29, 78, 216),
        Color32::from_rgb(4, 120, 87),
        Color32::from_rgb(180, 83, 9),
        Color32::from_rgb(220, 38, 38),
        Color32::from_rgb(109, 40, 217),
        Color32::from_rgb(14, 116, 144),
        Color32::from_rgb(190, 24, 93),
        Color32::from_rgb(77, 124, 15),
    ];

    Plot::new(id)
        .height(height)
        .width(ui.available_width().max(220.0))
        .legend(Legend::default())
        .x_axis_label("Hovedkomponent 1 (PC1)")
        .y_axis_label("Hovedkomponent 2 (PC2)")
        .show(ui, |plot_ui| {
            for c in 1..=max_c {
                let c_pts: Vec<[f64; 2]> = pts
                    .iter()
                    .filter(|p| p.cluster == c)
                    .map(|p| [p.pc1, p.pc2])
                    .collect();
                if !c_pts.is_empty() {
                    plot_ui.points(
                        Points::new(PlotPoints::new(c_pts))
                            .name(format!("Klynge {}", c))
                            .radius(4.0_f32)
                            .color(palette[(c - 1) % palette.len()]),
                    );
                }
            }
        });
}
