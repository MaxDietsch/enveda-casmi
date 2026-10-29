use crate::cache::Peak;
use eframe::egui::{
    Align2, Color32, CornerRadius, FontId, Pos2, Rect, Response, Sense, Stroke, StrokeKind, Ui,
    Vec2,
};

pub struct PlotItem {
    pub spectrum: u32,
    pub title: String,
    pub detail: String,
    pub peaks: Vec<Peak>,
    pub precursor_mz: f32,
    pub color: Color32,
}

fn nice_step(raw: f32) -> f32 {
    if !raw.is_finite() || raw <= 0.0 {
        return 1.0;
    }
    let exp = raw.log10().floor();
    let base = 10f32.powf(exp);
    let f = raw / base;
    let n = if f <= 1.0 {
        1.0
    } else if f <= 2.0 {
        2.0
    } else if f <= 5.0 {
        5.0
    } else {
        10.0
    };
    n * base
}

pub fn show(ui: &mut Ui, items: &[PlotItem], mirror: bool, label_peaks: bool) -> Response {
    let desired = ui.available_size();
    let (response, painter) = ui.allocate_painter(desired, Sense::hover());
    let rect = response.rect;
    let visuals = ui.visuals().clone();
    let bg = visuals.extreme_bg_color;
    let fg = visuals.text_color();
    let weak = visuals.weak_text_color();
    painter.rect_filled(rect, CornerRadius::same(6), bg);
    painter.rect_stroke(
        rect,
        CornerRadius::same(6),
        Stroke::new(1.0_f32, weak.gamma_multiply(0.4)),
        StrokeKind::Inside,
    );
    if rect.width() < 140.0 || rect.height() < 100.0 {
        return response;
    }
    if items.is_empty() {
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "Select one or more spectra to plot",
            FontId::proportional(14.0),
            weak,
        );
        return response;
    }
    let plot = Rect::from_min_max(
        Pos2::new(rect.left() + 52.0, rect.top() + 20.0),
        Pos2::new(rect.right() - 16.0, rect.bottom() - 30.0),
    );
    if plot.width() < 30.0 || plot.height() < 30.0 {
        return response;
    }
    let mut x_min = f32::INFINITY;
    let mut x_max = f32::NEG_INFINITY;
    for item in items {
        for peak in &item.peaks {
            if peak.mz.is_finite() && peak.mz > 0.0 {
                x_min = x_min.min(peak.mz);
                x_max = x_max.max(peak.mz);
            }
        }
        if item.precursor_mz.is_finite() && item.precursor_mz > 0.0 {
            x_min = x_min.min(item.precursor_mz);
            x_max = x_max.max(item.precursor_mz);
        }
    }
    if !x_min.is_finite() || !x_max.is_finite() {
        x_min = 0.0;
        x_max = 1000.0;
    }
    if x_max - x_min < 5.0 {
        x_min = (x_min - 10.0).max(0.0);
        x_max += 10.0;
    }
    let pad_x = (x_max - x_min) * 0.03;
    let x_min = (x_min - pad_x).max(0.0);
    let x_max = x_max + pad_x;
    let map_x =
        |mz: f32| -> f32 { plot.left() + (mz - x_min) / (x_max - x_min).max(1e-6) * plot.width() };

    let baseline = if mirror {
        plot.center().y
    } else {
        plot.bottom() - 2.0
    };
    let half_height = if mirror {
        (plot.height() * 0.5 - 14.0).max(12.0)
    } else {
        (plot.height() - 22.0).max(12.0)
    };

    let grid = weak.gamma_multiply(0.18);
    let step = nice_step((x_max - x_min) / 6.0);
    let mut tick = (x_min / step).ceil() * step;
    while tick <= x_max + 1e-4 {
        let x = map_x(tick);
        painter.line_segment(
            [Pos2::new(x, plot.top()), Pos2::new(x, plot.bottom())],
            Stroke::new(1.0_f32, grid),
        );
        let label = if step < 1.0 {
            format!("{tick:.1}")
        } else {
            format!("{tick:.0}")
        };
        painter.text(
            Pos2::new(x, plot.bottom() + 5.0),
            Align2::CENTER_TOP,
            label,
            FontId::monospace(10.0),
            weak,
        );
        tick += step;
    }

    if mirror {
        for (frac, label) in [
            (0.0, "+100"),
            (0.25, "+50"),
            (0.5, "0"),
            (0.75, "-50"),
            (1.0, "-100"),
        ] {
            let y = plot.top() + plot.height() * frac;
            painter.line_segment(
                [Pos2::new(plot.left(), y), Pos2::new(plot.right(), y)],
                Stroke::new(1.0_f32, grid),
            );
            painter.text(
                Pos2::new(plot.left() - 6.0, y),
                Align2::RIGHT_CENTER,
                label,
                FontId::monospace(10.0),
                weak,
            );
        }
    } else {
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let y = baseline - half_height * frac;
            painter.line_segment(
                [Pos2::new(plot.left(), y), Pos2::new(plot.right(), y)],
                Stroke::new(1.0_f32, grid),
            );
            painter.text(
                Pos2::new(plot.left() - 6.0, y),
                Align2::RIGHT_CENTER,
                format!("{:.0}", frac * 100.0),
                FontId::monospace(10.0),
                weak,
            );
        }
    }
    painter.line_segment(
        [
            Pos2::new(plot.left(), baseline),
            Pos2::new(plot.right(), baseline),
        ],
        Stroke::new(1.2_f32, weak.gamma_multiply(0.8)),
    );
    painter.text(
        Pos2::new(plot.center().x, plot.bottom() + 18.0),
        Align2::CENTER_TOP,
        "m/z",
        FontId::proportional(11.0),
        weak,
    );
    painter.text(
        Pos2::new(plot.left(), plot.top() - 14.0),
        Align2::LEFT_TOP,
        if mirror {
            "relative intensity (mirrored)"
        } else {
            "relative intensity"
        },
        FontId::proportional(11.0),
        weak,
    );

    let mut labels_to_draw: Vec<(Pos2, String, Color32, bool)> = Vec::new();
    let mut seen_labels: std::collections::HashSet<i32> = std::collections::HashSet::new();
    for (item_index, item) in items.iter().enumerate() {
        let sign = if mirror && item_index % 2 == 1 {
            -1.0
        } else {
            1.0
        };
        let color = item.color;
        for peak in &item.peaks {
            let x = map_x(peak.mz);
            if x < plot.left() - 1.0 || x > plot.right() + 1.0 {
                continue;
            }
            let y = baseline - sign * peak.intensity.clamp(0.0, 1.0) * half_height;
            painter.line_segment(
                [Pos2::new(x, baseline), Pos2::new(x, y)],
                Stroke::new(1.6_f32, color),
            );
        }
        if label_peaks {
            let mut ranked: Vec<&Peak> = item
                .peaks
                .iter()
                .filter(|p| p.intensity >= 0.16 && p.mz >= x_min && p.mz <= x_max)
                .collect();
            ranked.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
            for peak in ranked.into_iter().take(3) {
                if !seen_labels.insert((peak.mz * 10.0).round() as i32) {
                    continue;
                }
                let x = map_x(peak.mz).clamp(plot.left() + 12.0, plot.right() - 12.0);
                let y = baseline - sign * peak.intensity.clamp(0.0, 1.0) * half_height;
                labels_to_draw.push((
                    Pos2::new(x, y + if sign > 0.0 { -3.0 } else { 3.0 }),
                    format!("{:.1}", peak.mz),
                    color,
                    sign > 0.0,
                ));
            }
        }
    }
    for (pos, text, color, above) in labels_to_draw {
        painter.text(
            pos,
            if above {
                Align2::CENTER_BOTTOM
            } else {
                Align2::CENTER_TOP
            },
            text,
            FontId::monospace(9.5),
            color,
        );
    }

    if let Some(hover) = response.hover_pos() {
        if plot.contains(hover) {
            let mut best: Option<(f32, usize, usize)> = None;
            for (item_index, item) in items.iter().enumerate() {
                for (peak_index, peak) in item.peaks.iter().enumerate() {
                    let x = map_x(peak.mz);
                    let dx = (x - hover.x).abs();
                    if dx <= 7.0 && best.is_none_or(|(best_dx, _, _)| dx < best_dx) {
                        best = Some((dx, item_index, peak_index));
                    }
                }
            }
            if let Some((_, item_index, peak_index)) = best {
                let item = &items[item_index];
                let peak = &item.peaks[peak_index];
                let x = map_x(peak.mz);
                let sign = if mirror && item_index % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                let y = baseline - sign * peak.intensity.clamp(0.0, 1.0) * half_height;
                painter.line_segment(
                    [Pos2::new(x, plot.top()), Pos2::new(x, plot.bottom())],
                    Stroke::new(1.0_f32, item.color.gamma_multiply(0.55)),
                );
                painter.circle_filled(Pos2::new(x, y), 3.0, item.color);
                let text = format!(
                    "m/z {:.4}   {:.1}%   {}  {}",
                    peak.mz,
                    peak.intensity * 100.0,
                    item.title,
                    item.detail
                );
                let width = (text.chars().count() as f32 * 6.4 + 14.0).min(plot.width());
                let info = Rect::from_min_size(
                    Pos2::new(plot.left() + 4.0, plot.top() + 4.0),
                    Vec2::new(width, 20.0),
                );
                painter.rect_filled(info, CornerRadius::same(3), bg.gamma_multiply(0.92));
                painter.rect_stroke(
                    info,
                    CornerRadius::same(3),
                    Stroke::new(1.0_f32, item.color.gamma_multiply(0.7)),
                    StrokeKind::Inside,
                );
                painter.text(
                    info.left_center() + Vec2::new(7.0, 0.0),
                    Align2::LEFT_CENTER,
                    text,
                    FontId::monospace(10.5),
                    fg,
                );
            }
        }
    }
    response.on_hover_cursor(eframe::egui::CursorIcon::Crosshair)
}
