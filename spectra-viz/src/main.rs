mod cache;
mod crash;
mod plot;
mod structure;

use cache::{format_ce, Catalog, Molecule};
use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, FontId, Label, Layout, Pos2, Rect, RichText,
    ScrollArea, Sense, Stroke, StrokeKind, TextEdit, Ui, Vec2,
};
use plot::PlotItem;
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Duration;
use structure::{StructureCache, StructureTexture};

const MAX_SELECTED: usize = 12;

const PALETTE: [Color32; 10] = [
    Color32::from_rgb(0x3b, 0x82, 0xf6),
    Color32::from_rgb(0xef, 0x6c, 0x3b),
    Color32::from_rgb(0x1f, 0xa8, 0x6b),
    Color32::from_rgb(0xb2, 0x5c, 0xe0),
    Color32::from_rgb(0xd9, 0xa4, 0x1e),
    Color32::from_rgb(0x2c, 0xbf, 0xcb),
    Color32::from_rgb(0xe0, 0x4d, 0x91),
    Color32::from_rgb(0x8a, 0xa3, 0x2e),
    Color32::from_rgb(0x8c, 0x72, 0xea),
    Color32::from_rgb(0x6b, 0x74, 0x80),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum SortMode {
    Mass,
    Formula,
    Count,
    Order,
}

impl SortMode {
    fn label(self) -> &'static str {
        match self {
            SortMode::Mass => "precursor mass",
            SortMode::Formula => "formula",
            SortMode::Count => "spectra count",
            SortMode::Order => "dataset order",
        }
    }
}

struct Filters {
    lib: Option<u8>,
    adduct: Option<u16>,
    ion: Option<u8>,
    mz_min: f32,
    mz_max: f32,
    sort: SortMode,
}

struct Options {
    data_path: PathBuf,
    cache_path: PathBuf,
    max_peaks: usize,
    rebuild: bool,
    cache_only: bool,
    check_smiles: bool,
    render_smiles: Option<String>,
    render_molecule: Option<usize>,
    out_path: PathBuf,
    watchdog: Option<u32>,
}

enum CacheEvent {
    Progress { frac: f32, msg: String },
    Done(Result<Catalog, String>),
}

struct Loading {
    frac: f32,
    msg: String,
}

struct App {
    opts: Options,
    catalog: Option<Catalog>,
    rx: Option<Receiver<CacheEvent>>,
    loading: Option<Loading>,
    error: Option<String>,
    search: String,
    filter: Filters,
    filtered: Vec<u32>,
    filter_dirty: bool,
    scroll_top: bool,
    scroll_to_active: bool,
    scroll_frames: u8,
    filters_initialized: bool,
    data_min_mz: f32,
    data_max_mz: f32,
    active: Option<u32>,
    selection: BTreeSet<u32>,
    colors: HashMap<u32, Color32>,
    plot_items: Vec<PlotItem>,
    structures: StructureCache,
    dark: bool,
    mirror: bool,
    show_labels: bool,
    status: Option<String>,
    frames: u64,
    ui_pass_marked: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, opts: Options) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        let mut app = Self {
            opts,
            catalog: None,
            rx: None,
            loading: None,
            error: None,
            search: String::new(),
            filter: Filters {
                lib: None,
                adduct: None,
                ion: None,
                mz_min: 0.0,
                mz_max: 2000.0,
                sort: SortMode::Mass,
            },
            filtered: Vec::new(),
            filter_dirty: true,
            scroll_top: false,
            scroll_to_active: false,
            scroll_frames: 0,
            filters_initialized: false,
            data_min_mz: 0.0,
            data_max_mz: 2000.0,
            active: None,
            selection: BTreeSet::new(),
            colors: HashMap::new(),
            plot_items: Vec::new(),
            structures: StructureCache::new(),
            dark: true,
            mirror: false,
            show_labels: false,
            status: None,
            frames: 0,
            ui_pass_marked: false,
        };
        app.spawn_loader(&cc.egui_ctx);
        crash::mark("loader thread spawned");
        app
    }

    fn spawn_loader(&mut self, ctx: &egui::Context) {
        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        self.error = None;
        self.loading = Some(Loading {
            frac: 0.0,
            msg: "Starting...".to_string(),
        });
        let data = self.opts.data_path.clone();
        let cache = self.opts.cache_path.clone();
        let max_peaks = self.opts.max_peaks;
        let rebuild = self.opts.rebuild;
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let tx_progress = tx.clone();
            let ctx_progress = ctx.clone();
            let mut send = move |frac: f32, msg: String| {
                let _ = tx_progress.send(CacheEvent::Progress { frac, msg });
                ctx_progress.request_repaint();
            };
            let result: Result<Catalog, String> = (|| {
                if !rebuild && cache.exists() {
                    send(0.05, "Loading cache...".to_string());
                    match cache::load_cache(&cache) {
                        Ok(catalog) => return Ok(catalog),
                        Err(err) => {
                            send(0.1, format!("Cache unreadable ({err}); rebuilding..."));
                        }
                    }
                }
                if !data.exists() {
                    return Err(format!(
                        "Training parquet not found: {}\nUse --data <path> to point at train.parquet.",
                        data.display()
                    ));
                }
                cache::build_cache(&data, &cache, max_peaks, &mut send)
                    .map_err(|err| format!("{err:#}"))?;
                cache::load_cache(&cache).map_err(|err| format!("{err:#}"))
            })();
            let _ = tx.send(CacheEvent::Done(result));
            ctx.request_repaint();
        });
    }

    fn drain_events(&mut self, ctx: &egui::Context) {
        let mut events = Vec::new();
        let mut disconnected = false;
        if let Some(rx) = &self.rx {
            loop {
                match rx.try_recv() {
                    Ok(event) => events.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        for event in events {
            match event {
                CacheEvent::Progress { frac, msg } => {
                    if let Some(loading) = &mut self.loading {
                        loading.frac = frac;
                        loading.msg = msg;
                    }
                }
                CacheEvent::Done(Ok(catalog)) => {
                    self.loading = None;
                    self.rx = None;
                    self.apply_catalog(catalog);
                }
                CacheEvent::Done(Err(err)) => {
                    crash::mark(&format!("loader failed: {err}"));
                    self.loading = None;
                    self.rx = None;
                    self.error = Some(err);
                }
            }
        }
        if disconnected {
            self.rx = None;
            if self.catalog.is_none() && self.error.is_none() {
                self.loading = None;
                self.error = Some("The cache loader stopped unexpectedly.".to_string());
            }
        }
        if self.loading.is_some() {
            ctx.request_repaint_after(Duration::from_millis(80));
        }
    }

    fn apply_catalog(&mut self, catalog: Catalog) {
        let mut min_mz = f32::INFINITY;
        let mut max_mz = f32::NEG_INFINITY;
        for molecule in &catalog.molecules {
            if molecule.min_mz.is_finite() {
                min_mz = min_mz.min(molecule.min_mz);
            }
            if molecule.max_mz.is_finite() {
                max_mz = max_mz.max(molecule.max_mz);
            }
        }
        self.data_min_mz = if min_mz.is_finite() { min_mz } else { 0.0 };
        self.data_max_mz = if max_mz.is_finite() { max_mz } else { 2000.0 };
        if !self.filters_initialized {
            self.filter.mz_min = (self.data_min_mz.floor() - 1.0).max(0.0);
            self.filter.mz_max = self.data_max_mz.ceil() + 1.0;
            self.filters_initialized = true;
        }
        self.catalog = Some(catalog);
        self.active = None;
        self.selection.clear();
        self.colors.clear();
        self.plot_items.clear();
        self.filter_dirty = true;
        self.recompute_filter();
        let initial = self
            .filtered
            .iter()
            .copied()
            .find(|&index| {
                let molecule = &self.catalog.as_ref().unwrap().molecules[index as usize];
                molecule.min_mz.is_finite() && molecule.min_mz >= 100.0 && molecule.max_mz <= 1000.0
            })
            .or_else(|| self.filtered.first().copied());
        if let Some(first) = initial {
            self.activate(first);
        }
        self.scroll_to_active = true;
        self.scroll_frames = 3;
        self.scroll_top = false;
        self.status = None;
        if let Some(catalog) = &self.catalog {
            crash::mark(&format!(
                "catalog applied: {} molecules, {} spectra",
                catalog.molecules.len(),
                catalog.spectra.len()
            ));
        }
    }

    fn recompute_filter(&mut self) {
        let Some(catalog) = &self.catalog else {
            return;
        };
        let query = self.search.trim().to_lowercase();
        let tokens: Vec<&str> = query.split_whitespace().collect();
        let mz_filter = self.filter.mz_min > self.data_min_mz - 0.5
            || self.filter.mz_max < self.data_max_mz + 0.5;
        let spectrum_filter = self.filter.lib.is_some()
            || self.filter.adduct.is_some()
            || self.filter.ion.is_some()
            || mz_filter;
        let mut out = Vec::with_capacity(catalog.molecules.len());
        for (idx, molecule) in catalog.molecules.iter().enumerate() {
            let text_match = tokens.iter().all(|token| {
                molecule.smiles_lower.contains(token)
                    || molecule.formula_lower.contains(token)
                    || molecule.inchikey_lower.contains(token)
            });
            if !text_match {
                continue;
            }
            if spectrum_filter {
                let mut ok = false;
                for &spectrum_idx in &molecule.spectra {
                    let spectrum = &catalog.spectra[spectrum_idx as usize];
                    if let Some(lib) = self.filter.lib {
                        if spectrum.lib != lib {
                            continue;
                        }
                    }
                    if let Some(adduct) = self.filter.adduct {
                        if spectrum.adduct != adduct {
                            continue;
                        }
                    }
                    if let Some(ion) = self.filter.ion {
                        if spectrum.ionization != ion {
                            continue;
                        }
                    }
                    if spectrum.precursor_mz.is_finite()
                        && (spectrum.precursor_mz < self.filter.mz_min
                            || spectrum.precursor_mz > self.filter.mz_max)
                    {
                        continue;
                    }
                    if !spectrum.precursor_mz.is_finite() {
                        continue;
                    }
                    ok = true;
                    break;
                }
                if !ok {
                    continue;
                }
            }
            out.push(idx as u32);
        }
        match self.filter.sort {
            SortMode::Order => {}
            SortMode::Mass => out.sort_by(|&a, &b| {
                let mass_a = catalog.molecules[a as usize].min_mz;
                let mass_b = catalog.molecules[b as usize].min_mz;
                mass_a.total_cmp(&mass_b).then(a.cmp(&b))
            }),
            SortMode::Formula => out.sort_by(|&a, &b| {
                catalog.molecules[a as usize]
                    .formula_lower
                    .cmp(&catalog.molecules[b as usize].formula_lower)
                    .then(a.cmp(&b))
            }),
            SortMode::Count => out.sort_by(|&a, &b| {
                let ca = catalog.molecules[a as usize].spectra.len();
                let cb = catalog.molecules[b as usize].spectra.len();
                cb.cmp(&ca).then(a.cmp(&b))
            }),
        }
        self.filtered = out;
        self.filter_dirty = false;
        self.scroll_top = true;
    }

    fn next_color(&self) -> Color32 {
        for color in PALETTE {
            if !self.colors.values().any(|used| *used == color) {
                return color;
            }
        }
        PALETTE[self.selection.len() % PALETTE.len()]
    }

    fn activate(&mut self, molecule: u32) {
        if self.active == Some(molecule) {
            return;
        }
        self.active = Some(molecule);
        self.selection.clear();
        self.colors.clear();
        if let Some(catalog) = &self.catalog {
            if let Some(&first) = catalog.molecules[molecule as usize].spectra.first() {
                self.selection.insert(first);
                self.colors.insert(first, PALETTE[0]);
            }
        }
        self.rebuild_plot();
    }

    fn clear_selection(&mut self) {
        self.selection.clear();
        self.colors.clear();
        self.rebuild_plot();
    }

    fn select_all_active(&mut self) {
        let Some(catalog) = &self.catalog else {
            return;
        };
        let Some(active) = self.active else {
            return;
        };
        let total = catalog.molecules[active as usize].spectra.len();
        self.selection.clear();
        self.colors.clear();
        for &idx in catalog.molecules[active as usize]
            .spectra
            .iter()
            .take(MAX_SELECTED)
        {
            let color = self.next_color();
            self.colors.insert(idx, color);
            self.selection.insert(idx);
        }
        self.status = if total > MAX_SELECTED {
            Some(format!(
                "Selected the first {MAX_SELECTED} of {total} spectra"
            ))
        } else {
            None
        };
        self.rebuild_plot();
    }

    fn toggle_spectrum(&mut self, idx: u32) {
        if self.selection.contains(&idx) {
            self.selection.remove(&idx);
            self.colors.remove(&idx);
        } else {
            if self.selection.len() >= MAX_SELECTED {
                self.status = Some(format!("At most {MAX_SELECTED} spectra can be overlaid"));
                return;
            }
            let color = self.next_color();
            self.colors.insert(idx, color);
            self.selection.insert(idx);
        }
        self.rebuild_plot();
    }

    fn rebuild_plot(&mut self) {
        let Some(catalog) = self.catalog.as_mut() else {
            self.plot_items.clear();
            return;
        };
        let mut items = Vec::with_capacity(self.selection.len());
        for &idx in &self.selection {
            let spectrum = catalog.spectra[idx as usize];
            let count = if spectrum.num_peaks == 0 {
                catalog.peaks.max_peaks()
            } else {
                (spectrum.num_peaks as usize).min(catalog.peaks.max_peaks())
            };
            match catalog.peaks.load(idx, count) {
                Ok(peaks) => {
                    let molecule = &catalog.molecules[spectrum.mol as usize];
                    let adduct = catalog
                        .adducts
                        .get(spectrum.adduct as usize)
                        .map(String::as_str)
                        .unwrap_or("?");
                    let detail = format!(
                        "{}  CE {}",
                        adduct,
                        format_ce(spectrum.ce_lo, spectrum.ce_hi)
                    );
                    items.push(PlotItem {
                        spectrum: idx,
                        title: if molecule.formula.is_empty() {
                            "?".to_string()
                        } else {
                            molecule.formula.clone()
                        },
                        detail,
                        peaks,
                        precursor_mz: spectrum.precursor_mz,
                        color: *self.colors.get(&idx).unwrap_or(&PALETTE[0]),
                    });
                }
                Err(err) => {
                    self.status = Some(format!(
                        "Could not read peaks for row {}: {err}",
                        spectrum.row
                    ));
                }
            }
        }
        self.plot_items = items;
    }

    fn selected_molecule_color(&self, molecule: u32) -> Option<Color32> {
        let catalog = self.catalog.as_ref()?;
        for &idx in &self.selection {
            if catalog.spectra[idx as usize].mol == molecule {
                return self.colors.get(&idx).copied();
            }
        }
        None
    }

    fn selected_count(&self, molecule: u32) -> usize {
        let Some(catalog) = &self.catalog else {
            return 0;
        };
        self.selection
            .iter()
            .filter(|&&idx| catalog.spectra[idx as usize].mol == molecule)
            .count()
    }

    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                ui.strong("CASMI26 Spectra Viewer");
                ui.separator();
                if let Some(catalog) = &self.catalog {
                    ui.label(format!("{} spectra", catalog.spectra.len()));
                    ui.label(format!("{} molecules", catalog.molecules.len()));
                    ui.label(
                        RichText::new(format!("top {} peaks", catalog.peaks.max_peaks())).weak(),
                    );
                } else if self.loading.is_some() {
                    ui.spinner();
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let theme = if self.dark { "Light" } else { "Dark" };
                    if ui.button(theme).clicked() {
                        self.dark = !self.dark;
                        ui.ctx().set_theme(if self.dark {
                            egui::Theme::Dark
                        } else {
                            egui::Theme::Light
                        });
                    }
                    if let Some(status) = &self.status {
                        ui.label(RichText::new(status).small().weak());
                    }
                });
            });
            ui.add_space(3.0);
        });
    }

    fn loading_ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(100.0);
                if let Some(err) = &self.error {
                    ui.heading("Could not load the dataset");
                    ui.add_space(8.0);
                    ui.label(RichText::new(err).color(ui.visuals().error_fg_color));
                    ui.add_space(16.0);
                    if ui.button("Retry (rebuild cache)").clicked() {
                        self.opts.rebuild = true;
                        self.spawn_loader(ctx);
                    }
                } else if let Some(loading) = &self.loading {
                    ui.heading("Preparing spectra cache");
                    ui.add_space(8.0);
                    ui.label(&loading.msg);
                    ui.add_space(8.0);
                    ui.add(
                        egui::ProgressBar::new(loading.frac)
                            .desired_width(460.0)
                            .show_percentage(),
                    );
                    ui.add_space(12.0);
                    ui.label(
                        RichText::new(
                            "The first launch indexes train.parquet and writes a compact cache. \
                             Later launches load the cache directly.",
                        )
                        .weak(),
                    );
                    ui.label(
                        RichText::new(format!("cache: {}", self.opts.cache_path.display()))
                            .monospace()
                            .small()
                            .weak(),
                    );
                }
            });
        });
    }

    fn left_panel(&mut self, ctx: &egui::Context) {
        let Some(catalog) = &self.catalog else {
            return;
        };
        let libs = catalog.libs.clone();
        let adducts = catalog.adducts.clone();
        let mut molecule_colors: HashMap<u32, Color32> = HashMap::new();
        for &idx in &self.selection {
            let molecule = catalog.spectra[idx as usize].mol;
            if let Some(&color) = self.colors.get(&idx) {
                molecule_colors.entry(molecule).or_insert(color);
            }
        }
        let mut clicked = None;
        let active = self.active;
        let filtered = &self.filtered;
        egui::SidePanel::left("browser")
            .resizable(true)
            .default_width(370.0)
            .min_width(290.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Search");
                    let response = ui.add(
                        TextEdit::singleline(&mut self.search)
                            .hint_text("SMILES, formula, InChIKey")
                            .desired_width(f32::INFINITY),
                    );
                    if response.changed() {
                        self.filter_dirty = true;
                    }
                });
                ui.add_space(4.0);
                egui::Grid::new("filters")
                    .num_columns(2)
                    .spacing([6.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Library");
                        egui::ComboBox::from_id_salt("lib_filter")
                            .selected_text(
                                self.filter
                                    .lib
                                    .and_then(|id| libs.get(id as usize))
                                    .cloned()
                                    .unwrap_or_else(|| "all".to_string()),
                            )
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_value(&mut self.filter.lib, None, "all")
                                    .changed()
                                {
                                    self.filter_dirty = true;
                                }
                                for (id, name) in libs.iter().enumerate() {
                                    if ui
                                        .selectable_value(
                                            &mut self.filter.lib,
                                            Some(id as u8),
                                            name,
                                        )
                                        .changed()
                                    {
                                        self.filter_dirty = true;
                                    }
                                }
                            });
                        ui.end_row();
                        ui.label("Adduct");
                        egui::ComboBox::from_id_salt("adduct_filter")
                            .selected_text(
                                self.filter
                                    .adduct
                                    .and_then(|id| adducts.get(id as usize))
                                    .cloned()
                                    .unwrap_or_else(|| "all".to_string()),
                            )
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_value(&mut self.filter.adduct, None, "all")
                                    .changed()
                                {
                                    self.filter_dirty = true;
                                }
                                for (id, name) in adducts.iter().enumerate() {
                                    if ui
                                        .selectable_value(
                                            &mut self.filter.adduct,
                                            Some(id as u16),
                                            name,
                                        )
                                        .changed()
                                    {
                                        self.filter_dirty = true;
                                    }
                                }
                            });
                        ui.end_row();
                        ui.label("Ionization");
                        egui::ComboBox::from_id_salt("ion_filter")
                            .selected_text(match self.filter.ion {
                                Some(1) => "positive",
                                Some(2) => "negative",
                                _ => "all",
                            })
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_value(&mut self.filter.ion, None, "all")
                                    .changed()
                                {
                                    self.filter_dirty = true;
                                }
                                if ui
                                    .selectable_value(&mut self.filter.ion, Some(1), "positive")
                                    .changed()
                                {
                                    self.filter_dirty = true;
                                }
                                if ui
                                    .selectable_value(&mut self.filter.ion, Some(2), "negative")
                                    .changed()
                                {
                                    self.filter_dirty = true;
                                }
                            });
                        ui.end_row();
                        ui.label("Sort");
                        egui::ComboBox::from_id_salt("sort")
                            .selected_text(self.filter.sort.label())
                            .show_ui(ui, |ui| {
                                for mode in [
                                    SortMode::Mass,
                                    SortMode::Formula,
                                    SortMode::Count,
                                    SortMode::Order,
                                ] {
                                    if ui
                                        .selectable_value(&mut self.filter.sort, mode, mode.label())
                                        .changed()
                                    {
                                        self.filter_dirty = true;
                                    }
                                }
                            });
                        ui.end_row();
                        ui.label("m/z range");
                        ui.horizontal(|ui| {
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.filter.mz_min)
                                        .speed(5.0)
                                        .range(0.0..=5000.0)
                                        .prefix("min "),
                                )
                                .changed()
                            {
                                self.filter_dirty = true;
                            }
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.filter.mz_max)
                                        .speed(5.0)
                                        .range(0.0..=5000.0)
                                        .prefix("max "),
                                )
                                .changed()
                            {
                                self.filter_dirty = true;
                            }
                        });
                        ui.end_row();
                    });
                ui.horizontal(|ui| {
                    if ui.button("Reset filters").clicked() {
                        self.filter.lib = None;
                        self.filter.adduct = None;
                        self.filter.ion = None;
                        self.filter.sort = SortMode::Mass;
                        self.filter.mz_min = (self.data_min_mz.floor() - 1.0).max(0.0);
                        self.filter.mz_max = self.data_max_mz.ceil() + 1.0;
                        self.filter_dirty = true;
                    }
                    ui.label(RichText::new(format!("{} molecules", filtered.len())).weak());
                });
                ui.separator();
                if filtered.is_empty() {
                    ui.add_space(12.0);
                    ui.label(RichText::new("No molecules match the current filters.").weak());
                    return;
                }
                let row_height = 22.0;
                let mut offset = None;
                if self.scroll_to_active || self.scroll_frames > 0 {
                    if let Some(active) = active {
                        if let Some(position) = filtered.iter().position(|&item| item == active) {
                            offset = Some((position as f32 * row_height - 80.0).max(0.0));
                        }
                    }
                    self.scroll_to_active = false;
                    self.scroll_frames = self.scroll_frames.saturating_sub(1);
                    if self.scroll_frames > 0 {
                        ui.ctx().request_repaint();
                    }
                    self.scroll_top = false;
                } else if self.scroll_top {
                    offset = Some(0.0);
                    self.scroll_top = false;
                }
                let mut area = ScrollArea::vertical()
                    .id_salt("molecule_list")
                    .auto_shrink([false, false]);
                if let Some(offset) = offset {
                    area = area.vertical_scroll_offset(offset);
                }
                area.show_rows(ui, row_height, filtered.len(), |ui, range| {
                    if let Some(catalog) = &self.catalog {
                        for pos in range {
                            let molecule_idx = filtered[pos];
                            let molecule = &catalog.molecules[molecule_idx as usize];
                            let dot = molecule_colors.get(&molecule_idx).copied();
                            let response =
                                molecule_row(ui, molecule, active == Some(molecule_idx), dot);
                            if response.clicked() {
                                clicked = Some(molecule_idx);
                            }
                        }
                    }
                });
            });
        if let Some(molecule) = clicked {
            self.activate(molecule);
        }
    }

    fn right_panel(&mut self, ctx: &egui::Context) {
        let Some(catalog) = &self.catalog else {
            return;
        };
        let mut cards: Vec<(u32, Option<Color32>, usize, bool)> = Vec::new();
        if let Some(active) = self.active {
            cards.push((
                active,
                self.selected_molecule_color(active),
                self.selected_count(active),
                true,
            ));
        }
        let mut selected_molecules = Vec::new();
        for &idx in &self.selection {
            let molecule = catalog.spectra[idx as usize].mol;
            if !selected_molecules.contains(&molecule) {
                selected_molecules.push(molecule);
            }
        }
        for molecule in selected_molecules {
            if Some(molecule) != self.active {
                cards.push((
                    molecule,
                    self.selected_molecule_color(molecule),
                    self.selected_count(molecule),
                    false,
                ));
            }
        }
        egui::SidePanel::right("structures")
            .resizable(true)
            .default_width(400.0)
            .min_width(320.0)
            .show(ctx, |ui| {
                ScrollArea::vertical()
                    .id_salt("structure_list")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if cards.is_empty() {
                            ui.add_space(20.0);
                            ui.label("Select a molecule and one or more of its spectra.");
                            return;
                        }
                        if let Some(catalog) = &self.catalog {
                            for (molecule_idx, color, count, is_active) in &cards {
                                let molecule = &catalog.molecules[*molecule_idx as usize];
                                let structure =
                                    self.structures
                                        .get(ui.ctx(), *molecule_idx, &molecule.smiles);
                                let badge = if *is_active {
                                    format!("browsing | {count} selected")
                                } else {
                                    format!("{count} selected")
                                };
                                structure_card(ui, molecule, structure, *color, *is_active, &badge);
                                ui.add_space(6.0);
                            }
                        }
                    });
            });
    }

    fn plot_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("plot")
            .resizable(true)
            .default_height(330.0)
            .min_height(180.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.strong("Overlay");
                    ui.label(RichText::new(format!("{} spectra", self.plot_items.len())).weak());
                    ui.checkbox(&mut self.mirror, "Mirror");
                    ui.checkbox(&mut self.show_labels, "Peak labels");
                    if ui.button("Clear").clicked() {
                        self.selection.clear();
                        self.colors.clear();
                        self.rebuild_plot();
                    }
                    if let Some(catalog) = &self.catalog {
                        ui.label(
                            RichText::new(format!(
                                "| top {} peaks by intensity",
                                catalog.peaks.max_peaks()
                            ))
                            .weak()
                            .small(),
                        );
                    }
                });
                let mut remove = None;
                if !self.plot_items.is_empty() {
                    ui.horizontal_wrapped(|ui| {
                        for item in &self.plot_items {
                            let (rect, _) =
                                ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
                            ui.painter()
                                .rect_filled(rect, CornerRadius::same(2), item.color);
                            ui.label(RichText::new(&item.title).color(item.color).small());
                            ui.label(RichText::new(&item.detail).small().weak());
                            if ui.small_button("x").clicked() {
                                remove = Some(item.spectrum);
                            }
                            ui.add_space(6.0);
                        }
                    });
                }
                ui.separator();
                plot::show(ui, &self.plot_items, self.mirror, self.show_labels);
                if let Some(idx) = remove {
                    self.toggle_spectrum(idx);
                }
            });
    }

    fn central_panel(&mut self, ctx: &egui::Context) {
        let Some(catalog) = &self.catalog else {
            return;
        };
        let mut clicked = None;
        let mut select_all = false;
        let mut clear = false;
        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(active) = self.active else {
                ui.centered_and_justified(|ui| {
                    ui.label("Select a molecule from the list.");
                });
                return;
            };
            let molecule = &catalog.molecules[active as usize];
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let formula = if molecule.formula.is_empty() {
                    "?"
                } else {
                    molecule.formula.as_str()
                };
                ui.heading(formula);
                ui.label(
                    RichText::new(&molecule.inchikey14)
                        .monospace()
                        .small()
                        .weak(),
                );
                ui.label(RichText::new(format!("{} spectra", molecule.spectra.len())).weak());
                if ui.button("Select all").clicked() {
                    select_all = true;
                }
                if ui.button("Clear selection").clicked() {
                    clear = true;
                }
            });
            ui.add(Label::new(RichText::new(&molecule.smiles).monospace().size(11.0)).wrap());
            ui.add_space(4.0);
            ui.separator();
            ui.label(
                RichText::new("Click spectra to add or remove them from the overlay.")
                    .small()
                    .weak(),
            );
            spectrum_header(ui);
            ScrollArea::vertical()
                .id_salt("spectrum_list")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for &idx in &molecule.spectra {
                        let selected = self.selection.contains(&idx);
                        let color = self.colors.get(&idx).copied();
                        let response = spectrum_row(ui, catalog, idx, selected, color);
                        if response.clicked() {
                            clicked = Some(idx);
                        }
                    }
                });
        });
        if select_all {
            self.select_all_active();
        }
        if clear {
            self.clear_selection();
        }
        if let Some(idx) = clicked {
            self.toggle_spectrum(idx);
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frames += 1;
        if self.frames <= 2 {
            crash::mark(&format!("frame {}", self.frames));
        }
        self.drain_events(ctx);
        if self.filter_dirty {
            self.recompute_filter();
        }
        self.top_bar(ctx);
        if self.catalog.is_none() {
            self.loading_ui(ctx);
            return;
        }
        self.left_panel(ctx);
        self.right_panel(ctx);
        self.plot_panel(ctx);
        self.central_panel(ctx);
        if !self.ui_pass_marked && self.catalog.is_some() {
            self.ui_pass_marked = true;
            crash::mark("first full UI pass complete");
        }
    }
}

fn precursor_text(mz: f32) -> String {
    if mz.is_finite() && mz >= 10.0 {
        format!("{mz:.4}")
    } else {
        "-".to_string()
    }
}

fn shorten(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('~');
    out
}

fn molecule_row(
    ui: &mut Ui,
    molecule: &Molecule,
    active: bool,
    selected_color: Option<Color32>,
) -> egui::Response {
    let width = ui.available_width();
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, 22.0), Sense::click());
    if ui.is_rect_visible(rect) {
        let visuals = ui.visuals().clone();
        let painter = ui.painter();
        if active {
            painter.rect_filled(rect, CornerRadius::same(3), visuals.selection.bg_fill);
        } else if response.hovered() {
            painter.rect_filled(rect, CornerRadius::same(3), visuals.widgets.hovered.bg_fill);
        }
        let base_color = if active {
            visuals.selection.stroke.color
        } else {
            visuals.text_color()
        };
        let mut x = rect.left() + 6.0;
        if let Some(color) = selected_color {
            painter.circle_filled(Pos2::new(x + 3.0, rect.center().y), 3.2, color);
            x += 12.0;
        }
        let font = FontId::monospace(11.0);
        let formula = if molecule.formula.is_empty() {
            "?"
        } else {
            molecule.formula.as_str()
        };
        painter.text(
            Pos2::new(x, rect.center().y),
            Align2::LEFT_CENTER,
            formula,
            font.clone(),
            base_color,
        );
        let count = format!("{}", molecule.spectra.len());
        painter.text(
            Pos2::new(rect.right() - 6.0, rect.center().y),
            Align2::RIGHT_CENTER,
            count,
            font.clone(),
            base_color.gamma_multiply(0.7),
        );
        let max_chars = ((width - 168.0) / 6.3).max(6.0) as usize;
        painter.text(
            Pos2::new(rect.left() + 104.0, rect.center().y),
            Align2::LEFT_CENTER,
            shorten(&molecule.smiles, max_chars),
            font,
            base_color.gamma_multiply(0.82),
        );
    }
    response
}

fn spectrum_header(ui: &mut Ui) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 18.0), Sense::hover());
    let painter = ui.painter();
    let color = ui.visuals().weak_text_color();
    let font = FontId::monospace(10.0);
    painter.text(
        Pos2::new(rect.left() + 26.0, rect.center().y),
        Align2::LEFT_CENTER,
        "adduct",
        font.clone(),
        color,
    );
    painter.text(
        Pos2::new(rect.left() + 112.0, rect.center().y),
        Align2::LEFT_CENTER,
        "CE (eV)",
        font.clone(),
        color,
    );
    painter.text(
        Pos2::new(rect.left() + 198.0, rect.center().y),
        Align2::LEFT_CENTER,
        "precursor",
        font.clone(),
        color,
    );
    painter.text(
        Pos2::new(rect.left() + 292.0, rect.center().y),
        Align2::LEFT_CENTER,
        "peaks",
        font.clone(),
        color,
    );
    if width > 430.0 {
        painter.text(
            Pos2::new(rect.left() + 350.0, rect.center().y),
            Align2::LEFT_CENTER,
            "library",
            font,
            color,
        );
    }
}

fn spectrum_row(
    ui: &mut Ui,
    catalog: &Catalog,
    idx: u32,
    selected: bool,
    color: Option<Color32>,
) -> egui::Response {
    let spectrum = catalog.spectra[idx as usize];
    let width = ui.available_width();
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, 22.0), Sense::click());
    let adduct = catalog
        .adducts
        .get(spectrum.adduct as usize)
        .map(String::as_str)
        .unwrap_or("?");
    let lib = catalog
        .libs
        .get(spectrum.lib as usize)
        .map(String::as_str)
        .unwrap_or("?");
    if ui.is_rect_visible(rect) {
        let visuals = ui.visuals().clone();
        let painter = ui.painter();
        let fill = if selected {
            color
                .map(|c| Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), 36))
                .unwrap_or(visuals.selection.bg_fill)
        } else if response.hovered() {
            visuals.widgets.hovered.bg_fill
        } else {
            Color32::TRANSPARENT
        };
        if fill != Color32::TRANSPARENT {
            painter.rect_filled(rect, CornerRadius::same(3), fill);
        }
        if selected {
            if let Some(color) = color {
                painter.rect_filled(
                    Rect::from_min_size(rect.min, Vec2::new(3.0, rect.height())),
                    CornerRadius::same(1),
                    color,
                );
            }
        }
        let checkbox = Rect::from_center_size(
            Pos2::new(rect.left() + 12.0, rect.center().y),
            Vec2::splat(11.0),
        );
        painter.rect_stroke(
            checkbox,
            CornerRadius::same(2),
            Stroke::new(1.0_f32, visuals.weak_text_color()),
            StrokeKind::Inside,
        );
        if selected {
            painter.rect_filled(
                checkbox.shrink(2.0),
                CornerRadius::same(1),
                color.unwrap_or(visuals.selection.bg_fill),
            );
        }
        let font = FontId::monospace(11.5);
        let text_color = visuals.text_color();
        let weak = visuals.weak_text_color();
        painter.text(
            Pos2::new(rect.left() + 26.0, rect.center().y),
            Align2::LEFT_CENTER,
            adduct,
            font.clone(),
            text_color,
        );
        painter.text(
            Pos2::new(rect.left() + 112.0, rect.center().y),
            Align2::LEFT_CENTER,
            format_ce(spectrum.ce_lo, spectrum.ce_hi),
            font.clone(),
            text_color,
        );
        let precursor = precursor_text(spectrum.precursor_mz);
        painter.text(
            Pos2::new(rect.left() + 198.0, rect.center().y),
            Align2::LEFT_CENTER,
            precursor,
            font.clone(),
            text_color,
        );
        let peaks = if spectrum.num_peaks > 0 {
            spectrum.num_peaks.to_string()
        } else {
            "?".to_string()
        };
        painter.text(
            Pos2::new(rect.left() + 292.0, rect.center().y),
            Align2::LEFT_CENTER,
            peaks,
            font.clone(),
            text_color,
        );
        if width > 430.0 {
            painter.text(
                Pos2::new(rect.left() + 350.0, rect.center().y),
                Align2::LEFT_CENTER,
                lib,
                font.clone(),
                text_color,
            );
        }
        if width > 500.0 {
            painter.text(
                Pos2::new(rect.right() - 6.0, rect.center().y),
                Align2::RIGHT_CENTER,
                format!("#{}", spectrum.row),
                FontId::monospace(9.5),
                weak,
            );
        }
    }
    let base_peak = if spectrum.base_peak.is_finite() {
        format!("{:.0}", spectrum.base_peak)
    } else {
        "-".to_string()
    };
    response.on_hover_text(format!(
        "row {}\nprecursor m/z {}\nbase peak intensity {}\npeaks {}\nadduct {}\nCE {}\nlibrary {}",
        spectrum.row,
        precursor_text(spectrum.precursor_mz),
        base_peak,
        spectrum.num_peaks,
        adduct,
        format_ce(spectrum.ce_lo, spectrum.ce_hi),
        lib
    ))
}

fn structure_card(
    ui: &mut Ui,
    molecule: &Molecule,
    structure: &StructureTexture,
    color: Option<Color32>,
    is_active: bool,
    badge: &str,
) {
    let weak = ui.visuals().weak_text_color();
    let border = color.unwrap_or_else(|| {
        if is_active {
            weak
        } else {
            weak.gamma_multiply(0.5)
        }
    });
    let frame = egui::Frame::group(ui.style())
        .fill(ui.visuals().faint_bg_color)
        .stroke(Stroke::new(
            if color.is_some() { 1.5_f32 } else { 1.0_f32 },
            border,
        ))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(8.0);
    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            let formula = if molecule.formula.is_empty() {
                "?"
            } else {
                molecule.formula.as_str()
            };
            ui.strong(formula);
            ui.label(RichText::new(badge).small().weak());
        });
        let size = Vec2::new(ui.available_width(), 230.0);
        let (response, painter) = ui.allocate_painter(size, Sense::hover());
        let rect = response.rect;
        painter.rect_filled(rect, CornerRadius::same(4), Color32::WHITE);
        painter.rect_stroke(
            rect,
            CornerRadius::same(4),
            Stroke::new(1.0_f32, weak.gamma_multiply(0.35)),
            StrokeKind::Inside,
        );
        let _response = if let Some(texture) = &structure.texture {
            let texture_size = texture.size_vec2();
            let scale = (rect.width() / texture_size.x)
                .min(rect.height() / texture_size.y)
                .max(0.01);
            let image_rect = Rect::from_center_size(rect.center(), texture_size * scale);
            painter.image(
                texture.id(),
                image_rect,
                Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
            response
        } else {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "depiction failed",
                FontId::proportional(12.0),
                Color32::from_rgb(0xaa, 0x44, 0x44),
            );
            if let Some(error) = &structure.error {
                response.on_hover_text(error)
            } else {
                response
            }
        };
        ui.horizontal(|ui| {
            if ui.small_button("copy").clicked() {
                ui.ctx().copy_text(molecule.smiles.clone());
            }
            ui.label(RichText::new("SMILES").small().weak());
            if structure.texture.is_none() {
                ui.label(RichText::new("no depiction").small().weak());
            }
        });
        ui.add(Label::new(RichText::new(&molecule.smiles).monospace().size(10.5)).wrap());
        if !molecule.inchikey14.is_empty() {
            ui.label(
                RichText::new(&molecule.inchikey14)
                    .monospace()
                    .size(10.0)
                    .weak(),
            );
        }
    });
}

fn run_watchdog(parent: u32) {
    let pid = parent as i32;
    let deadline = std::time::Instant::now() + Duration::from_secs(6 * 60 * 60);
    loop {
        std::thread::sleep(Duration::from_millis(250));
        if std::time::Instant::now() > deadline {
            return;
        }
        if unsafe { libc::kill(pid, 0) } != 0 {
            break;
        }
    }
    std::thread::sleep(Duration::from_millis(1500));
    if crash::clean_exit() {
        return;
    }
    collect_crash_report();
}

fn collect_crash_report() {
    let Some(home) = std::env::var_os("HOME") else {
        return;
    };
    let dir = PathBuf::from(home).join("Library/Logs/DiagnosticReports");
    let target = crash::exe_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("spectra-viz-crash-report.txt");
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(path) = newest_crash_report(&dir) {
            match std::fs::copy(&path, &target) {
                Ok(_) => eprintln!(
                    "spectra-viz: copied macOS crash report {} to {}",
                    path.display(),
                    target.display()
                ),
                Err(err) => {
                    let _ = std::fs::write(&target, format!("could not copy crash report: {err}"));
                }
            }
            return;
        }
        if std::time::Instant::now() > deadline {
            let _ = std::fs::write(
                &target,
                "spectra-viz crashed (SIGNAL marker present) but no macOS crash report was \
                 found in ~/Library/Logs/DiagnosticReports within 30s (ReportCrash may be \
                 disabled). See spectra-viz-diagnostics.log for the signal handler output.",
            );
            return;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn newest_crash_report(dir: &std::path::Path) -> Option<PathBuf> {
    let cutoff = std::time::SystemTime::now() - Duration::from_secs(180);
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries {
        let entry = entry.ok()?;
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("spectra-viz") {
            continue;
        }
        let modified = entry.metadata().ok()?.modified().ok()?;
        if modified < cutoff {
            continue;
        }
        if best
            .as_ref()
            .map(|(time, _)| modified > *time)
            .unwrap_or(true)
        {
            best = Some((modified, entry.path()));
        }
    }
    best.map(|(_, path)| path)
}

fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        crash::mark(&format!("PANIC: {info}"));
        let backtrace = std::backtrace::Backtrace::capture();
        let text = format!(
            "spectra-viz panicked at {}:\n{info}\n{backtrace}\n",
            info.location()
                .map(|location| location.to_string())
                .unwrap_or_default(),
        );
        eprintln!("{text}");
        let mut targets = vec![PathBuf::from("spectra-viz-panic.log")];
        if let Some(dir) = std::env::current_exe()
            .ok()
            .and_then(|path| path.parent().map(|parent| parent.to_path_buf()))
        {
            targets.push(dir.join("spectra-viz-panic.log"));
        }
        for target in targets {
            let _ = std::fs::write(&target, &text);
        }
    }));
}

fn parse_args() -> Option<Options> {
    let mut data = None;
    let mut cache_path = None;
    let mut max_peaks = 32usize;
    let mut rebuild = false;
    let mut cache_only = false;
    let mut check_smiles = false;
    let mut render_smiles = None;
    let mut render_molecule = None;
    let mut watchdog = None;
    let mut out_path = PathBuf::from("molecule.svg");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data" => data = args.next().map(PathBuf::from),
            "--cache" => cache_path = args.next().map(PathBuf::from),
            "--max-peaks" => {
                max_peaks = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .filter(|&value| value > 0)
                    .unwrap_or(32);
            }
            "--rebuild" => rebuild = true,
            "--cache-only" => cache_only = true,
            "--check-smiles" => check_smiles = true,
            "--render-smiles" => render_smiles = args.next(),
            "--render-molecule" => {
                render_molecule = args.next().and_then(|value| value.parse().ok())
            }
            "--watchdog" => watchdog = args.next().and_then(|value| value.parse().ok()),
            "--out" => {
                if let Some(value) = args.next() {
                    out_path = PathBuf::from(value);
                }
            }
            "-h" | "--help" => {
                println!(
                    "Usage: spectra-viz [OPTIONS] [DATA]\n\
                     \n\
                     Options:\n\
                     \x20 --data <PATH>       train.parquet (default: auto-detect data/train.parquet)\n\
                     \x20 --cache <PATH>      cache file (default: <data-dir>/.spectra-viz/<stem>-k<K>.svz)\n\
                     \x20 --max-peaks <N>     peaks stored per spectrum (default: 32)\n\
                     \x20 --rebuild           rebuild the cache even if it exists\n\
                     \x20 --cache-only        build the cache and exit (no GUI)\n\
                     \x20 --check-smiles      parse every cached SMILES and report failures\n\
                     \x20 --render-smiles <S> render a SMILES depiction to --out (SVG or PNG)\n\
                     \x20 --render-molecule <N> render cached molecule N to --out (SVG or PNG)\n\
                     \x20 --out <PATH>        output path for depiction rendering (default: molecule.svg)\n\
                     \x20 -h, --help          show this help"
                );
                return None;
            }
            other if !other.starts_with('-') => data = Some(PathBuf::from(other)),
            other => {
                eprintln!("unknown argument: {other}");
                return None;
            }
        }
    }
    let data_path = data
        .or_else(|| {
            [
                PathBuf::from("data/train.parquet"),
                PathBuf::from("../data/train.parquet"),
                PathBuf::from("../../data/train.parquet"),
            ]
            .into_iter()
            .find(|path| path.exists())
        })
        .unwrap_or_else(|| PathBuf::from("data/train.parquet"));
    let cache_path = cache_path.unwrap_or_else(|| cache::default_cache_path(&data_path, max_peaks));
    Some(Options {
        data_path,
        cache_path,
        max_peaks,
        rebuild,
        cache_only,
        check_smiles,
        render_smiles,
        render_molecule,
        out_path,
        watchdog,
    })
}

fn sanitize_dyld_env() {
    use std::os::unix::process::CommandExt;
    let dyld_vars: Vec<String> = std::env::vars()
        .filter(|(key, _)| key.starts_with("DYLD_"))
        .map(|(key, _)| key)
        .collect();
    if dyld_vars.is_empty() {
        return;
    }
    eprintln!(
        "spectra-viz: re-executing with a clean dynamic-loader environment \
         ({} set by the shell/cargo can substitute incompatible Homebrew \
         libraries into system framework lookups)",
        dyld_vars.join(", ")
    );
    let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    let mut command = match std::env::current_exe() {
        Ok(exe) => std::process::Command::new(exe),
        Err(_) => {
            let argv0 = std::env::args_os().next().unwrap_or_default();
            std::process::Command::new(argv0)
        }
    };
    if args.len() > 1 {
        command.args(&args[1..]);
    }
    for key in &dyld_vars {
        command.env_remove(key);
    }
    command.env("SPECTRA_VIZ_DYLD_STRIPPED", "1");
    let error = command.exec();
    eprintln!(
        "spectra-viz: clean re-exec failed ({error}); continuing with the inherited environment"
    );
}

fn main() -> eframe::Result {
    sanitize_dyld_env();
    let Some(options) = parse_args() else {
        return Ok(());
    };
    if let Some(parent) = options.watchdog {
        run_watchdog(parent);
        return Ok(());
    }
    crash::install();
    install_panic_hook();
    crash::mark(&format!(
        "options: data={} cache={} max_peaks={} rebuild={}",
        options.data_path.display(),
        options.cache_path.display(),
        options.max_peaks,
        options.rebuild
    ));
    if options.cache_only {
        if !options.data_path.exists() {
            eprintln!(
                "training parquet not found: {}",
                options.data_path.display()
            );
            return Ok(());
        }
        if options.rebuild {
            let _ = std::fs::remove_file(&options.cache_path);
        }
        let mut last = -1.0f32;
        let mut send = |frac: f32, msg: String| {
            if frac - last >= 0.02 || frac >= 0.99 {
                println!("{:5.1}%  {msg}", frac * 100.0);
                last = frac;
            }
        };
        match cache::build_cache(
            &options.data_path,
            &options.cache_path,
            options.max_peaks,
            &mut send,
        ) {
            Ok(()) => {
                println!("cache written to {}", options.cache_path.display());
            }
            Err(err) => {
                eprintln!("cache build failed: {err:#}");
            }
        }
        return Ok(());
    }
    if options.render_smiles.is_some() || options.render_molecule.is_some() {
        let smiles = if let Some(smiles) = &options.render_smiles {
            smiles.clone()
        } else {
            match cache::load_cache(&options.cache_path) {
                Ok(catalog) => {
                    let index = options.render_molecule.unwrap_or(0);
                    match catalog.molecules.get(index) {
                        Some(molecule) => molecule.smiles.clone(),
                        None => {
                            eprintln!(
                                "molecule index {index} out of range ({} molecules)",
                                catalog.molecules.len()
                            );
                            return Ok(());
                        }
                    }
                }
                Err(err) => {
                    eprintln!("could not load cache: {err:#}");
                    return Ok(());
                }
            }
        };
        let molecule = match cosmolkit::Molecule::from_smiles(&smiles) {
            Ok(molecule) => molecule,
            Err(err) => {
                eprintln!("SMILES parse error: {err}");
                return Ok(());
            }
        };
        let png = options
            .out_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("png"))
            .unwrap_or(false);
        let result = if png {
            molecule
                .to_png(900, 650)
                .map_err(|err| format!("{err}"))
                .and_then(|bytes| {
                    std::fs::write(&options.out_path, bytes).map_err(|err| format!("{err}"))
                })
        } else {
            molecule
                .to_svg(900, 650)
                .map_err(|err| format!("{err}"))
                .and_then(|svg| {
                    std::fs::write(&options.out_path, svg.into_bytes())
                        .map_err(|err| format!("{err}"))
                })
        };
        match result {
            Ok(()) => println!(
                "wrote {} ({} atoms)",
                options.out_path.display(),
                molecule.atoms().len()
            ),
            Err(err) => eprintln!("could not write depiction: {err}"),
        }
        return Ok(());
    }
    if options.check_smiles {
        match cache::load_cache(&options.cache_path) {
            Ok(catalog) => {
                let started = std::time::Instant::now();
                let total = catalog.molecules.len();
                let step = (total / 20_000).max(1);
                let mut checked = 0usize;
                let mut parse_failures = 0usize;
                let mut render_checked = 0usize;
                let mut render_failures = 0usize;
                for (index, molecule) in catalog.molecules.iter().enumerate().step_by(step) {
                    checked += 1;
                    match cosmolkit::Molecule::from_smiles(&molecule.smiles) {
                        Ok(parsed) => {
                            if render_checked < 2_000 {
                                render_checked += 1;
                                if let Err(error) = parsed.to_png(600, 400) {
                                    render_failures += 1;
                                    if render_failures <= 10 {
                                        println!(
                                            "render {index}: {}  smiles={}  error={error}",
                                            molecule.formula, molecule.smiles
                                        );
                                    }
                                }
                            }
                        }
                        Err(error) => {
                            parse_failures += 1;
                            if parse_failures <= 20 {
                                println!(
                                    "molecule {index}: {}  smiles={}  error={error}",
                                    molecule.formula, molecule.smiles
                                );
                            }
                        }
                    }
                }
                println!(
                    "checked {checked} of {total} molecules, {parse_failures} parse failures, {render_failures}/{render_checked} render failures, {:?}",
                    started.elapsed()
                );
            }
            Err(err) => eprintln!("could not load cache: {err:#}"),
        }
        return Ok(());
    }
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 920.0])
            .with_min_inner_size([960.0, 620.0])
            .with_title("CASMI26 Spectra Viewer"),
        ..Default::default()
    };
    if let Ok(exe) = std::env::current_exe() {
        let _ = std::process::Command::new(exe)
            .arg("--watchdog")
            .arg(std::process::id().to_string())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
        crash::mark("watchdog spawned");
    }
    crash::mark("entering eframe::run_native");
    let result = eframe::run_native(
        "CASMI26 Spectra Viewer",
        native_options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, options)))),
    );
    crash::mark("run_native returned");
    result
}
