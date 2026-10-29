use eframe::egui::{ColorImage, Context, TextureHandle, TextureOptions};
use std::collections::{HashMap, VecDeque};

pub struct StructureTexture {
    pub texture: Option<TextureHandle>,
    pub error: Option<String>,
}

pub struct StructureCache {
    map: HashMap<u32, StructureTexture>,
    order: VecDeque<u32>,
    generation: u64,
    width: u32,
    height: u32,
}

impl StructureCache {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            generation: 0,
            width: 640,
            height: 400,
        }
    }

    pub fn get(&mut self, ctx: &Context, index: u32, smiles: &str) -> &StructureTexture {
        if !self.map.contains_key(&index) {
            if self.order.len() >= 64 {
                for _ in 0..32 {
                    if let Some(key) = self.order.pop_front() {
                        self.map.remove(&key);
                    }
                }
            }
            let entry = render(
                ctx,
                index,
                smiles,
                &mut self.generation,
                self.width,
                self.height,
            );
            self.map.insert(index, entry);
            self.order.push_back(index);
        }
        self.map.get(&index).unwrap()
    }
}

fn render(
    ctx: &Context,
    index: u32,
    smiles: &str,
    generation: &mut u64,
    width: u32,
    height: u32,
) -> StructureTexture {
    let molecule = match cosmolkit::Molecule::from_smiles(smiles) {
        Ok(molecule) => molecule,
        Err(err) => {
            return StructureTexture {
                texture: None,
                error: Some(format!("{err}")),
            }
        }
    };
    let png = match molecule.to_png(width, height) {
        Ok(png) => png,
        Err(err) => {
            return StructureTexture {
                texture: None,
                error: Some(format!("{err}")),
            }
        }
    };
    let image = match image::load_from_memory_with_format(&png, image::ImageFormat::Png) {
        Ok(image) => image.to_rgba8(),
        Err(err) => {
            return StructureTexture {
                texture: None,
                error: Some(format!("PNG decode failed: {err}")),
            }
        }
    };
    let size = [image.width() as usize, image.height() as usize];
    let pixels = ColorImage::from_rgba_unmultiplied(size, image.as_raw());
    *generation += 1;
    let texture = ctx.load_texture(
        format!("structure-{index}-{generation}"),
        pixels,
        TextureOptions::LINEAR,
    );
    StructureTexture {
        texture: Some(texture),
        error: None,
    }
}
