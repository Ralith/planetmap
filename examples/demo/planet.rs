use noise::{MultiFractal, NoiseFn};

use half::f16;
use planetmap::{chunk::{Face, Coords}, Chunk, ncollide::Terrain};

pub struct Planet {
    noise: noise::Fbm,
    radius: f32,
}

impl Planet {
    pub fn new() -> Self {
        Self {
            noise: noise::Fbm::new()
                .set_octaves(10)
                .set_frequency(1.0 / 64.0)
                .set_persistence(0.7),
            radius: 6371e3,
        }
    }

    pub fn generate_chunk(
        &self,
        chunk: &Chunk,
        height_resolution: u32,
        heights: &mut [f16],
        normal_resolution: u32,
        normals: &mut [[i8; 2]],
        color_resolution: u32,
        colors: &mut [[u8; 4]],
    ) {
        for (i, sample) in chunk.samples(height_resolution).enumerate() {
            heights[i] = f16::from_f32(self.height_at(&sample));
        }
        for (i, sample) in chunk.samples(normal_resolution).enumerate() {
            normals[i] = pack_normal(&self.normal_at(&sample));
        }
        for (i, sample) in chunk.samples(color_resolution).enumerate() {
            colors[i] = self.color_at(&sample);
        }
    }

    /// Radial heightmap function
    fn height_at(&self, dir: &na::Unit<na::Vector3<f32>>) -> f32 {
        let p = na::Point::from(
            na::convert::<_, na::Vector3<f64>>(dir.into_inner()) * self.radius as f64,
        );
        self.sample(p) as f32
    }

    fn normal_at(&self, dir: &na::Unit<na::Vector3<f32>>) -> na::Unit<na::Vector3<f32>> {
        let basis = Face::from_vector(dir).basis::<f64>();
        let dir = na::convert::<_, na::Vector3<f64>>(dir.into_inner());
        let perp = basis.matrix().index((.., 1));
        let x = dir.cross(&perp);
        let y = dir.cross(&x);
        let h = 1e-3;
        let x_off = x * h;
        let y_off = y * h;
        let center = na::Point::from(dir * self.radius as f64);
        let x_0 = self.sample(center - x_off);
        let x_1 = self.sample(center + x_off);
        let y_0 = self.sample(center - y_off);
        let y_1 = self.sample(center + y_off);
        let dx = (x_1 - x_0) / (2.0 * h);
        let dy = (y_1 - y_0) / (2.0 * h);
        na::Unit::new_normalize(na::convert(na::Vector3::new(-dx, -dy, 1.0)))
    }

    fn color_at(&self, dir: &na::Unit<na::Vector3<f32>>) -> [u8; 4] {
        let height = self.height_at(dir);
        blend(
            height,
            &[
                // deep ocean
                ([0, 0, 96, 255], [0, 0, 128, 255], -1000.0, -10.0),
                // beach
                ([0, 0, 128, 255], [192, 192, 128, 255], -10.0, 100.0),
                ([192, 192, 128, 255], [192, 192, 128, 255], 100.0, 200.0),
                ([192, 192, 128, 255], [160, 192, 80, 255], 200.0, 210.0),
                // grass, forest, deep forest
                ([160, 192, 80, 255], [160, 192, 80, 255], 210.0, 290.0),
                ([160, 192, 80, 255], [64, 192, 64, 255], 290.0, 300.0),
                ([64, 192, 64, 255], [0, 90, 0, 255], 300.0, 1290.0),
                // vegetation line
                ([0, 90, 0, 255], [160, 160, 160, 255], 1290.0, 1300.0),
                ([160, 160, 160, 255], [192, 192, 192, 255], 1300.0, 1990.0),
                // snow transition
                ([192, 192, 192, 255], [255, 255, 255, 255], 1990.0, 2000.0),
                ([255, 255, 255, 255], [255, 255, 255, 255], 2000.0, 200000.0),
            ],
        )
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }

    fn sample(&self, p: na::Point3<f64>) -> f64 {
        let point: [f64; 3] = (p.coords * 5e-5).into();
        let lat = (p.z / self.radius as f64).abs();
        let h = ((1.0 + self.noise.get(point)).powi(3) - 1.0) * 1500.0 + (lat - 0.3) * 3000.0;
        if h < self.min_height() as f64 {
            self.min_height() as f64
        } else {
            h.min(self.max_height() as f64)
        }
    }
}

fn blend(f: f32, ranges: &[([u8; 4], [u8; 4], f32, f32)]) -> [u8; 4] {
    for &(from, to, low, high) in ranges {
        if (f >= low) && (f < high) {
            let ff = (f - low) / (high - low);
            return [
                (from[0] as f32 + (to[0] as f32 - from[0] as f32) * ff) as u8,
                (from[1] as f32 + (to[1] as f32 - from[1] as f32) * ff) as u8,
                (from[2] as f32 + (to[2] as f32 - from[2] as f32) * ff) as u8,
                (from[3] as f32 + (to[3] as f32 - from[3] as f32) * ff) as u8,
            ];
        }
    }
    unreachable!("f = {}", f);
}

fn pack_normal(normal: &na::Unit<na::Vector3<f32>>) -> [i8; 2] {
    [(normal.x * 127.0) as i8, (normal.y * 127.0) as i8]
}

impl Terrain for Planet {
    fn max_height(&self) -> f32 {
        10_000.0
    }
    fn min_height(&self) -> f32 {
        0.0
    }

    fn sample(&self, resolution: u32, coords: &Coords, out: &mut [f32]) {
        for (i, sample) in coords.samples(2u32.pow(12), resolution).enumerate() {
            out[i] = self.height_at(&sample);
        }
    }
}
