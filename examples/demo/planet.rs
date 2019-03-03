use noise::NoiseFn;

pub struct Planet {
    noise: noise::Fbm,
    radius: f64,
}

impl Planet {
    pub fn new() -> Self {
        Self {
            noise: Default::default(),
            radius: 6371e3,
        }
    }

    /// Radial heightmap function
    pub fn height_at(&self, dir: &na::Unit<na::Vector3<f64>>) -> f64 {
        let point: [f64; 3] = (dir.into_inner() * self.radius * 5e-5).into();
        let h = ((1.0 + self.noise.get(point)).powi(3) - 1.0) * 1000.0;
        if h < 0.0 { 0.0 } else { h }
    }

    pub fn normal_at(&self, _: &na::Unit<na::Vector3<f64>>) -> na::Unit<na::Vector3<f32>> {
        na::Vector3::z_axis()
    }

    pub fn color_at(&self, dir: &na::Unit<na::Vector3<f64>>) -> [u8; 4] {
        let height = self.height_at(dir);
        if height == 0.0 {
            [0, 0, 128, 255]
        } else if height < 2000.0 {
            [0, 128, 0, 255]
        } else {
            [255, 255, 255, 255]
        }
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }
}
