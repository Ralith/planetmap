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
        self.noise.get(point) * 4000.0
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }
}
