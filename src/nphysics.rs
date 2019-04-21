use nphysics3d::force_generator::ForceGenerator;

#[derive(Debug, Clone)]
pub struct GravityWell<N: na::RealField> {
    factor: N,
    position: na::Point3<N>,
}

const G: f64 = 6.67408e-11;

impl<N: na::RealField> GravityWell<N> {
    pub fn new(mass: N, position: na::Point3<N>) -> Self {
        Self {
            factor: na::convert::<_, N>(G) * mass,
            position,
        }
    }

    pub fn set_mass(&mut self, mass: N) {
        self.factor = na::convert::<_, N>(G) * mass;
    }

    pub fn mass(&self) -> N {
        self.factor / na::convert::<_, N>(G)
    }

    pub fn set_position(&mut self, position: na::Point3<N>) {
        self.position = position;
    }

    pub fn position(&self) -> &na::Point3<N> {
        &self.position
    }
}

impl<N: na::RealField> ForceGenerator<N> for GravityWell<N> {
    fn apply(
        &mut self,
        _params: &nphysics3d::solver::IntegrationParameters<N>,
        bodies: &mut nphysics3d::object::BodySet<N>,
    ) -> bool {
        for body in bodies.bodies_mut() {
            let part = body.part(0).unwrap(); // TODO: Multibodies
            let r_2 = na::distance_squared(&self.position, &part.center_of_mass());
            if r_2.abs() < na::convert(1e-3) {
                continue;
            }
            let magnitude = self.factor / r_2;
            let direction = (self.position - part.center_of_mass()) / r_2.sqrt();
            body.apply_force(
                0,
                &nphysics3d::math::Force::new(direction * magnitude, na::zero()),
                nphysics3d::algebra::ForceType::AccelerationChange,
                false,
            );
        }
        true
    }
}
