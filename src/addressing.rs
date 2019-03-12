pub fn discretize(resolution: usize, texcoords: &na::Point2<f32>) -> (usize, usize) {
    let texcoords = texcoords * (resolution - 1) as f32 + na::Vector2::new(0.5, 0.5);
    (texcoords.x as usize, texcoords.y as usize)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn discretize_sanity() {
        assert_eq!(discretize(100, &na::Point2::new(1.0, 1.0)), (99, 99));
        assert_eq!(discretize(100, &na::Point2::new(0.0, 0.0)), (0, 0));
        assert_eq!(discretize(100, &na::Point2::new(0.996, 0.996)), (99, 99));
        assert_eq!(discretize(100, &na::Point2::new(0.004, 0.004)), (0, 0));
        assert_eq!(discretize(100, &na::Point2::new(0.006, 0.006)), (1, 1));
        assert_eq!(discretize(100, &na::Point2::new(0.994, 0.994)), (98, 98));

        assert_eq!(discretize(2, &na::Point2::new(0.4, 0.4)), (0, 0));
        assert_eq!(discretize(2, &na::Point2::new(0.6, 0.6)), (1, 1));
    }
}
