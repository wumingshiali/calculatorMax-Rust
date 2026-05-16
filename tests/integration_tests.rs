//! Integration tests for the calculator

#[cfg(test)]
mod tests {
    use calculator_max::calculator::{math_functions, Evaluator};

    #[test]
    fn test_basic_arithmetic() {
        let mut evaluator = Evaluator::new();

        assert_eq!(evaluator.evaluate("2 + 3").unwrap(), 5.0);
        assert_eq!(evaluator.evaluate("10 - 4").unwrap(), 6.0);
        assert_eq!(evaluator.evaluate("3 * 4").unwrap(), 12.0);
        assert_eq!(evaluator.evaluate("15 / 3").unwrap(), 5.0);
    }

    #[test]
    fn test_math_functions() {
        assert_eq!(math_functions::triangle_area(10.0, 5.0), 25.0);
        assert_eq!(math_functions::rectangle_area(4.0, 6.0), 24.0);
        assert_eq!(math_functions::circle_area(1.0), std::f64::consts::PI);
    }

    #[test]
    fn test_pythagorean_theorem() {
        let result = math_functions::pythagorean_theorem(3.0, 4.0);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_math_constants() {
        let mut evaluator = Evaluator::new();

        assert!((evaluator.evaluate("pi").unwrap() - std::f64::consts::PI).abs() < 1e-10);
        assert!((evaluator.evaluate("e").unwrap() - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_trigonometric_functions() {
        let mut evaluator = Evaluator::new();

        assert!((evaluator.evaluate("sin(0)").unwrap() - 0.0).abs() < 1e-10);
        assert!((evaluator.evaluate("cos(0)").unwrap() - 1.0).abs() < 1e-10);
        assert!((evaluator.evaluate("sqrt(4)").unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_functions() {
        let mut evaluator = Evaluator::new();

        assert_eq!(evaluator.evaluate("s_tri(10, 5)").unwrap(), 25.0);
        assert_eq!(evaluator.evaluate("s_rect(4, 6)").unwrap(), 24.0);
    }

    #[test]
    fn test_power_operator() {
        let mut evaluator = Evaluator::new();

        assert_eq!(evaluator.evaluate("2 ** 3").unwrap(), 8.0);
        assert_eq!(evaluator.evaluate("2^3").unwrap(), 8.0);
        assert_eq!(evaluator.evaluate("pow(2, 3)").unwrap(), 8.0);
    }

    #[test]
    fn test_floor_division() {
        let mut evaluator = Evaluator::new();

        assert_eq!(evaluator.evaluate("7 // 2").unwrap(), 3.0);
        assert_eq!(evaluator.evaluate("10 // 3").unwrap(), 3.0);
        assert_eq!(evaluator.evaluate("15 // 4").unwrap(), 3.0);
    }

    #[test]
    fn test_comparison_operators() {
        let mut evaluator = Evaluator::new();

        assert_eq!(evaluator.evaluate("5 == 5").unwrap(), 1.0);
        assert_eq!(evaluator.evaluate("5 == 3").unwrap(), 0.0);
        assert_eq!(evaluator.evaluate("5 > 3").unwrap(), 1.0);
        assert_eq!(evaluator.evaluate("3 > 5").unwrap(), 0.0);
        assert_eq!(evaluator.evaluate("3 < 5").unwrap(), 1.0);
        assert_eq!(evaluator.evaluate("5 >= 5").unwrap(), 1.0);
        assert_eq!(evaluator.evaluate("5 <= 5").unwrap(), 1.0);
    }

    #[test]
    fn test_logical_operators() {
        let mut evaluator = Evaluator::new();

        assert_eq!(evaluator.evaluate("1 or 0").unwrap(), 1.0);
        assert_eq!(evaluator.evaluate("1 and 1").unwrap(), 1.0);
        assert_eq!(evaluator.evaluate("not 0").unwrap(), 1.0);
    }

    #[test]
    fn test_new_functions() {
        let mut evaluator = Evaluator::new();

        // atan2
        assert!((evaluator.evaluate("atan2(1, 1)").unwrap() - std::f64::consts::FRAC_PI_4).abs() < 1e-10);

        // isinf and isnan
        assert_eq!(evaluator.evaluate("isinf(1.0/0.0)").unwrap(), 1.0);
        assert_eq!(evaluator.evaluate("isinf(1.0)").unwrap(), 0.0);
        assert_eq!(evaluator.evaluate("isnan(0.0/0.0)").unwrap(), 1.0);
        assert_eq!(evaluator.evaluate("isnan(1.0)").unwrap(), 0.0);

        // isclose
        assert_eq!(evaluator.evaluate("isclose(1.0, 1.0000000001)").unwrap(), 1.0);

        // gcd and lcm
        assert_eq!(evaluator.evaluate("gcd(12, 8)").unwrap(), 4.0);
        assert_eq!(evaluator.evaluate("lcm(4, 6)").unwrap(), 12.0);

        // modf (fractional part)
        assert!((evaluator.evaluate("modf(3.7)").unwrap() - 0.7).abs() < 1e-10);

        // trapezoid area
        assert_eq!(evaluator.evaluate("s_tra(3, 5, 4)").unwrap(), 16.0);

        // Heron's formula
        let result = evaluator.evaluate("hsf_s_tri(3, 4, 5)").unwrap();
        assert!((result - 6.0).abs() < 1e-10);

        // Pythagorean theorem
        assert_eq!(evaluator.evaluate("pt(3, 4)").unwrap(), 5.0);

        // randint and uniform
        let rand_int = evaluator.evaluate("randint(1, 10)").unwrap();
        assert!(rand_int >= 1.0 && rand_int <= 10.0);

        let uniform_val = evaluator.evaluate("uniform(0, 1)").unwrap();
        assert!(uniform_val >= 0.0 && uniform_val < 1.0);
    }
}
