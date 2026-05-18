use ndarray::{ArrayBase, Data, Dimension};

pub trait Ordf64Ext {
    fn max_f64(&self) -> f64;
    fn min_f64(&self) -> f64;
}

impl<S, D> Ordf64Ext for ArrayBase<S, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    /// Given any Array of f64 values, finds the maximum value among all coefficients, ignoring NaNs values.
    fn max_f64(&self) -> f64 {
        self.iter().fold(f64::NEG_INFINITY, |acc, &val| acc.max(val))
    }

    /// Given any Array of f64 values, finds the minimum value among all coefficients, ignoring NaN values.
    fn min_f64(&self) -> f64 {
        self.iter().fold(f64::INFINITY, |acc, &val| acc.min(val))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2, ArrayView1, ArrayView2};

    #[test]
    fn test_standard_1d_array() {
        let a = array![1.0, 5.0, 3.0, -2.0];
        assert_eq!(a.max_f64(), 5.0);
        assert_eq!(a.min_f64(), -2.0);
    }

    #[test]
    fn test_standard_2d_array() {
        let a = array![[1.0, 5.0], [3.0, -2.0]];
        assert_eq!(a.max_f64(), 5.0);
        assert_eq!(a.min_f64(), -2.0);
    }

    #[test]
    fn test_array_views() {
        let a = array![1.0, 2.0, 3.0, 4.0];
        let view: ArrayView1<f64> = a.view();
        assert_eq!(view.max_f64(), 4.0);
        assert_eq!(view.min_f64(), 1.0);

        let a2 = array![[1.0, 2.0], [3.0, 4.0]];
        let view2: ArrayView2<f64> = a2.view();
        assert_eq!(view2.max_f64(), 4.0);
        assert_eq!(view2.min_f64(), 1.0);
    }

    #[test]
    fn test_nan_handling() {
        // IEEE 754 specifies that max(NaN, x) is x for valid f64 x.
        let a = array![1.0, f64::NAN, 3.0];
        assert_eq!(a.max_f64(), 3.0);
        assert_eq!(a.min_f64(), 1.0);

        // Rust's f64::max(NEG_INFINITY, NAN) returns NEG_INFINITY.
        let all_nan = array![f64::NAN, f64::NAN];
        assert_eq!(all_nan.max_f64(), f64::NEG_INFINITY);
        assert_eq!(all_nan.min_f64(), f64::INFINITY);
    }

    #[test]
    fn test_infinity_handling() {
        let a = array![1.0, f64::INFINITY, 3.0, f64::NEG_INFINITY];
        assert_eq!(a.max_f64(), f64::INFINITY);
        assert_eq!(a.min_f64(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_empty_array() {
        let empty_1d: Array1<f64> = array![];
        assert_eq!(empty_1d.max_f64(), f64::NEG_INFINITY);
        assert_eq!(empty_1d.min_f64(), f64::INFINITY);

        let empty_2d: Array2<f64> = Array2::zeros((0, 0));
        assert_eq!(empty_2d.max_f64(), f64::NEG_INFINITY);
        assert_eq!(empty_2d.min_f64(), f64::INFINITY);
    }
}