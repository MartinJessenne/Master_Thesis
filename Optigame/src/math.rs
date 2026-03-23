use ndarray::{Array1, Array2};

pub type V = Array1<f64>;
pub type M = Array2<f64>;

// Defining the Simplex element custom type, to leverage the 
// type system to ensure consistent logic across all code. 
#[derive(Clone, Debug)]
pub struct S {
    inner: V,
}

impl S {
    /// Creates a simplex element, keeping the method private ensures that the simplex element
    /// has the right properties, leveraging the type system to validate code logic 
    pub fn from_projected(v: V) -> Self {
        Self { inner: v.simplex_projection(1.0)}
    }

    pub fn build(v: V) -> Result<Self, &'static str> {
        let sum: f64 = v.iter().sum();

        let all_positive = v.iter().all(|&x| x >= 0.0);

        if (sum - 1.0).abs() < 1e-6 && all_positive {
            return Ok(Self { inner: v});
        } else {
            return Err("v does not belong on the simplex.");
        }
    }

    // Retrieve the underlying array for math operations
    pub fn as_array(&self) -> &V {
        &self.inner
    }

    pub fn into_inner(self) -> V {
        self.inner
    }
}

// Implementing Deref allows you to use `&SimplexElement` as if it were `&Array1<f64>`
impl std::ops::Deref for S {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::Sub<&V> for &S {
    type Output = V;

    fn sub(self, rhs: &V) -> Self::Output {
        &self.inner - rhs // Deref self to &V, perform array subtraction, return V
    }
}

impl std::ops::Add<&V> for &S {
    type Output = V;

    fn add(self, rhs: &V) -> Self::Output {
        &self.inner + rhs // Deref self to &V, perform array addition, return V
    }
}


pub trait Projectable {
    fn simplex_projection(&self, z: f64) -> V;
    fn simplex_projection_inplace(&mut self, z:f64);
}

impl Projectable for V {
    fn simplex_projection(&self, z: f64) -> V {
        let mut u = self.clone();

        u.as_slice_mut()
        .expect("Array1 should be contiguous in memory")
        .sort_by(|a,b | b.partial_cmp(a).expect("NaN encountered"));

        let theta = u.iter()
        .scan(0.0,|cumsum, x| {
            *cumsum += x;
            Some((*cumsum - z, x))
        })
        .enumerate()
        .filter_map(|(i, (cssv_i, u_i))|{
            let idx = (i + 1) as f64;

            if u_i - cssv_i / idx > 0.0 {
                Some(cssv_i / idx)
            }
            else {None}
        })
        .last().expect("Issue computing theta");

        return self.iter().map(|x| {
            let x = x - theta;
            x.max(0.0)
        }).collect::<V>();
    }
    
    fn simplex_projection_inplace(&mut self, z:f64) {
        let projected = self.simplex_projection(z);
        *self = projected;
    }
}

// define helper function 
pub fn projection_simplex(v: &V, z: f64) -> V {
    // need to sort by descending order the vector 

    let mut u : V = v.clone();

    u.as_slice_mut()
     .expect("Vector should be contiguous in memory")
     .sort_by(|a, b| b.partial_cmp(a).expect("NaN encountered"));

    let theta = u.into_iter()
                             .scan(0.0, |cumsum,x | {
                                *cumsum += x;
                                Some((*cumsum - z, x)) // TODO: check if this method return the expect result
                             })
                             .enumerate()
                             .filter_map(|(i, (cssv_i, u_i))| {
                                let idx = (i+1) as f64;
                                if u_i - cssv_i / idx > 0.0 {
                                    Some(cssv_i / idx)
                                }
                                else {
                                    None
                                }
                             })
                             .last()
                             .expect("Issue while projecting on the simplex");

    let v :V = v.iter()
    .map(|x| {
        let x = x - theta;
        x.max(0.0)
        })
    .collect();
    
    return v
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_simplex_projection_zeros() {
        let v_zeros: V = array![0.0, 0.0, 0.0, 0.0];
        // Projecting a zero vector of dimension 4 onto the simplex should result in [0.25, 0.25, 0.25, 0.25]
        let splx = S::from_projected(v_zeros);
        
        let sum: f64 = splx.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum after projection of zeros should be 1.0, was {}", sum);
        
        for &val in splx.iter() {
            assert!((val - 0.25).abs() < 1e-6, "Expected equal distribution 0.25, got {}", val);
        }
    }

    #[test]
    fn test_simplex_projection_standard() {
        let v: V = array![0.5, 0.7, 0.65, 0.8];
        let splx = S::from_projected(v);
        
        let sum: f64 = splx.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum after projection was {}", sum);
        
        for &val in splx.iter() {
            assert!(val >= 0.0, "Found negative value in projection");
        }
    }

    #[test]
    fn test_splx_build_success() {
        // Correct distribution that sums to 1.0 and is positive
        let v: V = array![0.2, 0.3, 0.5];
        let splx = S::build(v);
        assert!(splx.is_ok(), "Build should succeed for valid simplex elements");
    }

    #[test]
    fn test_splx_build_fail_negative() {
        // Has a negative element (even if sum is 1.0)
        let v: V = array![-0.5, 1.0, 0.5];
        let splx = S::build(v);
        assert!(splx.is_err(), "Build should fail if there is a negative element");
    }

    #[test]
    fn test_splx_build_fail_sum() {
        // Sums to 1.5 instead of 1.0
        let v: V = array![0.5, 0.5, 0.5];
        let splx = S::build(v);
        assert!(splx.is_err(), "Build should fail if the sum is not 1.0");
    }
}


