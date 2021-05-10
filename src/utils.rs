use serde::Serialize;

pub trait ToFloatStream: Sized + Copy + Serialize {
    fn to_stream(self) -> f32;
}

impl ToFloatStream for f32 {
    fn to_stream(self) -> f32 {
        self
    }
}

impl ToFloatStream for &f32 {
    fn to_stream(self) -> f32 {
        *self
    }
}

impl ToFloatStream for f64 {
    fn to_stream(self) -> f32 {
        self as f32
    }
}

impl ToFloatStream for &f64 {
    fn to_stream(self) -> f32 {
        *self as f32
    }
}
