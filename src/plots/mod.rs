mod line2d;
pub use line2d::*;

mod roc;
pub use roc::*;

mod scatter2d;
pub use scatter2d::*;

mod training;
pub use training::*;


pub mod ui {

    #[derive(Debug, Default)]
    pub struct Labels {
        pub title: Option<String>,
        pub xlabel: Option<String>,
        pub ylabel: Option<String>,
    }
    #[rustfmt::skip] 
    pub trait Labels2d: Sized {
        fn labels_mut(&mut self) -> &mut Labels;
        
        /// Set the title of the plot
        fn title(mut self, title: &str) -> Self { self.labels_mut().title = Some(title.to_string()); self }

        /// Set the x-axis label
        fn xlabel(mut self, xlabel: &str) -> Self { self.labels_mut().xlabel = Some(xlabel.to_string()); self }

        /// Set the y-axis label
        fn ylabel(mut self, ylabel: &str) -> Self { self.labels_mut().ylabel = Some(ylabel.to_string()); self }
    }

    /// GRIDS
    #[derive(Debug, Default)]
    pub struct Grid {
        pub xmax: Option<f32>,
        pub xmin: Option<f32>,
        pub ymax: Option<f32>,
        pub ymin: Option<f32>,
    }
    #[rustfmt::skip] 
    pub trait Grid2D: Sized {
        fn grid_mut(&mut self) -> &mut Grid;
        fn xmax(mut self, xmax: f32) -> Self { self.grid_mut().xmax = Some(xmax); self }
        fn xmin(mut self, xmin: f32) -> Self { self.grid_mut().xmin = Some(xmin); self }
        fn ymax(mut self, ymax: f32) -> Self { self.grid_mut().ymax = Some(ymax); self }
        fn ymin(mut self, ymin: f32) -> Self { self.grid_mut().ymin = Some(ymin); self }
    }
}
