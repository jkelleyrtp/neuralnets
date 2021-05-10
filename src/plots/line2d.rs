use plotly::layout::Axis;

use super::ui::*;

#[derive(Debug)]
struct Line2DTrace {
    x: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Debug, Default)]
pub struct Line2D {
    traces: Vec<Line2DTrace>,
    labels: Labels,
    grid: Grid,
}

impl Line2D {
    pub fn empty() -> Self {
        Default::default()
    }

    /// Hey there
    pub fn new(x: &impl AsRef<[f64]>, y: &impl AsRef<[f64]>) -> Self {
        Self {
            traces: vec![Line2DTrace {
                x: x.as_ref().to_vec(),
                y: y.as_ref().to_vec(),
            }],
            labels: Default::default(),
            grid: Default::default(),
        }
    }

    pub fn add_trace(&mut self, x: &impl AsRef<[f64]>, y: &impl AsRef<[f64]>) -> &mut Self {
        self.traces.push(Line2DTrace {
            x: x.as_ref().to_vec(),
            y: y.as_ref().to_vec(),
        });
        self
    }

    pub fn plot(self) {
        use plotly::common::*;
        use plotly::*;

        let Self {
            traces,
            labels,
            grid,
            ..
        } = self;

        let Labels {
            title,
            xlabel,
            ylabel,
            ..
        } = labels;

        let Grid {
            xmax,
            xmin,
            ymax,
            ymin,
        } = grid;

        let layout = Layout::new()
            .title(title.unwrap_or("".to_string()).as_str().into())
            .x_axis({
                let ax = Axis::new().title(xlabel.unwrap_or("x".into()).as_str().into());
                if let (Some(xmin), Some(xmax)) = (xmin, xmax) {
                    ax.range(vec![xmin as f64, xmax as f64])
                } else {
                    ax
                }
            })
            .y_axis({
                let ax = Axis::new().title(ylabel.unwrap_or("y".into()).as_str().into());
                if let (Some(min), Some(max)) = (ymin, ymax) {
                    ax.range(vec![min as f64, max as f64])
                } else {
                    ax
                }
            });

        let mut plot = Plot::new();
        plot.set_layout(layout);

        for Line2DTrace { x, y } in traces {
            let scatter = Scatter::new(x.iter().cloned(), y.iter().cloned())
                .mode(Mode::LinesMarkers)
                .web_gl_mode(true);
            plot.add_trace(scatter);
        }

        plot.show();
    }
}

mod impls {
    use super::*;

    impl Labels2d for Line2D {
        fn labels_mut(&mut self) -> &mut Labels {
            &mut self.labels
        }
    }
    impl Grid2D for Line2D {
        fn grid_mut(&mut self) -> &mut Grid {
            &mut self.grid
        }
    }
}
