use ndarray::{Array, Ix1, Ix2};
use plotly::common::Mode;
use plotly::{Plot, Scatter};
// use plotly::{ArrayTraces, Plot, Scatter};

pub struct Scatter2D {}

impl Scatter2D {}

pub fn scatter2d(x: &impl AsRef<[f64]>, y: &impl AsRef<[f64]>) -> anyhow::Result<()> {
    // pub fn scatter2d(xy: impl AsRef<[(f64, f64)]>) -> anyhow::Result<()> {
    // let n: usize = 11;
    // let t: Array<f64, Ix1> = Array::range(0., 10., 10. / n as f64);
    // let mut ys: Array<f64, Ix2> = Array::zeros((11, 11));
    // let mut count = 0.;
    // for mut row in ys.gencolumns_mut() {
    //     for index in 0..row.len() {
    //         row[index] = count + (index as f64).powf(2.);
    //     }
    //     count += 1.;
    // }

    // .to_traces(t, ys, ArrayTraces::OverColumns);
    // .to_traces(t, ys, ArrayTraces::OverColumns);

    // plot.add_traces(traces);

    let _x = x.as_ref();
    let _y = y.as_ref();

    let traces = Scatter::new(_x.iter().cloned(), _y.iter().cloned()).mode(Mode::LinesMarkers);
    let mut plot = Plot::new();
    plot.add_trace(traces);
    plot.show();

    Ok(())
}
#[test]
fn entry_point() {
    scatter2d(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap()
    // scatter2d([(1.0, 2.0)]).unwrap()
}
