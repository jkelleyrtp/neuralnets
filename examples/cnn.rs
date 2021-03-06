//! The actual sprint1 deliverable

use anyhow::Result;
use neuronet::load_tch_data;
use tch::{
    nn,
    nn::Module,
    nn::{ConvConfig, OptimizerConfig},
    Device,
};

const SIGDIM: i64 = 4500;
const HIDDEN_NODES: i64 = 100;
const LABELS: i64 = 3;
const LEARNING_RATE: f64 = 1e-3;

fn create_net(vs: &nn::Path) -> impl Module {
    nn::seq()
        // .add(nn::conv(vs, SIGDIM, HIDDEN_NODES, 1, ..Default::default()))
        .add(nn::conv1d(
            vs,
            SIGDIM,
            HIDDEN_NODES,
            1,
            ConvConfig {
                ..Default::default()
            },
        ))
        // .add(nn::linear(
        //     vs / "layer1",
        //     SIGDIM,
        //     HIDDEN_NODES,
        //     Default::default(),
        // ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, HIDDEN_NODES, LABELS, Default::default()))
}

fn main() -> Result<()> {
    let m = load_tch_data()?;

    let vs = nn::VarStore::new(Device::Cpu);

    let net = create_net(&vs.root());

    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE)?;

    for epoch in 1..200 {
        let loss = net
            .forward(&m.train_signals)
            .cross_entropy_for_logits(&m.train_labels);

        opt.backward_step(&loss);

        let test_accuracy = net
            .forward(&m.test_signals)
            .accuracy_for_logits(&m.test_labels);

        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::from(&loss),
            100. * f64::from(&test_accuracy),
        );
    }
    Ok(())
}
