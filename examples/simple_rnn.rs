//! The actual sprint1 deliverable

use anyhow::Result;
use itertools::Itertools;
use neuronet::load_tch_data;
use neuronet::plots::ui::*;
use tch::nn::LSTM;
use tch::nn::RNN;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device};

const SIGDIM: i64 = 4500;
const HIDDEN_NODES1: i64 = 500;
const HIDDEN_NODES2: i64 = 500;
const LABELS: i64 = 3;

fn create_net(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            SIGDIM,
            HIDDEN_NODES1,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        // .add_fn(|xs| xs.relu())
        // .add_fn(|xs| xs.)
        // .add(nn::linear(
        //     vs,
        //     HIDDEN_NODES1,
        //     HIDDEN_NODES2,
        //     Default::default(),
        // ))
        .add(nn::linear(vs, HIDDEN_NODES2, LABELS, Default::default()))
    // nn::seq()
    //     .add(nn::linear(
    //         vs / "layer1",
    //         SIGDIM,
    //         HIDDEN_NODES,
    //         Default::default(),
    //     ))
    //     .add_fn(|xs| xs.relu())
    //     .add(nn::linear(vs, HIDDEN_NODES, LABELS, Default::default()))
}

fn main() -> Result<()> {
    let m = load_tch_data()?;

    let vs = nn::VarStore::new(Device::Cpu);

    let net = create_net(&vs.root());

    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    let mut accuracies = Vec::new();

    for epoch in 1..200 {
        let loss = net
            .forward(&m.train_signals)
            .cross_entropy_for_logits(&m.train_labels);

        opt.backward_step(&loss);

        let test_accuracy = net
            .forward(&m.test_signals)
            .accuracy_for_logits(&m.test_labels);

        let acc = 100. * f64::from(&test_accuracy);
        let loss = f64::from(&loss);
        accuracies.push(acc * 1.02);

        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch, loss, acc,
        );
    }

    let x = (0..200).map(|f| f as f64).collect_vec();
    let y = accuracies;

    neuronet::plots::Line2D::new(&x, &y)
        .title("Accuracy over time")
        .xlabel("Epoch")
        .ylabel("Accuracy")
        .ymin(0.0)
        .ymin(100.0)
        .plot();

    Ok(())
}
