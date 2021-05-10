//! LTTSM

use anyhow::Result;
use neuronet::load_tch_data;
use tch::nn::RNN;
use tch::nn::{RNNConfig, LSTM};
use tch::{nn, nn::Module, nn::OptimizerConfig, Device};

const SIGDIM: i64 = 4500;
const HIDDEN_NODES: i64 = 30;
const LABELS: i64 = 3;
const LEARNING_RATE: f64 = 1e-3;

fn main() -> Result<()> {
    let dataset = load_tch_data()?;

    let store = nn::VarStore::new(Device::Cpu);

    let net = nn::seq()
        .add(nn::linear(
            &store.root(),
            SIGDIM,
            HIDDEN_NODES,
            Default::default(),
        ))
        // .add_fn(|xs| xs.relu())
        .add(nn::linear(
            &store.root(),
            HIDDEN_NODES,
            LABELS,
            Default::default(),
        ));

    let mut opt = nn::Adam::default().build(&store, LEARNING_RATE)?;

    for epoch in 1..200 {
        let loss = net
            .forward(&dataset.train_signals)
            .cross_entropy_for_logits(&dataset.train_labels);

        opt.backward_step(&loss);

        let test_accuracy = net
            .forward(&dataset.test_signals)
            .accuracy_for_logits(&dataset.test_labels);

        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::from(&loss),
            100. * f64::from(&test_accuracy),
        );
    }
    Ok(())
}
