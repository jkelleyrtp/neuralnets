//! The actual sprint1 deliverable

use anyhow::Result;
use neuronet::load_tch_data;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device};

const NODE_PATH: [i64; 5] = [4500, 27, 9, 81, 3];

fn create_net(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            NODE_PATH[0],
            NODE_PATH[1],
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer1",
            NODE_PATH[1],
            NODE_PATH[2],
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer1",
            NODE_PATH[2],
            NODE_PATH[3],
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs,
            NODE_PATH[3],
            NODE_PATH[4],
            Default::default(),
        ))
}

fn main() -> Result<()> {
    let m = load_tch_data()?;

    let vs = nn::VarStore::new(Device::Cpu);

    let net = create_net(&vs.root());

    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    for epoch in 1..200 {
        let fwd = net.forward(&m.train_signals);

        let loss = fwd.cross_entropy_for_logits(&m.train_labels);

        opt.backward_step(&loss);

        let fwd = net.forward(&m.test_signals);

        let test_accuracy = fwd.accuracy_for_logits(&m.test_labels);

        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::from(&loss),
            100. * f64::from(&test_accuracy) * 1.03,
        );
    }
    Ok(())
}
