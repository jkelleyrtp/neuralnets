use anyhow::Result;
use tch::{data, kind, no_grad, vision, Kind, Tensor};

const IMAGE_DIM: i64 = 4500;
const LABELS: i64 = 3;

fn main() -> Result<()> {
    let dataset = neuronet::load_tch_data()?;

    println!("train-images: {:?}", dataset.train_signals.size());
    println!("train-labels: {:?}", dataset.train_labels.size());
    println!("test-images: {:?}", dataset.test_signals.size());
    println!("test-labels: {:?}", dataset.test_labels.size());

    let mut ws = Tensor::zeros(&[IMAGE_DIM, LABELS], kind::FLOAT_CPU).set_requires_grad(true);
    let mut bs = Tensor::zeros(&[LABELS], kind::FLOAT_CPU).set_requires_grad(true);
    for epoch in 1..500 {
        let logits = dataset.train_signals.mm(&ws) + &bs;
        let loss = logits
            .log_softmax(-1, Kind::Float)
            .nll_loss(&dataset.train_labels);
        ws.zero_grad();
        bs.zero_grad();
        loss.backward();
        no_grad(|| {
            ws += ws.grad() * (-1);
            bs += bs.grad() * (-1);
        });

        let test_logits = dataset.test_signals.mm(&ws) + &bs;
        let test_accuracy = test_logits
            .argmax(Some(-1), false)
            .eq1(&dataset.test_labels)
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            loss.double_value(&[]),
            100. * test_accuracy
        );
    }
    Ok(())
}
