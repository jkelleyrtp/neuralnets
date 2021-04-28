use anyhow::{anyhow, Context, Error, Result};
use matfile::{MatFile, NumericData};
use tch::{Kind, Tensor};

const NUM_SAMPLES: usize = 1500;
const NUM_TRIALS: usize = 30;
const NUM_CHANS: usize = 3;

pub type DataChannel = [f64; NUM_SAMPLES];
pub type TrialData = [DataChannel; NUM_CHANS];
pub type ExpData = Box<[TrialData; NUM_TRIALS]>;

pub struct Experiment {
    pub training: Box<[f64]>,
    pub training_labels: Vec<i64>,

    pub testing: Box<[f64]>,
    pub testing_labels: Vec<i64>,
}
pub struct RawExperiment {
    pub data: ExpData,
    pub labels: [RpsLabel; NUM_TRIALS],
}

#[derive(Debug, Clone, Copy)]
pub enum RpsLabel {
    rock = 0,
    paper = 1,
    scissors = 2,
}

pub fn load_trial_data(input: &[u8]) -> Result<ExpData> {
    // Load the mat file
    let mat_file = crate::MatFile::parse(input.as_ref())?;

    // Grab the first array from the file and then cast it to the right data type
    let mut data = match mat_file.arrays().into_iter().next().unwrap().data() {
        NumericData::Double { real, .. } => Ok(real),
        _ => Err(anyhow!("Not the right data type")),
    }?
    .into_iter();

    // Create an output for the data in a fixed buffer type
    let mut data_buffer = Box::new([[[0.0; NUM_SAMPLES]; NUM_CHANS]; NUM_TRIALS]);

    // Write the matlab format into our rusty format
    for trial in data_buffer.iter_mut() {
        for x in 0..NUM_SAMPLES {
            trial[0][x] = *data.next().unwrap();
            trial[1][x] = *data.next().unwrap();
            trial[2][x] = *data.next().unwrap();
        }
    }

    // Make sure the drained the buffer
    assert_eq!(data.next(), None);

    Ok(data_buffer)
}

pub fn load_all_data() -> Result<Experiment> {
    // "EMGdata-RPS-04231538.mat"
    // "EMGdata-RPS-04231542.mat"
    // "EMGdata-RPS-04231545.mat"
    // "EMGdata-RPS-04231549.mat"
    // "EMGdata-RPS-04231553.mat"
    // "EMGdata-RPS-04231556.mat"
    // "EMGdata-RPS-04231559.mat"

    use RpsLabel::*;

    let data = [
        RawExperiment {
            data: load_trial_data(include_bytes!("../data/rps/EMGdata-RPS-04231538.mat"))?,
            labels: [
                scissors, rock, rock, rock, scissors, paper, paper, scissors, paper, scissors,
                rock, rock, rock, rock, paper, rock, rock, paper, scissors, scissors, paper,
                scissors, paper, paper, scissors, paper, scissors, scissors, paper, rock,
            ],
        },
        RawExperiment {
            data: load_trial_data(include_bytes!("../data/rps/EMGdata-RPS-04231542.mat"))?,
            labels: [
                rock, rock, rock, scissors, scissors, scissors, paper, rock, scissors, scissors,
                scissors, scissors, rock, paper, rock, scissors, scissors, paper, paper, paper,
                paper, paper, rock, rock, rock, rock, scissors, paper, paper, paper,
            ],
        },
        RawExperiment {
            data: load_trial_data(include_bytes!("../data/rps/EMGdata-RPS-04231545.mat"))?,
            labels: [
                scissors, scissors, scissors, scissors, scissors, rock, scissors, rock, scissors,
                rock, paper, rock, rock, paper, rock, paper, paper, paper, paper, rock, paper,
                paper, scissors, rock, scissors, scissors, rock, paper, rock, paper,
            ],
        },
        RawExperiment {
            data: load_trial_data(include_bytes!("../data/rps/EMGdata-RPS-04231549.mat"))?,
            labels: [
                rock, scissors, paper, paper, scissors, rock, scissors, paper, paper, rock, paper,
                paper, rock, scissors, paper, paper, scissors, scissors, scissors, rock, rock,
                scissors, rock, rock, rock, scissors, rock, scissors, paper, paper,
            ],
        },
        RawExperiment {
            data: load_trial_data(include_bytes!("../data/rps/EMGdata-RPS-04231553.mat"))?,
            labels: [
                paper, rock, rock, paper, paper, paper, paper, rock, scissors, rock, scissors,
                rock, scissors, rock, paper, scissors, scissors, rock, paper, rock, paper,
                scissors, rock, scissors, scissors, scissors, paper, paper, scissors, rock,
            ],
        },
        RawExperiment {
            data: load_trial_data(include_bytes!("../data/rps/EMGdata-RPS-04231556.mat"))?,
            labels: [
                rock, paper, paper, paper, rock, rock, scissors, scissors, paper, rock, paper,
                rock, paper, rock, paper, scissors, scissors, scissors, rock, scissors, scissors,
                paper, scissors, scissors, rock, rock, scissors, paper, paper, rock,
            ],
        },
        RawExperiment {
            data: load_trial_data(include_bytes!("../data/rps/EMGdata-RPS-04231559.mat"))?,
            labels: [
                rock, paper, scissors, paper, rock, rock, scissors, rock, rock, paper, scissors,
                paper, paper, rock, scissors, rock, scissors, paper, scissors, scissors, rock,
                paper, rock, paper, rock, paper, paper, scissors, scissors, scissors,
            ],
        },
    ];

    // split the training and testing data apart
    let (training_array, testing_array) = data.split_at(5);

    let (training, training_labels) = {
        let mut labels = Vec::new();
        let mut buf = vec![0.0_f64; NUM_CHANS * NUM_SAMPLES * NUM_TRIALS * 5].into_boxed_slice();
        let mut buf_iter = buf.iter_mut();

        for (dx, exp) in training_array.into_iter().enumerate() {
            // 30 trials
            for trial in exp.data.iter() {
                for chan in trial {
                    for pt in chan {
                        *buf_iter.next().unwrap() = *pt;
                    }
                }
            }
            labels.extend(exp.labels.iter().map(|f| *f as i64))
        }
        (buf, labels)
    };

    let (testing, testing_labels) = {
        let mut labels = Vec::new();
        let mut buf = vec![0.0_f64; NUM_CHANS * NUM_SAMPLES * NUM_TRIALS * 2].into_boxed_slice();
        let mut buf_iter = buf.iter_mut();

        for exp in testing_array {
            for chan in exp.data.iter() {
                for sig in chan {
                    for pt in sig {
                        *buf_iter.next().unwrap() = *pt;
                    }
                }
            }
            labels.extend(exp.labels.iter().map(|f| *f as i64))
        }
        (buf, labels)
    };

    Ok(Experiment {
        training,
        testing,
        training_labels,
        testing_labels,
    })
}

#[derive(Debug)]
pub struct NeuralDataset {
    pub train_signals: Tensor,
    pub train_labels: Tensor,

    pub test_signals: Tensor,
    pub test_labels: Tensor,
}

pub fn load_tch_data() -> Result<NeuralDataset> {
    let dataset = load_all_data()?;

    let train_signals = Tensor::of_slice(dataset.training.as_ref())
        .view((NUM_TRIALS as i64 * 5, (NUM_CHANS * NUM_SAMPLES) as i64))
        .to_kind(Kind::Float);

    let test_signals = Tensor::of_slice(dataset.testing.as_ref())
        .view((NUM_TRIALS as i64 * 2, (NUM_CHANS * NUM_SAMPLES) as i64))
        .to_kind(Kind::Float);

    let train_labels = Tensor::of_slice(dataset.training_labels.as_slice());

    let test_labels = Tensor::of_slice(dataset.testing_labels.as_slice());

    Ok(NeuralDataset {
        train_signals,
        train_labels,
        test_signals,
        test_labels,
    })
}

#[test]
fn stack_is_fine() {
    let raw = load_all_data().unwrap();
}

#[test]
fn loads_fine() {
    load_tch_data().unwrap();
}
