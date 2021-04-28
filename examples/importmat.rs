use anyhow::{anyhow, Context, Error, Result};
use matfile::{MatFile, NumericData};
use NeuroNet::*;

// imports 'rock, paper, scissors' into scope
use RpsLabel::*;

fn main() -> Result<()> {
    let exp1 = RawExperiment {
        data: load_trial_data(include_bytes!("../data/rps/EMGdata-RPS-04231538.mat"))?,
        labels: [
            rock, paper, scissors, paper, rock, rock, scissors, rock, rock, paper, scissors, paper,
            paper, rock, scissors, rock, scissors, paper, scissors, scissors, rock, paper, rock,
            paper, rock, paper, paper, scissors, scissors, scissors,
        ],
    };

    Ok(())
}
