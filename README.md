# Vessel Detection from a Submarine Cable

Can a neural network hear a ship through an underwater cable?

This project uses fibre-optic sound recordings (Distributed Acoustic Sensing, DAS) to decide whether a vessel is within **1,000 m** of a submarine cable, and to estimate how far away it is. It is built in PyTorch and tested on days the model has never seen.


---

## Why this problem

Undersea cables carry most international internet traffic, and ships can damage them. Ships normally report their position with AIS, but AIS can be switched off. A DAS interrogator turns the cable itself into a long line of microphones, so the cable can detect ships without their cooperation.

## Data

The public **Marlinks-NS DAS** dataset ([Zenodo, DOI 10.5281/zenodo.15611778](https://doi.org/10.5281/zenodo.15611778), CC BY 4.0).

| Item | Value |
|---|---|
| Recordings | 74,771, each 10 seconds long |
| Period | 10 days, 16 to 25 June 2023 |
| Cable | 28 km, North Sea off Zeebrugge, Belgium |
| One recording | 250 channels x 100 frequency bands |
| Label | Distance to the nearest vessel, from AIS |
| Task | Near (under 1,000 m, 32% of recordings) or far |

Vessel identities are removed and the cable position is displaced for security.

The raw file (about 14 GB) is not stored in this repository. Download `dataset.h5` from Zenodo.

## Method

**Preprocessing** (per recording)
1. Remove channels 59, 60 and 61, which the dataset authors flag as faulty (about 97% of their values are exactly zero in the first 500 recordings).
2. For each frequency band, take the maximum, the mean and the 90th percentile across the remaining 247 channels. This gives 300 features.
3. Apply `log10`. The raw values span about ten orders of magnitude, and a model trained on them learned nothing (AUC 0.51).

**Models**
- **ANN:** 300 inputs, two hidden layers (256, 128), ReLU, dropout 0.3, one output. Weighted binary cross-entropy (near recordings count twice), Adam (learning rate 0.001, weight decay 0.0001), batch size 64, early stopping on a validation day.
- **LSTM:** reads sequences of consecutive recordings from the same day (5 recordings = 50 seconds), same loss and optimiser.
- **Distance estimation:** the same ANN trained with an L1 loss on near recordings only.

**Evaluation: leave-one-day-out, 10 folds.** In each fold one day is the test day (never seen), the next day is the validation day (only used to decide when to stop), and the other eight days are for training. The scaler and class weights are fitted on the training days only. A random split was rejected because recordings next to each other look alike and leak information.

Metrics: accuracy, near-class F1 (the main metric, because missing a nearby ship is the costly mistake), AUC, and mean absolute error for distance.

## Results

Mean of the 10 held-out test days.

| Model | Input | Accuracy | Near F1 | AUC |
|---|---|---|---|---|
| ANN | 100 features (max only) | 0.876 | 0.818 | 0.938 |
| ANN | 300 features (max, mean, 90th percentile) | 0.877 | 0.825 | 0.941 |
| LSTM | 5 recordings x 100 features | 0.869 | 0.814 | 0.938 |

Near F1 varies by day with a standard deviation of about 0.03. 22 June is the hardest day for every model.

**Distance estimation** (vessels within 1,000 m): mean absolute error **154 m**, against 256 m for always guessing the typical distance (day-by-day range 128 to 171 m).

**Detection by vessel length** (share of nearby vessels found, ANN with 300 features):

| Under 50 m | 50 to 100 m | 100 to 150 m | 150 to 200 m | Over 200 m |
|---|---|---|---|---|
| 0.51 | 0.57 | 0.80 | 0.93 | 0.94 |

### What was learned

- **Input scale and fair testing matter as much as the model.** `log10` took the first model from guessing to learning, and a day-wise split gave a more honest score than a random split.
- **Most ideas stayed near the same ceiling (about 0.82 near F1).** More channel features gave a small gain (0.818 to 0.825), smoothing predictions over time gave at most about half a point, and tuning dropout, weight decay and learning rate showed no clear winner (validation loss 0.369 to 0.376). Weighting small vessels more lowered the overall score (0.825 to 0.809).
- **Longer time context was the exception (preliminary).** On three test days (22, 24 and 25 June), LSTM near F1 was 0.805, 0.811 and 0.838 for 30, 50 and 100 seconds of context. This has not yet been run on all ten days.
- **Small vessels are the weak spot.** They are probably quieter, but length is mixed up with vessel type, so this is a pattern and not a proven cause.

### Comparison with the original study

The dataset comes with a published study (Ramirez-Torres et al., 2026). Its neural network scores about 0.80 near F1 and its best XGBoost system about 0.84, read from its figures and tables, so these values are approximate. The ANN here is in the same range. The comparison is not like for like: the study averages groups of channels and uses up to 50 seconds of context with majority voting, while the ANN here uses a single 10-second recording.

## Repository layout

```
Vessel_Detection_DAS/
  src/                     helper code
    data_loader.py           open the HDF5 file, labels, timestamps
    preprocessing.py         preprocessing helpers
    models.py                ANNModel, LSTMModel
    training.py              training loop with early stopping
    evaluation.py            predictions and metrics
  notebooks/
    01_data_inspection.ipynb       look at the data, dead channels, class balance
    02_small_data_ann.ipynb        first model on a small sample
    03_ann_daywise_cv.ipynb        3-day pilot of day-wise testing
    04_ann_10day_cv.ipynb          ANN, 100 features, all 10 folds
    05_lstm_daywise_cv.ipynb       LSTM, sequence-length checks
    06_ann_features_v2.ipynb       ANN, 300 features, distance estimation, error analysis
    08_demo_end_to_end.ipynb       whole pipeline in one notebook (for the demo)
  app/app.py               Streamlit dashboard
  results/
    tables/                  per-fold results and predictions (CSV)
    figures/                 saved figures
    models/                  saved demo model and scaler
  requirements.txt
```

The feature files (`results/features_all_days*.npz`) are generated locally and are not part of the repository.

## How to run it

**1. Set up.** Python 3.12. Install PyTorch for your machine from [pytorch.org](https://pytorch.org) (the runs here used PyTorch 2.6 with CUDA 12.4), then:

```bash
pip install -r requirements.txt
```

**2. Get the data.** Download `dataset.h5` from the Zenodo record above. In the notebooks and in `app/app.py`, set `DATA_PATH` (or `data_path`) to where you saved it.

**3. Run the whole pipeline.** Open `notebooks/08_demo_end_to_end.ipynb` and choose Run All. It reads the data, shows the preprocessing, builds the feature file on the first run (about 15 minutes, done once), and trains on three test days (about 8 minutes). Set `QUICK = False` to train all ten days (about 26 minutes). The full 10-day results for all three models are loaded from the saved CSV files in `results/tables/`.

**4. Reproduce the experiments.** Run notebook 04 (ANN, 100 features), 06 (ANN, 300 features, distance estimation) and 05 (LSTM). Each saves its per-fold results to `results/tables/`. The random seed is fixed at 42.

**5. Dashboard.** After the demo model has been saved by notebook 06 (into `results/models/`):

```bash
python -m streamlit run app/app.py
```

It opens in the browser at `http://localhost:8501`. Pick a day, move through the recording, and compare the real ship distance with the model's probability. The saved model never saw 24 June in training, so that is the day to use.

## Limitations

- One cable and 10 days of data. Results may not transfer to another cable.
- Distance labels come from AIS positions that arrive every 1 to 3 minutes and are interpolated in between, so labels near 1,000 m are uncertain, and a vessel with AIS off is labelled far.
- Differences below about one point of near F1 are within the day-to-day spread and come from a single run per fold.
- The distance experiment covers near recordings only, so it does not include near-or-far classification errors.
- The longer-context LSTM result is preliminary (three test days).

## Future work

Run longer context on all ten days, add vessel size as an input, try a 1D-CNN across channels, compare against XGBoost run locally, and test on another cable.

## References

- Ramirez-Torres, E. E., Macias-Guarasa, J., Pizarro-Perez, D., Tejedor, J., Palazuelos-Cagigas, S. E., Vidal-Moreno, P. J., Martin-Lopez, S., Gonzalez-Herraez, M. and Vanthillo, R. (2026) *Vessel detection and localization using distributed acoustic sensing in submarine optical fiber cables*. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. [arXiv:2509.11614](https://arxiv.org/abs/2509.11614).
- Ramirez-Torres, E. E. et al. (2026) *A distributed acoustic sensing dataset for vessel detection and localization in submarine cable protection*. [arXiv:2607.28306](https://arxiv.org/abs/2607.28306).
- Ramirez-Torres, E. E. et al. (2026) *Marlinks-NS DAS: dataset for vessel detection and distance estimation using distributed acoustic sensing in submarine cables*. Zenodo. [10.5281/zenodo.15611778](https://doi.org/10.5281/zenodo.15611778).

## Licence

The dataset is released under CC BY 4.0 by its authors. This repository is coursework from the Adelaide University.