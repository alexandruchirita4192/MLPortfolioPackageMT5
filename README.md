# MT5 Portfolio EA

This EA is a single master executor with two independent sleeves:

- BTCUSD M15 -> LinearSVC + calibrated probabilities
- XAGUSD M15 -> Scale-invariant ensemble (MLP + LightGBM + HGB)

## Required ONNX resources

Place these ONNX files where MetaEditor can compile them:

- ml_strategy_classifier_linearsvc_calibrated.onnx
- mlp.onnx
- lightgbm.onnx
- hgb.onnx

## What this solves

It avoids interference:
- one symbol closing another symbol position
- one strategy blocking another because some position is already open
- global bars-in-trade / cooldown / daily loss state

Everything important is stored per sleeve.

## Core rules

- BTCUSD and XAGUSD are managed independently
- each sleeve has its own magic number
- each sleeve only manages its own symbol + magic
- cooldown, bars-in-trade, and daily loss are per sleeve
- optional portfolio-level daily loss guard can stop both

## Recommended first run

Start with restrictive guards OFF on both:
- spread guard = false
- cooldown after close = false
- daily loss guard = false
- hour filter = false

Then re-enable protections one by one after confirming it trades correctly.


# Generate ONNX files for the Portfolio EA

This guide explains how to generate the ONNX files required by:

- `MT5_Portfolio_VA_BTCUSD_LinearSVC_XAGUSD_Ensemble.mq5`

The portfolio EA expects:

## BTCUSD sleeve
- `ml_strategy_classifier_linearsvc_calibrated.onnx`

## XAGUSD sleeve
- `mlp.onnx`
- `lightgbm.onnx`
- `hgb.onnx`

---

## 1. Required Python scripts

Use these two scripts exactly:

- `train_mt5_linearsvc_calibrated_classifier.py`
- `train_mt5_ensemble_scale_invariant.py`

I am intentionally **not merging them into one script** for now.

Reason:
- the BTCUSD sleeve uses a **single-model LinearSVC + calibration** pipeline
- the XAGUSD sleeve uses a **3-model ensemble**
- the portfolio EA already expects different ONNX filenames for each sleeve
- keeping them separate is simpler, clearer, and less error-prone

---

## 2. Python environment

Typical install:

```bash
pip install numpy pandas scikit-learn skl2onnx onnx MetaTrader5 lightgbm onnxmltools
```

You also need:
- a working MT5 terminal logged into the broker account you want to read data from
- or a CSV file if you prefer `--csv`

---

## 3. Generate BTCUSD ONNX for the BTC sleeve

Run the LinearSVC calibrated script for BTCUSD M15.

Example:

```bash
python train_mt5_linearsvc_calibrated_classifier.py --symbol BTCUSD --timeframe M15 --bars 80000 --horizon-bars 12 --train-ratio 0.70 --output-dir output_BTCUSD_linearsvc_M15_70
```

If you want to change model regularization:

```bash
python train_mt5_linearsvc_calibrated_classifier.py --symbol BTCUSD --timeframe M15 --bars 20000 --horizon-bars 12 --train-ratio 0.70 --c 1.0 --max-iter 3000 --calibration-cv 3 --output-dir output_BTCUSD_linearsvc_M15_70_custom_calibration
```

### Expected BTC output files

Inside `output_btc_linearsvc_m15` you should get:

- `ml_strategy_classifier_linearsvc_calibrated.onnx`
- `run_in_mt5.txt`
- `linearsvc_calibrated_metadata.json`
- `training_rates_snapshot.csv`
- `training_features_snapshot.csv`
- `train_predictions_snapshot.csv`
- `test_predictions_snapshot.csv`

---

## 4. Generate XAGUSD ONNX files for the XAG sleeve

Run the scale-invariant ensemble script for XAGUSD M15.

Example:

```bash
python train_mt5_ensemble_scale_invariant.py --symbol XAGUSD --timeframe M15 --bars 80000 --horizon-bars 8 --train-ratio 0.70 --mlp-weight 0.25 --lgbm-weight 0.25 --hgb-weight 0.50 --output-dir output_XAGUSD_ensemble_M15_70
```

If you want different weights:

```bash
python train_mt5_ensemble_scale_invariant.py --symbol XAGUSD --timeframe M15 --bars 80000 --horizon-bars 8 --train-ratio 0.70 --mlp-weight 0.20 --lgbm-weight 0.30 --hgb-weight 0.50 --output-dir output_XAGUSD_ensemble_M15_70
```

### Expected XAG output files

Inside `output_xag_ensemble_m15` you should get:

- `mlp.onnx`
- `lightgbm.onnx`
- `hgb.onnx`
- `run_in_mt5.txt`
- `ensemble_scale_invariant_metadata.json`
- `training_rates_snapshot.csv`
- `training_features_snapshot.csv`
- `train_predictions_snapshot.csv`
- `test_predictions_snapshot.csv`

---

## 5. Copy the ONNX files next to the Portfolio EA

The portfolio EA uses `#resource`, so the ONNX filenames must match exactly.

Put these files where MetaEditor can compile them together with the EA:

### From the BTC output folder
- `ml_strategy_classifier_linearsvc_calibrated.onnx`

### From the XAG output folder
- `mlp.onnx`
- `lightgbm.onnx`
- `hgb.onnx`

Do **not** rename them.

---

## 6. Compile the Portfolio EA

Compile:

- `MT5_Portfolio_VA_BTCUSD_LinearSVC_XAGUSD_Ensemble.mq5`

The EA is built as a master executor with two independent sleeves:

- BTCUSD M15 -> LinearSVC calibrated
- XAGUSD M15 -> scale-invariant ensemble

---

## 7. Use the Python output to fill the EA inputs

Read each `run_in_mt5.txt`.

### For BTCUSD sleeve, use values from:
- `output_btc_linearsvc_m15/run_in_mt5.txt`

Typical fields:
- `InpEntryProbThreshold`
- `InpMinProbGap`
- `InpMaxBarsInTrade`

Map them to BTC inputs:
- `InpBtcEntryProbThreshold`
- `InpBtcMinProbGap`
- `InpBtcMaxBarsInTrade`

### For XAGUSD sleeve, use values from:
- `output_xag_ensemble_m15/run_in_mt5.txt`

Map them to XAG inputs:
- `InpXagEntryProbThreshold`
- `InpXagMinProbGap`
- `InpXagMaxBarsInTrade`

Also copy the XAG weights into:
- `InpXagMlpWeight`
- `InpXagLgbmWeight`
- `InpXagHgbWeight`

---

## 8. Recommended first backtest settings

For the first validation, keep the restrictive protections off.

### BTCUSD
- `InpBtcUseSpreadGuard = false`
- `InpBtcUseCooldownAfterClose = false`
- `InpBtcUseDailyLossGuard = false`
- `InpBtcUseHourFilter = false`

### XAGUSD
- `InpXagUseSpreadGuard = false`
- `InpXagUseCooldownAfterClose = false`
- `InpXagUseDailyLossGuard = false`
- `InpXagUseHourFilter = false`

Reason:
you first want to confirm that:
- both sleeves load correctly
- both sleeves trade
- neither sleeve interferes with the other

Only after that should you re-enable guards one by one.

---

## 9. Typical workflow

1. Train BTCUSD with `train_mt5_linearsvc_calibrated_classifier.py`
2. Train XAGUSD with `train_mt5_ensemble_scale_invariant.py`
3. Copy the four ONNX files next to the portfolio EA
4. Compile the portfolio EA
5. Copy threshold / max-bars / weight values from the two `run_in_mt5.txt` files into the EA inputs
6. Backtest with loose guards first
7. Reintroduce protections gradually

---

## 10. If you want to use CSV instead of MT5 terminal data

Both scripts support:

```bash
--csv your_file.csv
```

Example:

```bash
python train_mt5_linearsvc_calibrated_classifier.py --csv btcusd_m15.csv --symbol BTCUSD --timeframe M15 --horizon-bars 12 --train-ratio 0.70 --output-dir output_btc_linearsvc_m15
```

```bash
python train_mt5_ensemble_scale_invariant.py --csv xagusd_m15.csv --symbol XAGUSD --timeframe M15 --horizon-bars 8 --train-ratio 0.70 --mlp-weight 0.25 --lgbm-weight 0.25 --hgb-weight 0.50 --output-dir output_xag_ensemble_m15
```

---

## 11. Important practical notes

### LinearSVC calibrated
ONNX export may depend on your exact sklearn / skl2onnx versions.

If export fails there, the training can still complete, but the ONNX export step may need version-compatible patching.

### Scale-invariant ensemble
This script exports three separate ONNX files and the portfolio EA combines them on the MT5 side through probability averaging.

That is why keeping the two Python scripts separate is the cleaner setup.

---

## 12. Final checklist

Before compiling the portfolio EA, confirm you have all four files:

- `ml_strategy_classifier_linearsvc_calibrated.onnx`
- `mlp.onnx`
- `lightgbm.onnx`
- `hgb.onnx`

If any one of them is missing, the portfolio EA will not compile correctly.
