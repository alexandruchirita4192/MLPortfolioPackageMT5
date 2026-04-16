# MT5 Portfolio EA Version A

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
