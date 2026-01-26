# Output layout

## extraction

Models directly output by extraction training.

## extraction/tensorboard_logs

Logging from TensorBoard. To view plots from TensorBoard, run

`tensorboard --logdir outputs/extraction/tensorboard_logs`

and open `http://localhost:6006` in a browser.

## models

Move models here to be used for inference.

## logs

Logs producing by logging library.

## old_checkpoints

Models from older experiments.

## predictions

Unused at the moment, if predictions are saved to a separate file, default to
saving here.

## custom

Model files from testing.
