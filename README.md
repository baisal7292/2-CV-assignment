# 2-CV-assignment
# Hand gesture recognition (peace / stop / thumbs up): A2 upgrade with SAM2 + DINOv2

Alym Baisal CV Assignment 2
Continuation of A1 (own phone-recorded dataset, 744 frames, split by video).

**Demo video:** [[VIDEO_LINK](https://drive.google.com/file/d/19DeHvBCWXzZAPt44D4rJ4cFpgvWwr2fk/view?usp=sharing)]
**Dataset (zip):** [[DATASET_LINK](https://drive.google.com/file/d/1ONOUFaHaJr-BarCfZ52Yz_v2cn7SIQHZ/view?usp=sharing)

## Idea
In A1, frozen MobileNetV2 reached ~99% on train but only ~60% on the test video, which had a different background and clothing.
In A2 I tried to reduce this background dependence:

1. **SAM2** (`sam2.1-hiera-small`) cuts the hand out of every training frame (one point prompt in the image centre).
2. The hand is pasted on **synthetic backgrounds** (solid, gradient, curtain, wood, noise) plus a fake grey sleeve, with small rotation/scale/brightness changes. All synthetic images are labelled as synthetic.
3. **DINOv2** (`dinov2-base`, frozen) replaces MobileNetV2 as the feature extractor. A linear head is trained on top.
4. 2x2 ablation: {MobileNetV2, DINOv2} x {original data, original + synthetic}, 3 seeds each, same head and early stopping.
5. Stress test per background (the `photo` family was never used in training), attention rollout, latency, Gradio app.

## Results (test set, 184 frames, mean over 3 seeds)

| Backbone | Training data | Accuracy | Macro-F1 |
|---|---|---|---|
| MobileNetV2 (A1 backbone) | original | 0.630 ± 0.014 | 0.540 ± 0.020 |
| MobileNetV2 | original + synthetic | 0.636 ± 0.009 | 0.544 ± 0.018 |
| DINOv2 | original | 0.645 ± 0.112 | 0.588 ± 0.125 |
| **DINOv2** | **original + synthetic** | **0.683 ± 0.027** | **0.612 ± 0.045** |

Accuracy with SAM2-pasted backgrounds (test hands on new backgrounds):
DINOv2 + synthetic is best on all six backgrounds, including the unseen `photo` family
(0.705 vs 0.605 for DINOv2 without synthetic data and 0.578 for MobileNetV2 + synthetic).
Full table: `stress_test.csv` in `outputs`.

Inference time: **21.4 ms / image** on a Colab Tesla T4 (DINOv2-base + linear head).

## Limitations (please read)
- Small dataset: 4 videos per class, 2 people, test = one video per class. The 3 seeds only change head initialisation and batch order, so the real uncertainty is larger than the std shown.
- Differences between configurations are small; the clearest effect is DINOv2 + synthetic data (higher accuracy, lower variance between seeds, better on the unseen background).
- `peace` remains the hardest class (F1 about 0.23); most peace frames are predicted as `stop`.
- Test hands are cut with the same SAM2 as the synthetic training data, so the background stress-test is partly favourable to the synthetic model. The `original test` column and the unseen `photo` background are the independent checks.
- Attention rollout of DINOv2 is noisy (a few bright points, not a hand contour). It shows where the features concentrate, not proof that only the hand is used.
- The app uses one point prompt in the image centre, so it works for close-up hands and can cut the face when the hand is small in the frame. In the demo, some stop frames on replaced backgrounds were wrongly predicted as peace.

## Ethics
All photos are of the author's own hands (recorded with consent). Synthetic images are generated procedurally and watermarked "SYNTHETIC BACKGROUND" in the app. Do not upload photos of other people without their consent.

## How to run
1. Open `A2_gesture_SAM2_DINOv2.ipynb` in Google Colab, set runtime to **T4 GPU**.
2. Put the dataset zip on Google Drive and set `ZIP_PATH` in the data cell (or upload it when asked).
3. Run all cells. The last cells launch the Gradio app and download `outputs.zip`.

`outputs (1).zip` contains the figures, result tables and the trained head from my run.
