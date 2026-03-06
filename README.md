# Speech Feature Extraction & Other pre-processing
A modular pipeline for extracting, slicing, and preprocessing speech features for the ZeroResource Speech Challenge (ZRC). This repository supports feature extraction from self-supervised models (like HuBERT or WavLM) and prepares them for downstream clustering or submission.

## File Overview

* **`extract.py`**: Extracts hidden unit representations from SSL models (e.g., WavLM, HuBERT, XLS-R). Features an Enlighten-powered UI with real-time RAM monitoring.
* **`feature_slice.py`**: Align and cut continuous feature vectors into segments based on provided boundary files (e.g., VAD or phoneme-level boundaries).
* **`preprocess_zrc.py`**: Standardizes ZRC dataset formats, ensuring consistent sampling rates and directory structures across different languages.
* **`process_submission.py`**: The final stage of the pipeline; converts clusters or quantized units into the official ZeroSpeech submission format.

---

## Usage

### 1. Preprocess ZRC Data
Standardize audio and alignment directories for a specific language.
```bash
python preprocess_zrc.py <audio_dir> <align_dir> <language> [--only_alignments] 
```
### 2. Extract Features
Example: Using wavlm_large on wav files for layers 13 and 22
```bash
python extract.py wavlm_large <input_dir> <output_dir> 13 22 wav [--not_layer_norm]
```
### 3. Slice Features
Align and cut continuous features or audio into segments based on alignment grids.
Use flags --features, --grids, or --audio to specify what to slice:
```bash
python feature_slice.py <features_dir> <align_dir> [--features] [--grids]
```
### 3. Process Submission for ZRC Evaluation
Generate the final partition file for ZRC submission.
```bash
python process_submission.py <partition_path> <output_path>
```
