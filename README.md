# Dolphin Whistle Detection

A comprehensive tool for detecting and extracting dolphin whistles from audio and video recordings using deep neural networks.

## 🌟 Features

- Automated detection of dolphin whistles using deep learning
- User-friendly GUI interface
- Command-line interface for advanced users
- Multi-threaded processing for improved performance
- Exportable results in CSV format
- Automatic extraction of detected whistle segments

## 📋 Prerequisites

- Python 3.9


## Installation

1. Clone the repository:
```bash
git clone https://github.com/AEmanuelli/Dolphins.git
cd Dolphins/DNN_whistle_detection/
```

2. Set up a virtual environment (recommended):
```bash
# Using conda (recommended)
conda create -n DolphinWhistleExtractor python=3.9
conda activate DolphinWhistleExtractor
conda install tk

```markdown
# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install tkinter
- Ubuntu
```bash
sudo apt-get install python3-tk 
```

- Fedora
```bash
sudo dnf install python3-tkinter
```

- MacOS
```bash
brew install python-tk
```


4. Install dependencies:
- **Option 1**: Use the GUI's "Install Dependencies" button
- **Option 2**: Manual installation via command line:
  ```bash
  pip install -r requirements.txt
  ```

## 💻 Usage

### GUI Interface (Recommended)

1. Launch the GUI:
```bash
python GUI_app.py
```

2. In the GUI:
   - Select your pre-trained model
   - Choose input folder containing recordings
   - Select output folder for results
   - Configure processing parameters:
     - Start/End times
     - Batch size
     - Frequency cutoffs (CLF/CHF)
     - Image normalization options
   - Click "Start Processing"

### Command Line Interface

For advanced users who prefer command-line operations:

```bash
# Run complete pipeline
python main.py --input_folder /path/to/recordings --output_folder /path/to/output

# Process specific files
python predict_and_extract_online.py --input_folder /path/to/recordings --model_path /path/to/model

# Process predictions
python process_predictions.py --predictions_folder /path/to/predictions
```

## 📁 Project Structure

```
├── __init__.py
├── models/
├── Models_vs_years_comparison
│   ├── figs/
│   └── models_vs_years_predictions.ipynb
├── Predict_and_extract
│   ├── Extraction_with_kaggle
│   │   ├── classify_again.ipynb
│   │   ├── utility_script (extraction_using_kaggle).py
│   │   └── Whistle_pipeline_(extraction_with_kaggle).ipynb
│   ├── __init__.py
│   ├── main.py
│   ├── GUI_app.py
│   ├── maintenance.py
│   ├── pipeline.ipynb
│   ├── predict_and_extract_online.py
│   ├── predict_online.py
│   ├── process_predictions.py
│   ├── Show_newly_detected_stuff.py
│   ├── utils.py
│   └── vidéoaudio.py
├── README.md (this file)
├── requirements.txt
└── Train_model
    ├── Create_dataset
    │   ├── AllWhistlesSubClustering_final.csv
    │   ├── create_batch_data.py
    │   ├── DNN précision - CSV_EVAL.csv
    │   ├── dolphin_signal_train.csv
    │   ├── save_spectrogram.m
    │   └── timings
    │       ├── negatives.csv
    │       └── positives.csv
    ├── csv2dataset.py
    ├── dataset2csv.py
    ├── fine-tune2.ipynb
    ├── fine-tune.ipynb
    ├── __init__.py
    └── Train.py
```

## 🔧 Key Components

### Prediction Pipeline

The system uses a three-stage pipeline:
1. **Detection**: Neural network processes audio/video files
2. **Extraction**: Identified whistles are extracted
3. **Post-processing**: Results are organized and saved

### Available Tools

- **GUI_app.py**: User-friendly interface for all operations
- **predict_and_extract_online.py**: Core detection engine
- **process_predictions.py**: Results processor
- **Show_newly_detected_stuff.py**: Comparison tool for new model outputs

## 📚 Additional Resources

- [Kaggle](https://www.kaggle.com/code/alexisemanuelli/whistles-detection-transfer-learning-0-95)
- [Model Fine-tuning Guide](https://www.kaggle.com/alexisemanuelli/fine-tune)

## Jean Zay Inference

For large-scale inference on Jean Zay, prepare file-list chunks and run a Slurm array:

```bash
python scripts/prepare_jeanzay_inference_chunks.py \
  --recordings-dir /store/rech/ioc/commun/dolphin_audio/2022 \
  --output-dir slurm/chunks_2022 \
  --chunk-size 100
```

Then submit the array:

```bash
sbatch --array=0-63 scripts/run_jeanzay_inference_array.sbatch
```

Override defaults at submission time when needed:

```bash
sbatch \
  --array=0-63 \
  --export=ALL,RECORDINGS_DIR=/store/rech/ioc/commun/dolphin_audio/2022,OUTPUT_DIR=/lustre/fswork/projects/rech/ioc/commun/DolphinWhistleExtractor/test/inference_2022_run3_jz,MODEL_PATH=/lustre/fswork/projects/rech/ioc/commun/DolphinWhistleExtractor/models/run3/model_vgg_best.pt,MAX_WORKERS=6,BATCH_SIZE=16,THRESHOLD=0.5 \
  scripts/run_jeanzay_inference_array.sbatch
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 📧 Contact

alexis.emanuelli@psl.eu
