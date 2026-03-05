# CLTR: Empatica Wearable Physiology Pipeline

CLTR is a modular software package designed for the preprocessing, analysis, and reporting of wearable physiology data, specifically tailored for Empatica (E4 and EmbracePlus) exports. It streamlines the path from raw sample streams to publication-ready reports and structured features.

## Core Features

- **Robust Preprocessing**: Includes cleaning and feature extraction for EDA, BVP, Accelerometer, and Skin Temperature.
- **Protocol-Aware Analysis**: Aligns physiological data with protocol timelines, enabling phase-level aggregation and interpretation.
- **Advanced QC**: Legible minute-level quality control flags, plausibility range checks, and high-motion detection.
- **Multi-Format Reporting**:
    - **PDF Reports**: Static, publication-ready summaries with detailed visualizations using Matplotlib.
    - **Interactive HTML Reports**: Rich, Plotly-based dashboards for deep data exploration and session comparison.
- **Cohort Analysis**: Automated aggregation across multiple sessions with meta-analysis of correlations and cohort-wide visualizations.
- **Flexible Integration**: Can be used as a standalone command-line tool or as a Python library.

## Installation

Install CLTR directly from the source in editable mode:

```bash
git clone https://github.com/tomarp/CLTR_test
cd CLTR_test
pip install -e .
```

### Dependencies

- Python >= 3.8
- NumPy, Pandas, SciPy
- Matplotlib, Plotly
- NeuroKit2 (required for EDA/BVP processing)

## Command-Line Usage

The primary entry point is the `cltr-pipeline` command.

### Basic Batch Processing

Processes a directory of session folders and generates PDF reports and CSV features:

```bash
cltr-pipeline \
  --sessions-root /path/to/raw_data \
  --outdir /path/to/results \
  --timeline-csv /path/to/protocol_timeline.csv \
  --timeline-tz Europe/Paris
```

### Generating Interactive Reports

To generate interactive HTML reports along with the standard pipeline, add the `--interactive` flag:

```bash
cltr-pipeline \
  --sessions-root /path/to/raw_data \
  --outdir /path/to/results \
  --timeline-csv /path/to/timeline.csv \
  --interactive
```

### CLI Arguments

| Argument | Description |
| :--- | :--- |
| `--sessions-root` | Path to the directory containing per-session subfolders. |
| `--outdir` | Path where results will be saved. |
| `--timeline-csv` | Path to the protocol timeline CSV file (required). |
| `--timeline-tz` | Timezone for timeline localization (default: Europe/Paris). |
| `--interactive` | Flag to enable generation of Plotly-based HTML reports. |

## Library Usage

You can also import CLTR modules into your own Python scripts:

```python
from cltr.processing import eda_process
from cltr.io import read_csv_safely

# Load and process EDA data
eda_data = read_csv_safely("eda.csv")
processed_df, info = eda_process(eda_data)
```

## Package Structure

- `cltr.processing`: Signal cleaning and feature extraction logic.
- `cltr.analysis`: Statistical routines, gap detection, and aggregation.
- `cltr.pdf_report`: Matplotlib logic for static PDF generation.
- `cltr.interactive_report`: Plotly logic for interactive HTML dashboards.
- `cltr.pipeline`: High-level orchestration of the processing pipeline.
- `cltr.io`: Robust file reading/writing and schema management.
- `cltr.cli`: Command-line interface definition.

## License

This project is intended for research and educational purposes.
