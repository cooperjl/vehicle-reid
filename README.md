# Vehicle Re-Identification with Deep Learning
University of York dissertation project in the field of vehicle re-identification.

## Installation
Recommended usage is with uv:
```shell
uv run vehicle-reid --help
```
Alternatively, if uv unavailable on the system being used, setup using Python and pip:
```shell
python -m venv .venv
source .venv/bin/activate
pip install .
vehicle-reid --help
```
## Configuring
Configuration files with the parameters from Appendix B in the report are available already in this repository.
Their default configuration expects data to be stored in the `data` folder for each dataset. 
In particular, for VeRi-776, the root directory is expected to be `data/VeRi`, and for VRIC, it is expected to be
`data/vric`. No modification to either dataset is required, it is simply enough to extract the contents of the
archives into these folders. For VeRi-776, the archive contains the `VeRi` folder, and for VRIC, it should be extracted
into the `vric` folder.

## Usage
If not using uv, remove `uv run` from following commands, and otherwise use the same way.
### Training
```
uv run vehicle-reid --config-file CONFIG_FILE train
```
This will start training from the beginning, according to parameters of the CONFIG_FILE.
### Testing
```
uv run vehicle-reid --config-file CONFIG_FILE test MODEL.CHECKPOINT CHECKPOINT_FILE
```
This will test the model, loading the file CHECKPOINT_FILE.
```
uv run vehicle-reid --config-file configs/veri.yml MODEL.CHECKPOINT VeRi-model.pth
```
This will test the model for the VeRi-776 dataset.

```
uv run vehicle-reid --config-file configs/vric.yml MODEL.CHECKPOINT VRIC-model.pth
```
This will test the model for the VRIC dataset.
### Other Operations
```
uv run vehicle-reid --config-file CONFIG_FILE command ...
```
The config file is required for all uses, as it specifies the directories where files can be accessed,
which is required in some way for each command.

Other command options include: 
- `compute-relational-data`, which computes the relational data. This data
has already been computed, and is available in `data/rel` in this repository.
- `calculate-normal-values`, which computes the mean and standard deviation for the dataset specified,
which has already been computed, and is available in the respective configuration files.
- `visualise`, which visualises the outputs in the form of a ranked list.
- `plot`, which parses log files and plots the average loss and mean average precision, as seen in the report.


