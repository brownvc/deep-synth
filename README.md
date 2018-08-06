# Deep Convolutional Priors for Indoor Scene Synthesis
PyTorch code for our SIGGRAPH Paper [Deep Convolutional Priors for Indoor Scene Synthesis](https://kwang-ether.github.io/pdf/deepsynth.pdf)

Requires PyTorch 0.4 to run, additional python library requirements could be found at [/deep-synth/requirements.txt](/deep-synth/requirements.txt). Run
```bash
pip install -r requirements.txt
```
to install.

## Creating the Dataset
To create the dataset, first obtain a copy of the [original SUNCG Dataset](http://suncg.cs.princeton.edu/) by signing the agreement form. The dataset should come in the form of a download script, say with the name `script.py`. 
Run 
```bash
python script.py
```
to download the newest version of the dataset, extract `house`, `object` and `texture`. Those should be the only files required. 
In addition, run 
```bash
python script.py --version v1 --type room
```
to download the room geometry in obj format, extract that as well.

Now, create a directory named `suncg_data` under the data root directory. This defaults to `/deep-synth/data`, but could be changed by exporting an environment variable named `SCENESYNTH_DATA_PATH`. Move the four directories downloaded previously to `suncg_data`.

Now, navigate to `/deep-synth`, and run
```bash
python create_data.py
```
To convert SUNCG into the format used by our code. This should create several new directories under the data root directory, the only important ones are `bedroom`, `living` and `office`, which are the datasets for the three types of rooms we include in the paper. Since the SUNCG version is newer than what we used, there might be minor discrepancies.

## Training the Models
We provide three training scripts: `continue_train.py`, `location_train.py` and `rotation_train.py` that trains each of the three neural network components. The neural networks are described in detail in section 5 of our paper.

Pre-trained models will be released soon.

### 1.Training the continue predictor (Section 5.1)

Run
```bash
python continue_train.py --data-dir dataset_location --save-dir save_destination --train-size train_set_size --use-count
```
to train in the same way as what we did. Available arguments:

- `--data-dir`: location of the dataset under data root directory.

- `--save-dir`: location to save the models

- `--use-count`: when set, the global category counts are included in the training process, vice versa. Since our test time synthesis code is currently hardcoded to use category counts, it is recommended not to drop this argument.

- `--lr` and `--eps`: used to control the parameters for the Adam optimizer. We did not fine tune those parameters, any reasonable setting should yield comparable training results.

- `--train-size`: size of the training set. Currently, it is hardcoded that the 160 rooms directly succeeding the training set is used as the validation set, so make sure the size of the training set + 160 does not exceed the size of the entire dataset.

- `--num-workers`: number of workers for the data loader. Since actual training dataset is generated on the fly, the training process is fairly CPU-hungry, we would recommend using at least 6 workers to load the data. (If the total available RAM is less than 32GB, consider reduce the number of workers)

- `--last-epoch`: if set, resume training from the specified epoch.

- `--ablation`: Controls input channel ablations. If not set, train with all the channels. If set to "depth", use only the depth channels. If set to "basic", use everything apart from the one hot category channels.

Since the size of different room datasets are different, we standardized it and call each 10,000 rooms seen as a epoch. Empirically, the continue predictor should be usable after 50 epochs.

### 2.Training the location-category predictor (Section 5.2)

Run
```bash
python location_train.py --data-dir dataset_location --save-dir save_destination --train-size train_set_size --use-count --progressive-p 
```
to train in the same way as what we did. In addition to the parameters outlined above, there are:

- `--p-auxiliary`: Chance that an auxiliary category is sampled. 

- `--progressive-p`: If set, increase p_auxiliary gradually from 0 to 0.95. Overrides `--p-auxiliary`.

Location-category predictor should be usable after 300 epochs. The training process for this is quite unstable, so diffrent epochs might behave differently, especially after test-time tempering. Experiment with them if you like.

### 3.Training the instance-orientation predictor (Section 5.3)

Run
```bash
python rotation_train.py --data-dir dataset_location --save-dir save_destination --train-size train_set_size
```
to train in the same way as what we did. Note that we actually did not include category count information for this network, so `--use-count` is not available.

Instance-orientation predictor should be usable after 300 epochs of training.

## Test-time Synthesis
[scene_synth.py](/deep-synth/scene_synth.py) contains the code used for test time synthesis. [batch_synth.py](/deep-synth/batch_synth.py) is a simple script that calls the synthesis code. To use it, run
```bash
python batch_synth.py --save-dir save_destination --data-dir dataset_location --model-dir model_location --continue-epoch epoch_number --location_epoch epoch_number --rotation_epoch epoch_number --start start_room_index --end end_room_index
```

Available arguments are:

- `--data-dir`: location of the dataset under data root directory.

- `--model-dir`: location of the trained models, relative to the location of the code. To modify the root directory for the models, check `scene_synth.py` (`SceneSynth.__init__`). It is assumed that all the trained models are in this directory, with names in the format that was written by the training code.

- `--save-dir`: location to save the models

- `--continue-epoch`, `--location-epoch`, `--rotation-epoch`: epoch number to use for the three neural networks, respectively.

- `--start`, `--end`: specifies the range of room indices used for synthesis.

- `--trials`: number of synthesis per each input room

In addition, four parameters can be specified to change the synthesizer behavior, they defaults to what we used to produce the results in the paper:

- `--temperature-cat`, `--temperature-pixel`: temperature settings used to temper the location-category distributions. Refer to the final part of Section 5.2 for details about this. The two parameters control τ_1 and τ_2 respectively.

- `--min-p`: minimum probability (by the instance-orientation network) that a insertion can be accepted.

- `--max-collision`: maximum amount of collision (specified as a negative float number) that is allowed for a new insertion. The actual way of handling insertion is a bit more complex, refer to the code for more details on that.

There are three ways you can view a synthesized scene:

- Directly browsing the png files generated by the code.

- Use the Scene Viewer to view the generated .json files. We will setup a instance of that later.

- Use Scene Toolkit to convert the .json files to standard mesh files, which could then be loaded/rendered by various softwares. We will update the instructions on that later.

## Citation
Please cite the paper if you use this code for research:
```
@article{wang2018deep,
  title={Deep convolutional priors for indoor scene synthesis},
  author={Wang, Kai and Savva, Manolis and Chang, Angel X and Ritchie, Daniel},
  journal={ACM Transactions on Graphics (TOG)},
  volume={37},
  number={4},
  pages={70},
  year={2018},
  publisher={ACM}
}
```
