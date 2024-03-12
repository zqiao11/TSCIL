<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->


<!-- ABOUT THE PROJECT -->
## Class-incremental Learning for Time Series: Benchmark and Evaluation

A unified experimental framework for Time Series Class-Incremental Learning (TSCIL) based on Pytorch. The paper is currently under review and avaible on [arxiv](https://arxiv.org/abs/2402.12035). Our CIL benchmarks are established with open-sourced real-world time series datasets. Based on these, our toolkit provides a simple way to customize the continual learning settings. Hyperparameter selection is based on [Ray Tune](https://docs.ray.io/en/latest/tune/index.html). 


[//]: # (Public time series datasets for Human Activity Recognition or Gesture Recognition are used. All the selected datasets are balanced. Two backbones &#40;1D-CNN and Time Series Transformer&#41; are included. We also consider the impact of different normalization layers &#40;BN and LN&#41; in Continual Learning.)

[//]: # ( )
[//]: # (We follow the standard CL experiment protocol as [AGEM]&#40;https://arxiv.org/abs/1812.00420&#41;, spliting the tasks into `Val Tasks` and `Exp Tasks`, for hyperparameter tuning and CL experiment, respectively.  )

## Requirements
![](https://img.shields.io/badge/python-3.10-green.svg)

![](https://img.shields.io/badge/pytorch-1.13.1-blue.svg)
![](https://img.shields.io/badge/ray-2.3.1-blue.svg)
![](https://img.shields.io/badge/PyYAML-6.0-blue.svg)
![](https://img.shields.io/badge/scikit--learn-1.0.2-blue.svg)
![](https://img.shields.io/badge/matplotlib-3.7.1-blue.svg)
![](https://img.shields.io/badge/pandas-1.5.3-blue.svg)
![](https://img.shields.io/badge/seaborn-0.12.2-blue.svg)

### Create Conda Environment

1. Create the environment from the file
   ```sh
   conda env create -f environment.yml
   ```

2. Activate the environment `tscl`
   ```sh
   conda activate tscl
   ```
----

## Dataset
### Available Datasets
1. [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
2. [UWAVE](http://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibraryAll)
3. [Dailysports](https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities) 
4. [WISDM](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
5. [GrabMyo](https://physionet.org/content/grabmyo/1.0.2/)


### Data Prepareation
We process each dataset individually by executing the corresponding `.py` files located in `data` directory. This process results in the formation of training and test `np.array` data, which are saved as `.pkl` files in `data/saved`. 

For UCI-HAR, Uwave and Dailysports, we directly use the original raw time series as samples. For the remaining, we use sliding windows to extract samples with appropriate sequence length (downsampling may be applied before window sliding). If the original dataset is not pre-divided into training and testing sets, a manual split into train and test sets will be conducted. Information about the processed data can be found in `utils/setup_elements.py`. The saved data are **not preprocessed with normalization** since the continual learning setup. Instead, we add a non-trainable input normalization layer before the encoder to do the sample-wise normalization. 

We provide the processed data files for convenience. Please check the setup in the "Get Started" section.

### Adding New Dataset
Create a new python file in `data` for the new dataset. Process the data to form samples with identical sequence length using sliding window. Then perform the train-test split. Save the numpy arrays of training data, training labels, test data, and test labels into `x_train.pkl`, `state_train.pkl`,`x_test.pkl`, `state_test.pkl` in a new folder in `data/saved`. Finally, add the necessary information of the dataset in `utils/setup_elements.py`

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Continual Learning Algorithms
### Existing Algorithms
Regularization-based:
* [LwF](https://arxiv.org/abs/1606.09282)
* [EWC](https://arxiv.org/abs/1612.00796)
* [SI](https://arxiv.org/abs/1703.04200)
* [MAS](https://arxiv.org/abs/1711.09601)
* [DT2W](https://ieeexplore.ieee.org/abstract/document/10094960)

Replay-based:
* [ER](https://arxiv.org/abs/1811.11682)
* [DER](https://arxiv.org/abs/2004.07211)
* [Herding](https://arxiv.org/abs/1611.07725)
* [ASER](https://arxiv.org/abs/2009.00093)
* [CLOPS](https://www.nature.com/articles/s41467-021-24483-0)
* [Generative Replay](https://arxiv.org/abs/1705.08690)
* [DeepInversion](https://arxiv.org/abs/1912.08795) (beta)
* [Mnemonics](https://arxiv.org/abs/2002.10211)

### Adding New Algorithm
Create a new python file in `agent` for the new algorithm. Create a subclass that inherits from the `BaseLearner` class in `agent/base.py`. Then custom methods `train_epoch()`, `after_task()` and `learn_task()` based on your needs. Finally, add the new method to the dictionary in `agents/utils/name_match.py`.
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started


### Setup
1. Download the processed data from [Google Drive](https://drive.google.com/drive/folders/1EFdD07myqmqHhRsjeQ83MdF8gHZXDWLR?usp=share_link). Put it into `data/saved` and unzip
   ```sh
   cd data/saved
   unzip <dataset>.zip
   ```
   You can also download the raw datasets and process the data with the corresponding python files.
2. Revise the following to suit your device:
    * `resources` in `tune_hyper_params` in `experiment/tune_and_exp.py` (See [here](https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html) for details)
    * GPU numbers in the `.sh` files in `shell`

### Run Experiment

There are two functions to run experiments. Set the arguments in the corresponding files or in the command line.

1. Run CIL experiments with custom configurations. Note that this function cannot tune/change the hyperparameters for multiple runs. It is used for sanity check or debugging.
   ```sh
   python main_config.py
   ```

2. Tune the hyperparameters on the `Val Tasks` first, and then use the best hyperparameters to run experiments on the `Exp Tasks`:
   
   ```sh
   python main_tune.py
   ```
   To run multiple experiments, you can revise and call `shell/tune_and_exp.sh`:
   ```sh
   nohup sh shell/tune_and_exp.sh &
   ```
   To reproduce all the experiments in the paper, use the corresponding `.sh` files:
   ```sh
   nohup sh shell/{data}_all_exp.sh &
   ```
    We run the experiment for multiple times/runs to compute average performance. In each run, we randomize the class order and tune the best hyperparameters. So the hyperparameters are different across runs. Experiment results will be saved into `result/tune_and_exp`.

### Custom Experiment Setup
Change the configurations in 
* `utils/setup_elements.py`: Parameters for data and task stream, including Number of tasks / Number of classes per task / Task split
* `experiment/tune_config.py`: Parameters for `main_tune.py` experiments, such as Memory Budget / Classifier Type / Number of runs / Agent-specific parameters, etc.

For ablation study, revise the corresponding parameters in `experiment/tune_config.py` and rerun the experiments.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Acknowledgements -->
## Acknowledgements
Our implementation uses the source code from the following repositories:

* Framework & Buffer & LwF & ER & ASER: [Online Continual Learning in Image Classification: An Empirical Survey](https://github.com/RaptorMai/online-continual-learning)
* EWC & SI & MAS: [Avalanche: an End-to-End Library for Continual Learning](https://github.com/ContinualAI/avalanche)
* DER: [Mammoth - An Extendible (General) Continual Learning Framework for Pytorch](https://github.com/aimagelab/mammoth)
* DeepInversion: [Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion](https://github.com/NVlabs/DeepInversion)
* Herding & Mnemonics: [Mnemonics Training: Multi-Class Incremental Learning without Forgetting](https://github.com/yaoyao-liu/class-incremental-learning)
* Soft-DTW: [Soft DTW for PyTorch in CUDA](https://github.com/Maghoumi/pytorch-softdtw-cuda)
* CNN: [AdaTime: A Benchmarking Suite for Domain Adaptation on Time Series Data](https://github.com/emadeldeen24/AdaTime)
* TST & lr scheduler: [PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://github.com/yuqinie98/PatchTST)
* Generator: [TimeVAE for Synthetic Timeseries Data Generation](https://github.com/abudesai/timeVAE)



<!-- CONTACT -->
## Contact
For any issues/questions regarding the repo, please contact the following.

Zhongzheng Qiao - qiao0020@e.ntu.edu.sg

School of Electrical and Electronic Engineering (EEE),
Nanyang Technological University (NTU), Singapore.
<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
