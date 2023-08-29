## Installation

Details regarding installation of IsaacGym can be found [here](https://developer.nvidia.com/isaac-gym). **We currently support the `Preview Release 3/4` version of IsaacGym.**

### Pre-requisites

The code has been tested on Ubuntu 18.04/20.04 with Python 3.7/3.8. The minimum recommended NVIDIA driver
version for Linux is `470.74` (dictated by support of IsaacGym).

It uses [Anaconda](https://www.anaconda.com/) to create virtual environments.
To install Anaconda, follow instructions [here](https://docs.anaconda.com/anaconda/install/linux/).

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` 
directory, like `joint_monkey.py`. Please follow troubleshooting steps described in the Isaac Gym Preview Release 3/4
install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install this repo:

#### Install from source code
You can also install this repo from the source code:

```bash
pip install -e .
```

## Getting Started

### Interative GUI

Firstly, download our checkpoint in [here](https://drive.google.com/file/d/1rfi257wjXhYr_MuDuPbyXU-GesWme-cP/view?usp=sharing), and unzip it to the SeqDex/dexteroushandenvs/ folder (it will be SeqDex/dexteroushandenvs/checkpoint/). 

Then run the command line in `dexteroushandenvs` folder:

```bash
python train_rlgames.py --task BlockAssemblyGUI --algo lego --seed 22 --num_envs=1 --play
```

It may take a while for the first run to decompose mesh. It may be a bit long but Only needed for the first run. 

After entering the GUI of Isaac Gym, you can use the keyboard to select the pose and type of the block to be assembled. The control instructions are as follows:

| Key | Function |
|:-:|:-:|
|W|Move front|
|S|Move back|
|A|Move left|
|D|Move right|
|Q|Move up|
|E|Move down|
|T|Rotate clockwise|
|Y|Rotate counterclockwise|
|G|Select the target pose|
|H|Switch different block type|

<!-- ### Tasks

Source code for tasks can be found in `dexteroushandenvs/tasks`. -->

<!-- | Environments | Description | Demo     |
|  :----:  | :----:  | :----:  |
|ShadowHand Over| These environments involve two fixed-position hands. The hand which starts with the object must find a way to hand it over to the second hand. | <img src="assets/image_folder/0v2.gif" width="250"/>    |
|ShadowHandCatch Underarm|These environments again have two hands, however now they have some additional degrees of freedom that allows them to translate/rotate their centre of masses within some constrained region. | <img src="assets/image_folder/hand_catch_underarmv2.gif" align="middle" width="250"/>    |
|ShadowHandCatch Over2Underarm| This environment is made up of half ShadowHandCatchUnderarm and half ShadowHandCatchOverarm, the object needs to be thrown from the vertical hand to the palm-up hand | <img src="assets/image_folder/2v2.gif" align="middle" width="250"/>    |
|ShadowHandCatch Abreast| This environment is similar to ShadowHandCatchUnderarm, the difference is that the two hands are changed from relative to side-by-side posture. | <img src="assets/image_folder/1v2.gif" align="middle" width="250"/>    |
|ShadowHandCatch TwoCatchUnderarm| These environments involve coordination between the two hands so as to throw the two objects between hands (i.e. swapping them). | <img src="assets/image_folder/two_catchv2.gif" align="middle" width="250"/>    |
|ShadowHandLift Underarm | This environment requires grasping the pot handle with two hands and lifting the pot to the designated position  | <img src="assets/image_folder/3v2.gif" align="middle" width="250"/>    |
|ShadowHandDoor OpenInward | This environment requires the closed door to be opened, and the door can only be pulled inwards | <img src="assets/image_folder/door_open_inwardv2.gif" align="middle" width="250"/>    |
|ShadowHandDoor OpenOutward | This environment requires a closed door to be opened and the door can only be pushed outwards  | <img src="assets/image_folder/open_outwardv2.gif" align="middle" width="250"/>    |
|ShadowHandDoor CloseInward | This environment requires the open door to be closed, and the door is initially open inwards | <img src="assets/image_folder/close_inwardv2.gif" align="middle" width="250"/>    |
|ShadowHand BottleCap | This environment involves two hands and a bottle, we need to hold the bottle with one hand and open the bottle cap with the other hand  | <img src="assets/image_folder/bottle_capv2.gif" align="middle" width="250"/>    |
|ShadowHandPush Block | This environment requires both hands to touch the block and push it forward | <img src="assets/image_folder/push_block.gif" align="middle" width="250"/>    |
|ShadowHandOpen Scissors | This environment requires both hands to cooperate to open the scissors | <img src="assets/image_folder/scissors.gif" align="middle" width="250"/>    |
|ShadowHandOpen PenCap | This environment requires both hands to cooperate to open the pen cap  | <img src="assets/image_folder/pen.gif" align="middle" width="250"/>    |
|ShadowHandSwing Cup | This environment requires two hands to hold the cup handle and rotate it 90 degrees | <img src="assets/image_folder/swing_cup.gif" align="middle" width="250"/>    |
|ShadowHandTurn Botton | This environment requires both hands to press the button | <img src="assets/image_folder/switch.gif" align="middle" width="250"/>    |
|ShadowHandGrasp AndPlace | This environment has a bucket and an object, we need to put the object into the bucket  | <img src="assets/image_folder/g&p.gif" align="middle" width="250"/>    | -->

### Training

If you want to train a policy for the BlockAssemblyGraspSim task by the PPO algorithm, run this line in `dexteroushandenvs` folder:

```bash
python train_rlgames.py --task=BlockAssemblyGraspSim --algo=lego --seed=22 --num_envs=1024
```

### Inference

The trained model will be saved to `runs`folder.

To load a trained model and only perform inference (no training), pass `--play` as an argument, and pass `--checkpoint` to specify the trained models which you want to load.

```bash
python train_rlgames.py --task=BlockAssemblyGraspSim --algo=lego --checkpoint=./checkpoint/block_assembly/last_AllegroHandLegoTestPAISim_ep_19000_rew_1530.9819.pth --play --seed=22 --num_envs=1
```

