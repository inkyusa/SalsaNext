[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/salsanext-fast-semantic-segmentation-of-lidar/3d-semantic-segmentation-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=salsanext-fast-semantic-segmentation-of-lidar) [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2003.03653)

# SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving

## Abstract 

In this paper, we introduce SalsaNext for the uncertainty-aware semantic segmentation of a full 3D LiDAR point cloud in real-time. SalsaNext is the next version of SalsaNet which has an encoder-decoder architecture where the encoder unit has a set of ResNet blocks and the decoder part combines upsampled features from the residual blocks. In contrast to SalsaNet, we introduce a new context module, replace the ResNet encoder blocks with a new residual dilated convolution stack with gradually increasing receptive fields and add the pixel-shuffle layer in the decoder. Additionally, we switch from stride convolution to average pooling and also apply central dropout treatment. To directly optimize the Jaccard index, we further combine the weighted cross-entropy loss with Lovasz-Softmax loss . We finally inject a Bayesian treatment to compute the epistemic and aleatoric uncertainties for each point in the cloud. We provide a thorough quantitative evaluation on the Semantic-KITTI dataset, which demonstrates that the proposed SalsaNext outperforms other state-of-the-art semantic segmentation.
## Examples 
![Example Gif](/images/SalsaNext.gif)

### Video 
[![Inference of Sequence 13](https://img.youtube.com/vi/MlSaIcD9ItU/0.jpg)](http://www.youtube.com/watch?v=MlSaIcD9ItU)



### Semantic Kitti Segmentation Scores

The up-to-date scores can be found in the Semantic-Kitti [page](http://semantic-kitti.org/tasks.html#semseg).

## How to use the code

First create the anaconda env with:
```conda env create -f salsanext_cuda10.yml --name salsanext``` then activate the environment with ```conda activate salsanext```.

To train/eval you can use the following scripts:


 * [Training script](train.sh) (you might need to chmod +x the file)
   * We have the following options:
     * ```-d [String]``` : Path to the dataset
     * ```-a [String]```: Path to the Architecture configuration file 
     * ```-l [String]```: Path to the main log folder
     * ```-n [String]```: additional name for the experiment
     * ```-c [String]```: GPUs to use (default ```no gpu```)
     * ```-u [String]```: If you want to train an Uncertainty version of SalsaNext (default ```false```) [Experimental: tests done so with uncertainty far used pretrained SalsaNext with [Deep Uncertainty Estimation](https://github.com/uzh-rpg/deep_uncertainty_estimation)]
   * For example if you have the dataset at ``/dataset`` the architecture config file in ``/salsanext.yml``
   and you want to save your logs to ```/logs``` to train "salsanext" with 2 GPUs with id 3 and 4:
     * ```./train.sh -d /dataset -a /salsanext.yml -m salsanext -l /logs -c 3,4```
   * Command line used for training on a local PC
     * ```./train.sh -d /media/user/programfox2TB/dataset/semantic_kitti/dataset -a /home/user/workspace/SalsaNext/salsanext.yml -l /home/user/workspace/SalsaNext/ -c 0```
  
  You should be able to see the following console output
```
----------
INTERFACE:
dataset /media/user/programfox2TB/dataset/semantic_kitti/dataset
arch_cfg /home/user/workspace/SalsaNext/salsanext.yml
data_cfg config/labels/semantic-kitti.yaml
uncertainty False
Total of Trainable Parameters: 6.71M
log /home/user/workspace/SalsaNext/logs/2021-1-21-16:33
pretrained 
----------

----------

Opening arch config file /home/user/workspace/SalsaNext/salsanext.yml
Opening data config file config/labels/semantic-kitti.yaml
No pretrained directory found.
Copying files to /home/user/workspace/SalsaNext/logs/2021-1-21-16:33 for further reference.
Sequences folder exists! Using sequences from /media/user/programfox2TB/dataset/semantic_kitti/dataset/sequences
parsing seq 00
parsing seq 01
parsing seq 02
parsing seq 03
parsing seq 04
parsing seq 05
parsing seq 06
parsing seq 07
parsing seq 09
parsing seq 10
Using 19130 scans from sequences [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
Sequences folder exists! Using sequences from /media/user/programfox2TB/dataset/semantic_kitti/dataset/sequences
parsing seq 08
Using 4071 scans from sequences [8]
Loss weights from content:  tensor([  0.0000,  22.9317, 857.5627, 715.1100, 315.9618, 356.2452, 747.6170,
        887.2239, 963.8915,   5.0051,  63.6247,   6.9002, 203.8796,   7.4802,
         13.6315,   3.7339, 142.1462,  12.6355, 259.3699, 618.9667])
...
Lr: 4.182e-06 | Update: 2.258e-01 mean,4.181e-01 std | Epoch: [0][0/2391] | Time 6.548 (6.548) | Data 0.692 (0.692) | Loss 4.0624 (4.0624) | acc 0.050 (0.050) | IoU 0.015 (0.015) | [30 days, 1:16:33]
Lr: 4.601e-05 | Update: 2.045e-02 mean,3.893e-02 std | Epoch: [0][10/2391] | Time 1.158 (1.655) | Data 0.036 (0.100) | Loss 4.1408 (4.1349) | acc 0.044 (0.046) | IoU 0.013 (0.013) | [7 days, 6:48:44]
Lr: 8.783e-05 | Update: 1.056e-02 mean,2.059e-02 std | Epoch: [0][20/2391] | Time 1.308 (1.434) | Data 0.037 (0.070) | Loss 4.1282 (4.1354) | acc 0.041 (0.045) | IoU 0.014 (0.013) | [6 days, 5:48:37]
Lr: 1.297e-04 | Update: 6.514e-03 mean,1.258e-02 std | Epoch: [0][30/2391] | Time 1.165 (1.351) | Data 0.038 (0.059) | Loss 4.1375 (4.1366) | acc 0.049 (0.046) | IoU 0.014 (0.013) | [5 days, 20:30:02]
Lr: 1.715e-04 | Update: 6.290e-03 mean,1.245e-02 std | Epoch: [0][40/2391] | Time 1.175 (1.316) | Data 0.036 (0.054) | Loss 4.1102 (4.1336) | acc 0.050 (0.047) | IoU 0.015 (0.014) | [5 days, 16:27:57]
```


 * [Eval script](eval.sh) (you might need to chmod +x the file)
   * We have the following options:
     * ```-d [String]```: Path to the dataset
     * ```-p [String]```: Path to save label predictions
     * ``-m [String]``: Path to the location of saved model
     * ``-s [String]``: Eval on Validation or Train (standard eval on both separately)
     * ```-u [String]```: If you want to infer using an Uncertainty model (default ```false```)
     * ```-c [Int]```: Number of MC sampling to do (default ```30```)
     * ```-g [Int]```: GPU ID (default ```x```)
     * ```-n [Int]```: what is this? (default ```xx```)
   * If you want to infer&evaluate a model that you saved to ````/home/user/workspace/SalsaNext/pretrained```` and you
   want to infer and eval only the validation and save the label prediction to ```/home/user/workspace/SalsaNext/pred```:
     * <del> ```./eval.sh -d /dataset -p /pred -m /salsanext/logs/[the desired run] -s validation -n salsanext``` </del>
     * ```./eval.sh -d /home/user/workspace/dataset/semantic_kitti/dataset/ -p /home/user/workspace/SalsaNext/pred -m /home/user/workspace/SalsaNext/pretrained -s valid -n salsanext -c 30 -g 0```
     
  * [Visualization](train/tasks/semantic/visualize.py)
    * 
    
### Pretrained Model download
<del> [SalsaNext](https://cutt.ly/bpadjGj) </del>

One can easily download the author provided pretrained weights and our in-house trained model (150 epoch) by running
```
SalsaNext$./get_pretrained_models.sh
```
This script will download two models from this repo and store them under `SalsaNext/models/pretrained` and `SalsaNext/models/first_trained`.

     
### Disclamer

We based our code on [RangeNet++](https://github.com/PRBonn/lidar-bonnetal), please go show some support!
 

### Citation

```
@misc{cortinhal2020salsanext,
    title={SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving},
    author={Tiago Cortinhal and George Tzelepis and Eren Erdal Aksoy},
    year={2020},
    eprint={2003.03653},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

