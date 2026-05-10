# Attention-Mamba
Official Implementation of Attention-Mamba: A Mamba-Enhanced Multi-Scale Parallel Inference Network for Medical Image Segmentation (2026).

## Usage

### 0. To be noted:

- We will gradually optimize the code to make it more readable and standardized.

- 

### 1. Download pre-trained Resnet models

Download the pre-trained Resnet models and put them into the folder 'datasets_preprocessed'.

- resnet18-deep-stem:[link](https://drive.google.com/file/d/1q1VBV37acIte0GynoS054BWfwwdx1NiZ/view?usp=sharing)

### 2. Prepare data

Download our preprocessed datasets and put it into the folder 'preprocessed_data'.

- Download the [Synapse dataset](https://drive.google.com/file/d/1efK3jAh38_S_0M2MKWzbQHGjyAxyvVe_/view?usp=sharing). 

- Download the [ACDC dataset](https://drive.google.com/file/d/1zTOQH4nVbPMl6Ck8bEmQ6Fz9otM0JD2j/view?usp=sharing).

- Download the [ISIC2018+PH2 dataset](https://drive.google.com/file/d/1LjRRF94c-JhCGfBjjKfKQfBGWcZGn5Hl/view?usp=sharing). 

### 3. Environment

We trained our model on one NVIDIA A800 (80GB) with the CUDA 11.7 and CUDNN 8500.

- Python 3.8.13.

- PyTorch 2.0.1. 

- Please refer to 'requirements.txt' for other dependencies.

### 4. Test our trained model 

- Download the trained model:[link](https://drive.google.com/file/d/1DDqsDNoWuvn8Uy9H4_qfNNXGJQK917UA/view?usp=drive_link). This trained model reached 84.82% DSC and 12.66 mm HD on the Synapse dataset, without using sophisticated data augmentation methods. 

- Put 'epoch_149.pth' into this folder: 'Results/model_Trained/My_Model_Synapse224/Model/My_Model_pretrain_resnet50_Deep_V10_epo150_bs24_lr0.1_224_s1290'. Run the following order:

```bash
cd MultiTrans_extension
```

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --Model_Name My_Model --bran_weights 0.4 0.3 0.2 0.1 --base_lr 0.1 --branch_depths 5 5 5 5 5 --branch_in_channels 256 256 256 256 256 --branch_key_channels 32 32 32 32 32 --seed 1290
```

### 5. Train/Test by yourself

```bash
cd MultiTrans_extension
```

- Run the train script.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --Model_Name My_Model --bran_weights 0.4 0.3 0.2 0.1 --base_lr 0.1 --branch_depths 5 5 5 5 5 --branch_in_channels 256 256 256 256 256 --branch_key_channels 32 32 32 32 32 --seed 1290
```

- Run the test script.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --Model_Name My_Model --bran_weights 0.4 0.3 0.2 0.1 --base_lr 0.1 --branch_depths 5 5 5 5 5 --branch_in_channels 256 256 256 256 256 --branch_key_channels 32 32 32 32 32 --seed 1290
```

## Reference
* [TransUNet](https://github.com/Beckschen/TransUNet)

## Citations

```bibtex

xxx

```
