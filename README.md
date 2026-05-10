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

- Download the trained model:[link](https://drive.google.com/drive/folders/1R342qkJUHctw6KTcOXDw464r9kYtmHit?usp=sharing). This trained model reached 85.62% DSC on the Synapse dataset. 

- Put 'best_trained_model_train_main_loss.pth' into this folder: 'results\Datase1_Synapse_8classes_My_Attention_Mamba_2D_My_Attention_Mamba_2D_epo300_bs24_lr0.0005_wd0.01_calc_MultipleOutput_CE_and_Dice_Loss_Ex_PreTrained_AdamW_seed1290\fold_0\Train\Model_best'. Run the following order:

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
