# Attention-Mamba
Official Implementation of Attention-Mamba: A Mamba-Enhanced Multi-Scale Parallel Inference Network for Medical Image Segmentation (2026).

## Usage

### 0. To be noted:

- We will gradually optimize the code to make it more readable and standardized.

- For validation on the Synapse and ACDC test sets, each sequence is treated as a batch of 2D inputs, and sliding-window inference is applied. So, GPU memory consumption during validation depends on the sequence length and in-plane resolution of each case, ranging from over 24 GB to as much as 60 GB on the Synapse dataset. Please ensure that sufficient GPU memory is available.

### 1. Download pre-trained Resnet models

Download the pre-trained Resnet models and put them into the folder 'datasets_preprocessed'.

- resnet18-deep-stem:[link](https://drive.google.com/file/d/1q1VBV37acIte0GynoS054BWfwwdx1NiZ/view?usp=sharing)

### 2. Prepare data

Download our preprocessed datasets and put it into the folder 'preprocessed_data'.

- Download the [Synapse dataset](https://drive.google.com/file/d/1efK3jAh38_S_0M2MKWzbQHGjyAxyvVe_/view?usp=sharing). 

- Download the [ACDC dataset](https://drive.google.com/file/d/1zTOQH4nVbPMl6Ck8bEmQ6Fz9otM0JD2j/view?usp=sharing).

- Download the [ISIC2018+PH2 dataset](https://drive.google.com/file/d/1LjRRF94c-JhCGfBjjKfKQfBGWcZGn5Hl/view?usp=sharing). 

### 3. Environment

- We trained our model on one NVIDIA A800 (80GB) with the CUDA 11.7 and CUDNN 8500. Python 3.8.13. PyTorch 2.0.1.

- Environment for Mamba: pip install mamba-ssm

- Please refer to 'requirements.txt' for other dependencies.

### 4. Test our trained model 

#### 1) Synapse dataset

- Download the trained model:[link](https://drive.google.com/drive/folders/1R342qkJUHctw6KTcOXDw464r9kYtmHit?usp=sharing). This trained model reached 85.62% DSC on the Synapse dataset. 

- Put 'best_trained_model_train_main_loss.pth' into this folder: 'results\Datase1_Synapse_8classes_My_Attention_Mamba_2D_My_Attention_Mamba_2D_epo300_bs24_lr0.0005_wd0.01_calc_MultipleOutput_CE_and_Dice_Loss_Ex_PreTrained_AdamW_seed1290\fold_0\Train\Model_best'. Run the following order:

```bash
cd Attention-Mamba
```

- Test:

```bash
CUDA_VISIBLE_DEVICES=1 python run_main.py --dataset='Datase1_Synapse_8classes' --val_interval 100 --loss_weights 1.0 1.0 1.0 1.0 1.0 0.25 0.25 0.25 0.25 --img_size 224 224 --inference_scales 1.0 --base_lr 5e-4 --betas 0.9 0.999 --T_max 100 --eta_min 1e-5 --weight_decay 1e-2 --max_epochs 300 --batch_size 24 --seed 1290 --fold_name 0 --marker='Ex_PreTrained_AdamW' --if_inferece_testset='false' --if_calc_metrics_test='false' --load_model_type='load_best_train_main_loss_model' --if_training='false'
```

- Save results to Excel:

```bash
CUDA_VISIBLE_DEVICES=1 python run_main.py --dataset='Datase1_Synapse_8classes' --val_interval 100 --loss_weights 1.0 1.0 1.0 1.0 1.0 0.25 0.25 0.25 0.25 --img_size 224 224 --base_lr 5e-4 --betas 0.9 0.999 --T_max 100 --eta_min 1e-5 --weight_decay 1e-2 --max_epochs 300 --batch_size 24 --seed 1290 --fold_name 0 --marker='Ex_PreTrained_AdamW' --if_training='false' --if_inferece_valset='false' --if_inferece_testset='false' --if_calc_metrics_val='false' --if_calc_metrics_test='false' --fold_ids_select 0 --if_avg_5_folds_val='true' --load_model_type='load_best_train_main_loss_model'
```

#### 2) ACDC dataset

- Download the trained model:[link](https://drive.google.com/drive/folders/1odOE44vo5nF6WrTFasPpaLot3X8amcXJ?usp=sharing). This trained model reached 91.94% DSC and 92.03% DSC on the validation and test sets, respectively. 

- Put 'best_trained_model_val_acc.pth' into this folder: 'results\Dataset2_ACDC_My_Attention_Mamba_2D_My_Attention_Mamba_2D_epo150_bs24_lr0.0005_wd0.05_calc_MultipleOutput_CE_and_Dice_Loss_Ex_PreTrained_AdamW_seed1290\fold_1\Train\Model_best'. Run the following order:

```bash
cd Attention-Mamba
```

- Test:

```bash
CUDA_VISIBLE_DEVICES=1 python run_main.py --dataset='Dataset2_ACDC' --val_interval 1 --loss_weights 1.0 1.0 1.0 1.0 1.0 0.25 0.25 0.25 0.25 --img_size 192 192 --inference_scales 1.0 --base_lr 5e-4 --betas 0.9 0.999 --T_max 50 --eta_min 1e-5 --weight_decay 5e-2 --max_epochs 150 --batch_size 24 --seed 1290 --fold_name 1 --marker='Ex_PreTrained_AdamW' --if_training='false'
```

- Save results to Excel:

```bash
CUDA_VISIBLE_DEVICES=1 python run_main.py --dataset='Dataset2_ACDC' --val_interval 1 --loss_weights 1.0 1.0 1.0 1.0 1.0 0.25 0.25 0.25 0.25 --img_size 192 192 --base_lr 5e-4 --betas 0.9 0.999 --T_max 50 --eta_min 1e-5 --weight_decay 5e-2 --max_epochs 150 --batch_size 24 --seed 1290 --fold_name 0 --marker='Ex_PreTrained_AdamW' --if_training='false' --if_inferece_valset='false' --if_inferece_testset='false' --if_calc_metrics_val='false' --if_calc_metrics_test='false' --fold_ids_select 1 --if_avg_5_folds_val='true' --if_avg_5_folds_test='true'
```

#### 3) ISIC2018+PH2 dataset

- Download the trained model:[link](https://drive.google.com/drive/folders/1B8-9UmhWYu8wff267Rm9JI80ss1q2P3g?usp=sharing). This trained model reached 90.12% DSC and 91.56% DSC on the ISIC2018 and PH2 datasets, respectively. 

- Put 'best_trained_model_train_main_loss.pth' into this folder: 'results\Dataset3_ISIC2018_PH2_My_Attention_Mamba_2D_My_Attention_Mamba_2D_epo150_bs12_lr0.00015_wd0.01_calc_MultipleOutput_BCE_and_Dice_loss_Ex_PreTrained_AdamW_seed1290\fold_1\Train\Model_best'. Run the following order:

```bash
cd Attention-Mamba
```

- Test:

```bash
CUDA_VISIBLE_DEVICES=1 python run_main.py --dataset='Dataset3_ISIC2018_PH2' --val_interval 100 --loss_weights 1.0 1.0 1.0 1.0 1.0 0.25 0.25 0.25 0.25 --loss_calculator_name='calc_MultipleOutput_BCE_and_Dice_loss' --if_mask_to_long_type='false' --img_size 256 256 --inference_scales 1.0 --base_lr 1.5e-4 --betas 0.9 0.999 --T_max 50 --eta_min 1e-5 --weight_decay 1e-2 --max_epochs 150 --batch_size 12 --augmentation_type='ISIC_Aug_V4' --seed 1290 --fold_name 1 --marker='Ex_PreTrained_AdamW' --load_model_type='load_best_train_main_loss_model' --if_training='false'
```

- Save results to Excel:

```bash
CUDA_VISIBLE_DEVICES=1 python run_main.py --dataset='Dataset3_ISIC2018_PH2' --val_interval 100 --loss_weights 1.0 1.0 1.0 1.0 1.0 0.25 0.25 0.25 0.25 --loss_calculator_name='calc_MultipleOutput_BCE_and_Dice_loss' --if_mask_to_long_type='false' --img_size 256 256 --base_lr 1.5e-4 --betas 0.9 0.999 --T_max 50 --eta_min 1e-5 --weight_decay 1e-2 --max_epochs 150 --batch_size 12 --augmentation_type='ISIC_Aug_V4' --seed 1290 --fold_name 1 --marker='Ex_PreTrained_AdamW' --if_training='false' --if_inferece_valset='false' --if_inferece_testset='false' --if_calc_metrics_val='false' --if_calc_metrics_test='false' --fold_ids_select 1 --if_avg_5_folds_val='false' --if_avg_5_folds_test='true' --load_model_type='load_best_train_main_loss_model'
```

### 5. Train/Test by yourself

#### 1) Synapse dataset

```bash
cd Attention-Mamba
```

- Run the train and test script.

```bash
CUDA_VISIBLE_DEVICES=1 python run_main.py --dataset='Datase1_Synapse_8classes' --val_interval 100 --loss_weights 1.0 1.0 1.0 1.0 1.0 0.25 0.25 0.25 0.25 --img_size 224 224 --inference_scales 1.0 --base_lr 5e-4 --betas 0.9 0.999 --T_max 100 --eta_min 1e-5 --weight_decay 1e-2 --max_epochs 300 --batch_size 24 --seed 1290 --fold_name 0 --marker='Ex_PreTrained_AdamW' --if_inferece_testset='false' --if_calc_metrics_test='false' --load_model_type='load_best_train_main_loss_model'
```

- Save results to Excel.

```bash
CUDA_VISIBLE_DEVICES=1 python run_main.py --dataset='Datase1_Synapse_8classes' --val_interval 100 --loss_weights 1.0 1.0 1.0 1.0 1.0 0.25 0.25 0.25 0.25 --img_size 224 224 --base_lr 5e-4 --betas 0.9 0.999 --T_max 100 --eta_min 1e-5 --weight_decay 1e-2 --max_epochs 300 --batch_size 24 --seed 1290 --fold_name 0 --marker='Ex_PreTrained_AdamW' --if_training='false' --if_inferece_valset='false' --if_inferece_testset='false' --if_calc_metrics_val='false' --if_calc_metrics_test='false' --fold_ids_select 0 --if_avg_5_folds_val='true' --load_model_type='load_best_train_main_loss_model'
```

#### 2) ACDC dataset

```bash
cd Attention-Mamba
```

- Run the train and test script.

```bash
CUDA_VISIBLE_DEVICES=1 python run_main.py --dataset='Dataset2_ACDC' --val_interval 1 --loss_weights 1.0 1.0 1.0 1.0 1.0 0.25 0.25 0.25 0.25 --img_size 192 192 --inference_scales 1.0 --base_lr 5e-4 --betas 0.9 0.999 --T_max 50 --eta_min 1e-5 --weight_decay 5e-2 --max_epochs 150 --batch_size 24 --seed 1290 --fold_name 1 --marker='Ex_PreTrained_AdamW'
```

- Save results to Excel.

```bash
CUDA_VISIBLE_DEVICES=1 python run_main.py --dataset='Dataset2_ACDC' --val_interval 1 --loss_weights 1.0 1.0 1.0 1.0 1.0 0.25 0.25 0.25 0.25 --img_size 192 192 --base_lr 5e-4 --betas 0.9 0.999 --T_max 50 --eta_min 1e-5 --weight_decay 5e-2 --max_epochs 150 --batch_size 24 --seed 1290 --fold_name 0 --marker='Ex_PreTrained_AdamW' --if_training='false' --if_inferece_valset='false' --if_inferece_testset='false' --if_calc_metrics_val='false' --if_calc_metrics_test='false' --fold_ids_select 1 --if_avg_5_folds_val='true' --if_avg_5_folds_test='true'
```

#### 3) ISIC2018+PH2 dataset

```bash
cd Attention-Mamba
```

- Run the train and test script.

```bash
CUDA_VISIBLE_DEVICES=1 python run_main.py --dataset='Dataset3_ISIC2018_PH2' --val_interval 100 --loss_weights 1.0 1.0 1.0 1.0 1.0 0.25 0.25 0.25 0.25 --loss_calculator_name='calc_MultipleOutput_BCE_and_Dice_loss' --if_mask_to_long_type='false' --img_size 256 256 --inference_scales 1.0 --base_lr 1.5e-4 --betas 0.9 0.999 --T_max 50 --eta_min 1e-5 --weight_decay 1e-2 --max_epochs 150 --batch_size 12 --augmentation_type='ISIC_Aug_V4' --seed 1290 --fold_name 1 --marker='Ex_PreTrained_AdamW' --load_model_type='load_best_train_main_loss_model'
```

- Save results to Excel.

```bash
CUDA_VISIBLE_DEVICES=1 python run_main.py --dataset='Dataset3_ISIC2018_PH2' --val_interval 100 --loss_weights 1.0 1.0 1.0 1.0 1.0 0.25 0.25 0.25 0.25 --loss_calculator_name='calc_MultipleOutput_BCE_and_Dice_loss' --if_mask_to_long_type='false' --img_size 256 256 --base_lr 1.5e-4 --betas 0.9 0.999 --T_max 50 --eta_min 1e-5 --weight_decay 1e-2 --max_epochs 150 --batch_size 12 --augmentation_type='ISIC_Aug_V4' --seed 1290 --fold_name 1 --marker='Ex_PreTrained_AdamW' --if_training='false' --if_inferece_valset='false' --if_inferece_testset='false' --if_calc_metrics_val='false' --if_calc_metrics_test='false' --fold_ids_select 1 --if_avg_5_folds_val='false' --if_avg_5_folds_test='true' --load_model_type='load_best_train_main_loss_model'
```

## Reference
* [MultiTrans](https://github.com/Yanhua-Zhang/MultiTrans-extension)

## Citations

```bibtex

xxx

```
