import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import torch
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F


# save feature functions for MRI or CT 


# -----------------------------------------------------------------------------
def fun_GT_Pre_Overlap(GT_tensor, Pre_tensor, patient_name, slice_num, marker, save_path):

    # GT_tensor:  1, 1, h, w
    # Pre_tensor: 1, 1, h, w

    path_color_GT_Pre_Overlap = './utils/1_color_maps_Synapse/colors_GT_Pre_Overlap.txt'
    colors_GT_Pre_Overlap = np.loadtxt(path_color_GT_Pre_Overlap).astype('uint8')  # array

    weight_img = 0.75
    weight_mask = 0.5

    #---------------------------------------------
    saved_img_path = save_path + '/' + patient_name + '/' + str(slice_num) + '/Input_Img_Gray.png'
    if not os.path.exists(saved_img_path):
        raise ValueError('This file is lost:' + saved_img_path)
    img_cv2 = cv2.imread(saved_img_path)

    # saved_GT_path = save_path + '/' + patient_name + '/' + str(slice_num) + '/GT.png'
    # if not os.path.exists(saved_GT_path):
    #     raise ValueError('This file is lost:' + saved_GT_path)
    # GT_cv2 = cv2.imread(saved_GT_path)

    #---------------------------------------------
    Pre_up_tensor = F.interpolate(Pre_tensor.float(), GT_tensor.size(), mode='nearest')

    GT_np = GT_tensor.squeeze().detach().cpu().numpy()

    Pre_up_np = Pre_up_tensor.squeeze().detach().cpu().numpy()

    # -------------------------------------------
    # GT_Pre_Overlap: id to color

    GT_minus_Pre = GT_np - Pre_up_np

    GT_Pre_Overlap_case_np_slice = np.zeros(np.shape(GT_minus_Pre))

    GT_Pre_Overlap_case_np_slice[GT_minus_Pre==0] = 1  # right pred + backgorund
    GT_Pre_Overlap_case_np_slice[GT_np==0] = 0  # all backgorund to 0
    GT_Pre_Overlap_case_np_slice[GT_minus_Pre>0] = 2  # wrong pred
    GT_Pre_Overlap_case_np_slice[GT_minus_Pre<0] = 2  # wrong pred

    GT_Pre_Overlap_case_np_slice_color_Image = colorize(GT_Pre_Overlap_case_np_slice, colors_GT_Pre_Overlap)  # id2color

    #---------------------------------------------
    # save path

    path_folder = save_path + '/' + patient_name + '/' + str(slice_num)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)  
    path_GT_Pre_Overlap = path_folder + '/' + marker + '_Pre_color.png'

    GT_Pre_Overlap_case_np_slice_color_Image.save(path_GT_Pre_Overlap)
    GT_Pre_Overlap_case_np_slice_color_cv = cv2.imread(path_GT_Pre_Overlap)  # Image to cv2 by saving and reading

    # -------------------------------------------
    # save overlapped img

    overlap_img = cv2.addWeighted(img_cv2, weight_img, GT_Pre_Overlap_case_np_slice_color_cv, weight_mask, 0)

    path_folder = save_path + '/' + patient_name + '/' + str(slice_num) 
    path_overlap_img = path_folder + '/' + marker + '_Pre_over_Img.png'

    cv2.imwrite(path_overlap_img, overlap_img)



# -----------------------------------------------------------------------------
def fun_Heatmap_2D(tensor_in, patient_name, slice_num, marker, save_path, figsize=None, if_normalized=False, if_show_axis=False):
    

    feature_map_np = tensor_in.detach().cpu().numpy()

    feature_map_np_norm = fun_norm_to_0_1(feature_map_np)

    if if_normalized:
        fea_show = feature_map_np_norm
    else:
        fea_show = feature_map_np

    #---------------------------------------------
    # Creating a Custom Gradient
    # max_value = tensor_in.min()
    # max_value = tensor_in.max()

    # Define colors: (0 is lowest value, 1 is highest)
    # colors = ["black", "darkgreen", "red"] 
    # colors = ["black", "green", "red"]
    colors = ["black", "#3b4cc0", "#b40426"]
    # colors = ["black", "yellow", "red"]
    nodes = [0.0, 0.5, 1.0]
    my_cmap = LinearSegmentedColormap.from_list("my_list", list(zip(nodes, colors)))    

    #---------------------------------------------

    # fig, ax = plt.subplots(figsize=figsize)
    fig, ax = plt.subplots()
    # im = ax.imshow(fea_show, cmap=my_cmap, vmin=0.0, vmax=1.0)
    im = ax.imshow(fea_show, cmap='gray', vmin=0.0, vmax=1.0)   # cmap="viridis"  'Greys'

    cbar = plt.colorbar(im, pad=0.05)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['0', '0.5', '1.0'])
    cbar.set_label('Attention Intensity', fontsize=5, color='black')
    cbar.ax.tick_params(labelsize=10, colors='black') # Change tick color
    
    if if_show_axis:
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
        ax.set_yticklabels(['0', 'Sp', 'Ki(R)', 'Ki(L)', 'Ga', 'Li', 'St', 'Ao', 'Pa'])
        ax.tick_params(axis='y', colors='darkblue', labelsize=2.5, rotation=45)
        ax.get_xaxis().set_visible(False)

    else:
        plt.axis('off')

    #---------------------------------------------
    path_folder = save_path + '/' + patient_name + '/' + str(slice_num)

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)  

    #---------------------------------------------
    # Save
    path_save = path_folder + '/' + marker + '.png'

    plt.savefig(path_save, bbox_inches='tight', dpi=300)

    # close
    # plt.show()
    plt.cla()   # Clear the current axes
    plt.clf()   # Clear the entire figure
    plt.close(fig) # Completely close the figure window and free memory

# -----------------------------------------------------------------------------
def fun_Heatmap_1D(tensor_in, patient_name, slice_num, marker, save_path):
    
    #---------------------------------------------
    # Creating a Custom Gradient
    # max_value = tensor_in.min()
    # max_value = tensor_in.max()

    # Define colors: (0 is lowest value, 1 is highest)
    # colors = ["black", "darkgreen", "red"] 
    # colors = ["black", "green", "red"]
    colors = ["black", "#3b4cc0", "#b40426"]
    nodes = [0.0, 0.5, 1.0]
    my_cmap = LinearSegmentedColormap.from_list("my_list", list(zip(nodes, colors)))    

    #---------------------------------------------
    # plt.figure(figsize=(15, 2))
    # plt.imshow(tensor_in.detach().cpu().numpy(), aspect='auto', cmap='RdBu')
    # plt.colorbar(label='Value Intensity')
    # plt.title(f"Heatmap of 1D Tensor")
    # plt.axis('off') # Optional: hides the axis numbers

    fig, ax = plt.subplots(figsize=(15, 2))
    im = ax.imshow(tensor_in.detach().cpu().numpy(), aspect='auto', cmap=my_cmap)

    cbar = plt.colorbar(im, pad=0.05)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['0', '0.5', '1.0'])
    cbar.set_label('Activation Intensity', fontsize=12, color='blue')
    cbar.ax.tick_params(labelsize=10, colors='red') # Change tick color

    #---------------------------------------------
    path_folder = save_path + '/' + patient_name + '/' + str(slice_num)

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)  

    #---------------------------------------------
    # Save
    path_save = path_folder + '/' + marker + '.png'

    plt.savefig(path_save, bbox_inches='tight', dpi=300)
    # plt.show()



# -----------------------------------------------------------------------------
def fun_Bar_Chart(tensor_in, patient_name, slice_num, marker, save_path):
    
    #---------------------------------------------
    plt.figure(figsize=(10, 4))
    # We flatten to (W,) for plotting
    plt.plot(tensor_in.view(-1).detach().cpu().numpy()) 

    plt.title("Tensor Values Bar Chart")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)

    #---------------------------------------------
    path_folder = save_path + '/' + patient_name + '/' + str(slice_num)

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)  

    #---------------------------------------------
    # Save
    path_save = path_folder + '/' + marker + '.png'

    plt.savefig(path_save)
    # plt.show()


# -----------------------------------------------------------------------------
def colorize(gray, palette):

    # array ---> Img
    # This conversion from 'L' (grayscale) to 'P' creates a palette where index 0 = black (0,0,0), index 1 = slightly lighter gray, etc., up to index 255 = white (255,255,255).
    color = Image.fromarray(gray.astype(np.uint8)).convert(
        'P')   
    
    # applies a custom color palette to an already existing 'P' (palette) mode image.
    color.putpalette(palette)

    return color


# -----------------------------------------------------------------------------
# GT of Synapse

def fun_save_GT_Synapse(feature_map, patient_name, slice_num, marker, save_path):

    path_colors = './utils/1_color_maps_Synapse/Synapse_colors.txt'
    colors = np.loadtxt(path_colors).astype('uint8')  # 19*3 array

    feature_map_np = feature_map.cpu().numpy()

    #---------------------------------------------
    path_folder = save_path + '/' + patient_name + '/' + str(slice_num)

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)  

    #---------------------------------------------
    # Save
    path_GT = path_folder + '/' + marker + '.png'

    color_GT = colorize(feature_map_np, colors)
    color_GT.save(path_GT) 


# -----------------------------------------------------------------------------
def fun_Z_score_norm(img_np, do_clip=False):

    if do_clip:
        lower_bnd = np.percentile(img_np, 100-99.5)
        upper_bnd = np.percentile(img_np, 99.5)
        img_np_clip = np.clip(img_np, lower_bnd, upper_bnd)
    else:
        img_np_clip = img_np

    # perform z-score normalization
    mean = np.mean(img_np_clip)
    std = np.std(img_np_clip)

    if std > 0:
        img_np_normalized = (img_np_clip - mean) / std
    else:
        img_np_normalized = img_np_clip * 0.

    return img_np_normalized


# -----------------------------------------------------------------------------
def fun_norm_to_0_1(img_np):

    min_value = img_np.min()
    max_value = img_np.max()
    
    if (max_value - min_value) > 0:
        # normalized to 0 ~ 1
        img_np_normalized = (img_np - min_value) / (max_value - min_value)

    else:
        img_np_normalized = img_np * 0.

    return img_np_normalized


# -----------------------------------------------------------------------------
# save feature maps
# normalized by fun_Z_score_norm

def fun_save_feature_maps(feature_map, patient_name, slice_num, marker, save_path, if_normalized=True, if_save_RGB=False):

    feature_map_np = feature_map.cpu().numpy()

    feature_map_np_norm = fun_Z_score_norm(feature_map_np)

    feature_map_np_norm_Gary = feature_map_np_norm*255

    #---------------------------------------------
    path_folder = save_path + '/' + patient_name + '/' + str(slice_num)

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)  

    #---------------------------------------------
    # Gray
    # path_Gary = save_path + '/' + patient_name + '_' + str(slice_num) + '_' + marker + '_Gray.png'
    path_Gary = path_folder + '/' + marker + '_Gray.png'

    if if_normalized:
        cv2.imwrite(path_Gary, feature_map_np_norm_Gary)  # normalized
    else:
        cv2.imwrite(path_Gary, feature_map_np*255)          # not normalized

    if if_save_RGB:
        #---------------------------------------------
        # Gray to JET
        img_Gary = cv2.imread(path_Gary, cv2.IMREAD_GRAYSCALE)
        feature_map_np_norm_JET = cv2.applyColorMap(img_Gary, cv2.COLORMAP_JET)
        # path_JET = save_path + '/' + patient_name + '_' + str(slice_num) + '_' + marker + '_JET.png'
        path_JET = path_folder + '/' + marker + '_JET.png'
        cv2.imwrite(path_JET, feature_map_np_norm_JET)

        #---------------------------------------------
        # Gray to HSV
        # img_Gary = cv2.imread(path_Gary, cv2.IMREAD_GRAYSCALE)
        feature_map_np_norm_HSV = cv2.applyColorMap(img_Gary, cv2.COLORMAP_HSV)
        # path_HSV = save_path + '/' + patient_name + '_' + str(slice_num) + '_' + marker + '_HSV.png'
        path_HSV = path_folder + '/' + marker + '_HSV.png'
        cv2.imwrite(path_HSV, feature_map_np_norm_HSV)

        #---------------------------------------------
        # Gray to HSV
        # img_Gary = cv2.imread(path_Gary, cv2.IMREAD_GRAYSCALE)
        feature_map_np_norm_HOT = cv2.applyColorMap(img_Gary, cv2.COLORMAP_HOT)
        # path_HOT = save_path + '/' + patient_name + '_' + str(slice_num) + '_' + marker + '_HOT.png'
        path_HOT = path_folder + '/' + marker + '_HOT.png'
        cv2.imwrite(path_HOT, feature_map_np_norm_HOT)


# -----------------------------------------------------------------------------
# save feature maps
# normalized by fun_norm_to_0_1

def fun_save_feature_maps_V2(feature_map, patient_name, slice_num, marker, save_path, if_normalized=True, if_save_RGB=False, if_save_Heatmap=False, if_show_axis=False):

    feature_map_np = feature_map.cpu().numpy()

    feature_map_np_norm = fun_norm_to_0_1(feature_map_np)

    feature_map_np_norm_Gary = feature_map_np_norm*255

    #---------------------------------------------
    path_folder = save_path + '/' + patient_name + '/' + str(slice_num)

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)  

    #---------------------------------------------
    # Gray
    # path_Gary = save_path + '/' + patient_name + '_' + str(slice_num) + '_' + marker + '_Gray.png'
    path_Gary = path_folder + '/' + marker + '_Gray.png'

    if if_normalized:
        cv2.imwrite(path_Gary, feature_map_np_norm_Gary)  # normalized
    else:
        cv2.imwrite(path_Gary, feature_map_np*255)          # not normalized

    if if_save_RGB:
        #---------------------------------------------
        # Gray to JET
        img_Gary = cv2.imread(path_Gary, cv2.IMREAD_GRAYSCALE)
        feature_map_np_norm_JET = cv2.applyColorMap(img_Gary, cv2.COLORMAP_JET)
        # path_JET = save_path + '/' + patient_name + '_' + str(slice_num) + '_' + marker + '_JET.png'
        path_JET = path_folder + '/' + marker + '_JET.png'
        cv2.imwrite(path_JET, feature_map_np_norm_JET)

        #---------------------------------------------
        # Gray to HSV
        # img_Gary = cv2.imread(path_Gary, cv2.IMREAD_GRAYSCALE)
        feature_map_np_norm_HSV = cv2.applyColorMap(img_Gary, cv2.COLORMAP_HSV)
        # path_HSV = save_path + '/' + patient_name + '_' + str(slice_num) + '_' + marker + '_HSV.png'
        path_HSV = path_folder + '/' + marker + '_HSV.png'
        cv2.imwrite(path_HSV, feature_map_np_norm_HSV)

        #---------------------------------------------
        # Gray to HSV
        # img_Gary = cv2.imread(path_Gary, cv2.IMREAD_GRAYSCALE)
        feature_map_np_norm_HOT = cv2.applyColorMap(img_Gary, cv2.COLORMAP_HOT)
        # path_HOT = save_path + '/' + patient_name + '_' + str(slice_num) + '_' + marker + '_HOT.png'
        path_HOT = path_folder + '/' + marker + '_HOT.png'
        cv2.imwrite(path_HOT, feature_map_np_norm_HOT)

    if if_save_Heatmap:
        fun_Heatmap_2D(tensor_in = feature_map, patient_name = patient_name, slice_num = slice_num, marker = marker + '_Heatmap', save_path = save_path, if_normalized = if_normalized, if_show_axis = if_show_axis)
        
# -----------------------------------------------------------------------------
def fun_save_visualize_grid_vectors(grid, patient_name, slice_num, marker, save_path):
    # input is the warp grid

    if torch.is_tensor(grid):
        grid_np = grid.detach().cpu().numpy()

    H, W, _ = grid_np.shape
    
    # Create regular grid for comparison
    y_reg, x_reg = np.mgrid[0:H, 0:W]

    x_reg = (x_reg / (W-1)) * 2 - 1  # Normalize to [-1, 1]
    y_reg = (y_reg / (H-1)) * 2 - 1
    
    # Calculate displacement vectors
    dx = grid_np[:, :, 0] - x_reg
    dy = grid_np[:, :, 1] - y_reg

    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    # --------------------------------------
    fig, ax = plt.subplots()
    
    # Plot vector field
    ax.quiver(X, Y, dx, dy, scale=1, scale_units='xy', angles='xy', color='red')
    
    # --------------------------------------
    ax.set_xlim(-1, W)
    ax.set_xlabel('Width')
    ax.xaxis.set_major_locator(MultipleLocator(1))

    ax.set_ylim(-1, H)
    ax.set_ylabel('Height')
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # --------------------------------------
    ax.set_aspect('equal')
    ax.set_title('Grid Vector Field')
    ax.invert_yaxis()  # Match image coordinate system

    # --------------------------------------
    # set scale
    ax.tick_params(axis='both', labelsize=9)

    # --------------------------------------
    # grid
    ax.grid(True, linestyle=':', linewidth = 0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #---------------------------------------------
    path_folder = save_path + '/' + patient_name + '/' + str(slice_num)

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)  

    path = path_folder + '/' + marker + '_grid_vectors.png'

    plt.savefig(path, dpi=600)
    # plt.show()
    plt.close()


# -----------------------------------------------------------------------------
def fun_save_visualize_flow_field_vectors(grid, patient_name, slice_num, marker, save_path):
    # input is the learned flow field

    if torch.is_tensor(grid):
        grid_np = grid.detach().cpu().numpy()

    H, W, _ = grid_np.shape
    
    # Calculate displacement vectors
    dx = grid_np[:, :, 0]
    dy = grid_np[:, :, 1]

    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    # --------------------------------------
    fig, ax = plt.subplots()
    
    # Plot vector field
    ax.quiver(X, Y, dx, dy, scale=1, scale_units='xy', angles='xy', color='red')
    
    # --------------------------------------
    # ax.set_xlim(-1, W)
    ax.set_xlim(-4, W+4)
    ax.set_xlabel('Width')
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # ax.set_ylim(-1, H)
    ax.set_ylim(-4, H+4)
    ax.set_ylabel('Height')
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # --------------------------------------
    ax.set_aspect('equal')
    ax.set_title('Grid Vector Field')
    ax.invert_yaxis()  # Match image coordinate system

    # --------------------------------------
    # set scale
    ax.tick_params(axis='both', labelsize=9)

    # --------------------------------------
    # grid
    ax.grid(True, linestyle=':', linewidth = 0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #---------------------------------------------
    path_folder = save_path + '/' + patient_name + '/' + str(slice_num)

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)  

    path = path_folder + '/' + marker + '_grid_vectors.png'

    plt.savefig(path, dpi=600)
    # plt.show()
    plt.close()