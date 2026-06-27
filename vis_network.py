import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn import datasets
from nilearn import surface 

def visualize_nifti_results(data_dir='./assets/SIM_data', output_dir='./assets/SIM_Figures', max_plots=None, target_networks=None):
    """
    遍历 NIfTI 文件并生成可视化。
    
    参数:
    - max_plots (int): 最大绘制数量。如果为 None，则不限制。
    - target_networks (list): 包含目标网络序号或名称的字符串列表，例如 ['Motif_01', 'Motif_07']。
    """
    os.makedirs(output_dir, exist_ok=True)
    nii_files = glob.glob(os.path.join(data_dir, '*.nii.gz'))
    
    # 强制排序，确保按照 Motif_01, Motif_02 的顺序处理
    nii_files.sort()
    
    if not nii_files:
        print(f"❌ 在 {data_dir} 目录下没有找到任何 .nii.gz 文件！")
        return
        
    # ==========================================
    # 🎯 核心升级：按需过滤文件列表
    # ==========================================
    # 1. 如果指定了特定网络，则先进行精准筛选
    if target_networks:
        filtered_files = []
        for f in nii_files:
            # 如果文件名中包含目标列表里的任意一个词，就保留它
            if any(target in f for target in target_networks):
                filtered_files.append(f)
        nii_files = filtered_files
        print(f"🎯 已开启特定目标过滤，匹配到 {len(nii_files)} 个网络。")
        
    # 2. 如果指定了最大数量，则截断列表
    if max_plots and len(nii_files) > max_plots:
        nii_files = nii_files[:max_plots]
        print(f"⏸️ 已限制最大绘图数量为 {max_plots} 个。")

    print(f"🔍 最终准备绘制 {len(nii_files)} 个脑网络文件...\n")
    # ==========================================

    # 下载 3D 皮层模板
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

    for img_path in nii_files:
        file_name = os.path.basename(img_path)
        network_name = file_name.replace('.nii.gz', '')
        print(f"🎨 正在绘制: {network_name}")
        
        # 加载数据与计算阈值
        img = nib.load(img_path)
        data = img.get_fdata()
        threshold = np.percentile(np.abs(data[data != 0]), 90)

        # 1: 玻璃脑 (Glass Brain)
        fig_glass = plt.figure(figsize=(10, 4))
        plotting.plot_glass_brain(
            img_path, threshold=threshold, colorbar=True, display_mode='lyrz', 
            title=f"{network_name} (Glass Brain)", figure=fig_glass
        )
        fig_glass.savefig(os.path.join(output_dir, f"{network_name}_glass.png"), dpi=300)
        plt.close(fig_glass)

        # 2: 正交切片 (Orthogonal Slices)
        fig_stat = plt.figure(figsize=(10, 4))
        plotting.plot_stat_map(
            img_path, threshold=threshold, display_mode='ortho', cut_coords=None, 
            colorbar=True, title=f"{network_name} (Slices)", figure=fig_stat
        )
        fig_stat.savefig(os.path.join(output_dir, f"{network_name}_slices.png"), dpi=300)
        plt.close(fig_stat)

        # 3: 3D 皮层 (Cortical Surface)
        fig_surf = plt.figure(figsize=(12, 5))
        texture = surface.vol_to_surf(img_path, fsaverage.pial_right)
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, texture, hemi='right', title=f"{network_name} (Right Surface)",
            colorbar=True, threshold=threshold, bg_map=fsaverage.sulc_right, figure=fig_surf
        )
        fig_surf.savefig(os.path.join(output_dir, f"{network_name}_surface.png"), dpi=300)
        plt.close(fig_surf)
        
    print(f"\n✅ 绘图任务完成！请前往 '{output_dir}' 文件夹查看。")

if __name__ == '__main__':
    # -------------------------------------------------------------
    # 💡 玩法示例：你可以根据需求取消注释以下某一种调用方式
    # -------------------------------------------------------------
    
    # 玩法 A：正常画全部
    visualize_nifti_results()
    
    # 玩法 B：不管有多少个，我只画前 3 个看看效果
    # visualize_nifti_results(max_plots=3)
    
    # 玩法 C：我只想看 DMN 和工作记忆相关的，指定序号或名字过滤
    # 只要文件名里包含 '02' 或者 'working_memory' 就会被画出来
    # visualize_nifti_results(target_networks=[ 'working_memory'])