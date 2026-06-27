import numpy as np
from scipy.spatial import cKDTree
from nimare.dataset import Dataset
from nimare.meta.kernel import MKDAKernel
import nibabel as nib
import mne
import logging
import os
from mne.datasets import fetch_fsaverage
import matplotlib.pyplot as plt

from pathlib import Path
import nibabel as nib
import scipy.sparse 
from joblib import Parallel, delayed


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def build_macm_surface_z_matrix(
    dset_path,
    fsaverage_src_path=None,
    radius=10.0,
    n_jobs=1,
    min_studies=5,
    output_npz='macm_surface_z.npz'
):
    """
    基于 MACM 原理构建 fsaverage5 表面空间上的顶点共激活 Z 矩阵。

    参数:
        dset_path: NiMARE Dataset 文件路径
        fsaverage_src_path: fsaverage-ico-5-src.fif 路径，None 则自动下载
        radius: MKDA 核半径 (mm)
        n_jobs: 并行进程数，注意内存！单进程更安全
        min_studies: 种子点的最小研究数，低于此值则跳过该顶点
        output_npz: 输出文件名

    返回:
        M: (n_vertices, n_vertices) float32 矩阵，M[i,j] 表示以 i 为种子时顶点 j 的 Z 值
        coords: (n_vertices, 3) MNI 坐标 (mm)
    """
    # 1. 加载数据集
    logging.info(f"加载数据集: {dset_path}")
    dset = Dataset.load(dset_path)
    masker = dset.masker
    mask_img = masker.mask_img
    mask_data = mask_img.get_fdata().astype(bool)
    n_studies = len(dset.ids)
    logging.info(f"数据集加载完毕，研究数: {n_studies}")

    # 2. 生成 MA maps（连续密度图）
    logging.info("生成 MA maps（MKDA with r={}）...".format(radius))
    kernel = MKDAKernel(r=radius)
    ma_maps = kernel.transform(dset, return_type="array")
    if scipy.sparse.issparse(ma_maps):
        ma_maps = ma_maps.toarray()
    ma_maps = ma_maps.astype(np.float32)
    n_voxels = ma_maps.shape[1]
    logging.info(f"MA maps 形状: {ma_maps.shape}")

    # 3. 二进制焦点矩阵 F（稀疏）
    logging.info("构建二进制焦点矩阵...")
    # F = (ma_maps > 0).astype(np.int32)
    # F_csr = csr_matrix(F) if not scipy.sparse.issparse(F) else F.tocsr()
    # del F  # 释放密集版 F 的内存
    F_csc = scipy.sparse.csc_matrix(ma_maps > 0)

    # 4. 全局逐体素统计量
    logging.info("计算逐体素均值和标准差...")
    mean_all = ma_maps.mean(axis=0, dtype=np.float64).astype(np.float32)
    std_all = ma_maps.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
    std_all[std_all == 0] = 1.0

    # 5. 获取 fsaverage5 表面顶点和体素坐标映射
    if fsaverage_src_path is None:
        fs_dir = Path(fetch_fsaverage())
        fsaverage_src_path = str(fs_dir / 'bem' / 'fsaverage-ico-5-src.fif')
    logging.info(f"读取源空间: {fsaverage_src_path}")
    src = mne.read_source_spaces(fsaverage_src_path, verbose=False)
    # 顶点坐标 (mm)
    coords_lh = src[0]['rr'][src[0]['vertno']] * 1000.0
    coords_rh = src[1]['rr'][src[1]['vertno']] * 1000.0
    all_surf_coords = np.concatenate([coords_lh, coords_rh]).astype(np.float32)
    n_verts = all_surf_coords.shape[0]
    # 左右半球顶点索引（用于稍后可能的可视化）
    vertno_lh = src[0]['vertno'].copy()
    vertno_rh = src[0]['vertno'].copy() if False else src[1]['vertno'].copy()
    logging.info(f"表面顶点总数: {n_verts}")

    # 体素 MNI 坐标
    voxel_ijk = np.argwhere(mask_data)  # (N_vox, 3)
    voxel_mni = nib.affines.apply_affine(mask_img.affine, voxel_ijk).astype(np.float32)
    logging.info(f"有效体素数: {voxel_mni.shape[0]}")

    # 体素 -> 表面顶点的 KD-Tree
    logging.info("构建体素到表面的映射...")
    tree_vox = cKDTree(voxel_mni)
    _, seed_vox_idx = tree_vox.query(all_surf_coords, k=1)  # 每个顶点对应的种子体素

    tree_surf = cKDTree(all_surf_coords)
    _, vox2surf = tree_surf.query(voxel_mni)  # 每个体素最近的表面顶点
    # 预计算每个顶点的体素计数（用于平均）
    vertex_counts = np.bincount(vox2surf, minlength=n_verts)

    # 6. 单顶点处理函数
    def process_one_vertex(vi):
        j = seed_vox_idx[vi]  # 种子体素列索引
        start_ptr = F_csc.indptr[j]
        end_ptr = F_csc.indptr[j+1]
        
        if (end_ptr - start_ptr) < min_studies:
            return (vi, None)
            
        study_ids = F_csc.indices[start_ptr:end_ptr]
        # 计算激活组的平均 MA 图
        A_j = ma_maps[study_ids].mean(axis=0, dtype=np.float64).astype(np.float32)
        Z = (A_j - mean_all) / std_all
        # 映射到表面顶点（bincount 求平均）
        vert_vals = np.bincount(vox2surf, weights=Z, minlength=n_verts)
        vert_vals /= vertex_counts  # 可能会有除零，但 vertex_counts 中全脑应有值
        vert_vals[vertex_counts == 0] = 0.0
        return (vi, vert_vals)

    # 7. 并行/串行执行
    if n_jobs > 1:
        logging.info(f"使用 {n_jobs} 个进程并行处理 {n_verts} 个顶点...")
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_one_vertex)(vi) for vi in range(n_verts)
        )
    else:
        logging.info(f"单进程处理 {n_verts} 个顶点...")
        results = []
        for vi in range(n_verts):
            res = process_one_vertex(vi)
            results.append(res)
            if vi % 1000 == 0:
                logging.info(f"  已处理 {vi}/{n_verts} 个顶点")

    # 8. 构建结果矩阵
    M = np.zeros((n_verts, n_verts), dtype=np.float32)
    processed = 0
    for vi, vert_vals in results:
        if vert_vals is not None:
            M[vi, :] = vert_vals
            processed += 1
    logging.info(f"成功处理 {processed}/{n_verts} 个顶点（研究数 >= {min_studies}）")

    # 9. 保存
    np.savez_compressed(output_npz,
                        M=M,
                        coords=all_surf_coords,
                        vertno_lh=vertno_lh,
                        vertno_rh=vertno_rh,
                        voxel_mni=voxel_mni,
                        seed_vox_idx=seed_vox_idx,
                        vox2surf=vox2surf)
    logging.info(f"矩阵已保存至 {output_npz}")
    return M, all_surf_coords
    
if not os.path.exists('macm_surface_z.npz'):
    M, coords = build_macm_surface_z_matrix(
        dset_path='./neurosyn_nimare_data/neurosynth_dataset.pkl.gz',
        fsaverage_src_path=None,   # 自动下载 fsaverage-ico-5
        radius=10.0,
        n_jobs=1,                  # 先使用单进程测试，确认可行后可增加
        min_studies=5,
        output_npz='macm_surface_z.npz'
    )
else:
    data = np.load('macm_surface_z.npz')
    M = data['M']
    coords = data['coords']
    seed_vox_idx = data['seed_vox_idx']
    print('不同种子体素的数量:', len(np.unique(seed_vox_idx)))
    print('前10个种子索引:', seed_vox_idx[:10])
# def build_macm_surface_matrix(dset_path, fsaverage_src_path=None, batch_size=200):
#     """
#     构建 fsaverage5 源空间的 MACM 共激活相关矩阵（稳健版）。
#     """
#     # 1. 加载 Dataset
#     dset = Dataset.load(dset_path)
#     masker = dset.masker
#     mask_img = masker.mask_img
#     if fsaverage_src_path is None:
#         fs_dir = mne.datasets.fetch_fsaverage()
#         fsaverage_src_path = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
#         logging.info(f"自动使用 fsaverage5 源空间：{fsaverage_src_path}")

#     # 2. 获取与 MA maps 列顺序严格一致的体素坐标
#     #    利用 masker.volume 属性得到 3D 掩模，再展平为与内部表示相同顺序的坐标
#     mask_data = mask_img.get_fdata().astype(bool)
#     # nimare 的 masker 内部多用 Fortran 序（order='F'）展平
#     flat_indices = np.where(mask_data.ravel(order='F'))[0]  # 一维索引
#     voxel_ijk = np.column_stack(np.unravel_index(flat_indices, mask_data.shape, order='F'))
#     voxel_mni = nib.affines.apply_affine(mask_img.affine, voxel_ijk)

#     # 3. 生成 MA maps（确保列顺序正确）
#     logging.info("生成 MA maps...")
#     kernel = MKDAKernel(r=10)
#     ma_maps_2d = kernel.transform(dset, return_type="array")  # (n_studies, n_voxels)
#     # if issparse(ma_maps_2d):
#     #     ma_maps_2d = ma_maps_2d.toarray()                    # 转为密集（若内存不足可保留稀疏）
    
#     # 4. 获取 fsaverage5 顶点坐标（mm）
#     src = mne.read_source_spaces(fsaverage_src_path, verbose=False)
#     lh_rr = src[0]['rr'][src[0]['vertno']] * 1000
#     rh_rr = src[1]['rr'][src[1]['vertno']] * 1000
#     surf_coords = np.concatenate([lh_rr, rh_rr])
#     n_verts = surf_coords.shape[0]

#     # 5. KD‑Tree 快速匹配
#     logging.info("空间匹配顶点与体素...")
#     tree = cKDTree(voxel_mni)
#     dists, nn_idx = tree.query(surf_coords, k=1)
#     # 标记越界点
#     valid = dists <= 15.0

#     # 6. 切片得到表面活动矩阵
#     A_surf = ma_maps_2d[:, nn_idx]          # (n_studies, n_verts)
#     A_surf[:, ~valid] = 0.0                 # 越界置零

#     # 7. 分块计算皮尔逊相关（避免内存爆炸）
#     logging.info("分块计算 Pearson 相关...")
#     # 先 z‑score 标准化列（去均值除标准差，忽略常数列）
#     A_std = np.zeros_like(A_surf, dtype=np.float32)
#     col_std = np.std(A_surf, axis=0, ddof=1)
#     col_mean = np.mean(A_surf, axis=0)
#     const_cols = col_std == 0
#     A_std[:, ~const_cols] = (A_surf[:, ~const_cols] - col_mean[~const_cols]) / col_std[~const_cols]

#     # 分块计算相关矩阵
#     M = np.eye(n_verts, dtype=np.float32)
#     for start in range(0, n_verts, batch_size):
#         end = min(start + batch_size, n_verts)
#         # corr = (X.T @ Y) / (N - 1)  当 X,Y 已标准化
#         block = (A_std.T @ A_std[:, start:end]) / (A_surf.shape[0] - 1)
#         M[start:end, :] = block.T
#         M[:, start:end] = block         # 对称填充（因为矩阵对称）
#         logging.info(f"  处理 {end}/{n_verts} 顶点")

#     # 8. 清理对角线并填充无效列
#     M[:, const_cols] = 0.0
#     M[const_cols, :] = 0.0
#     np.fill_diagonal(M, 1.0)

#     # 确保对称（数值误差修正）
#     M = (M + M.T) / 2.0

#     logging.info("完成！")
#     return M, surf_coords



# if not os.path.exists('./neurosyn_nimare_data/macm_matrix.npz'):
#     M, coords = build_macm_surface_matrix(
#         dset_path='./neurosyn_nimare_data/neurosynth_dataset.pkl.gz',
#         fsaverage_src_path=None   # 自动获取
#     )

#     np.savez_compressed('./neurosyn_nimare_data/macm_matrix.npz', M=M, coords=coords)
# else:
#     data = np.load('./neurosyn_nimare_data/macm_matrix.npz')
#     M = data['M']
#     coords = data['coords']

mne.viz.set_3d_backend('pyvista') 
# 1) 读取源空间（ico-5）
fs_dir = fetch_fsaverage()
fsaverage_src_path = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
src = mne.read_source_spaces(fsaverage_src_path, verbose=False)

# 2) 确保 M 为 float 且对称（实际只需用前 3 行）
data_top3 = M[:10, :].T  # 形状 (3, 20484)
# 3) 创建 SourceEstimate：每一行作为一个“时间点”
# 注意：顶点列表必须与源空间匹配：左半球 vertno，右半球 vertno
vertno_lh = src[0]['vertno']  # 左半球的顶点索引（10242 个）
vertno_rh = src[1]['vertno']  # 右半球的顶点索引（10242 个）

# 构建 SourceEstimate 对象
stc = mne.SourceEstimate(
    data=data_top3,                                 # (n_times, n_verts) → (3, 20484)
    vertices=[vertno_lh, vertno_rh],                # 左、右半球的顶点编号
    tmin=0, tstep=1                                 # 时间参数（这里仅用于区分行）
)

# 4) 用 PySurfer 绘制（如果安装了 pyvista 和 vtk）
# 将每一行（即每一个顶点）的共激活图保存为图片
plt.ioff()  # 仍然保留，防止弹出太多窗口

for i in range(10):
  
    print(data_top3[:,i])
    # 注意：initial_time 必须为 0！
    fig = mne.viz.plot_source_estimates(
        stc, subject='fsaverage', hemi='both', surface='white',
        initial_time=i,      # ← 这里是唯一改动的地方
        views='lateral',
        background='white', foreground='black',
        colormap='coolwarm',
        time_label=f'Vertex {i}'
    )
    fig.save_image(f'temp_{i}.png')
    x = input(" ")
    

