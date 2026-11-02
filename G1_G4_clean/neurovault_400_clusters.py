import os
import numpy as np
import logging
from network_cluster_utils import mask_aligned_images, write_provenance

def create_wordcloud(words_weights, vocabulary, output_path, title="Motif", *, seed=0):
    """辅助函数：生成并保存消除了解剖学噪音的词云"""
    # 过滤掉高频无意义词汇
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update([
        'cortex', 'gyrus', 'sulcus', 'brain', 'network', 'region',
        'activation', 'activity', 'area', 'task', 'associated'
    ])

    # 构造词频字典
    word_freq = {vocabulary[i]: float(words_weights[i]) for i in range(len(vocabulary)) if words_weights[i] > 0}

    # 剔除停用词
    clean_word_freq = {k: v for k, v in word_freq.items() if k.lower() not in custom_stopwords}

    if not clean_word_freq:
        return

    wc = WordCloud(scale=4, background_color='white', max_words=20, collocations=False, random_state=seed)
    wc.generate_from_frequencies(clean_word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main(data_dir='./neurovault_production', output_dir='./analysis_results', *, seed=0):
    """主函数：从读取到网络提取的全流程"""
    # 1. 创建输出目录结构
    import pandas as pd
    import nibabel as nib
    import scipy.stats
    import sklearn.cluster
    import sklearn.decomposition
    import sklearn.preprocessing
    import nilearn.datasets
    import nilearn.image
    import nilearn.maskers
    import nilearn.plotting
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    rng = np.random.default_rng(seed)
    dirs = {
        'nifti': os.path.join(output_dir, 'NIfTI_maps'),
        'figs': os.path.join(output_dir, 'Figures'),
        'arrays': os.path.join(output_dir, 'Numpy_arrays')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # =========================================================================
    # 阶段 1：本地数据读取与严格对齐
    # =========================================================================
    logging.info("📂 [阶段 1] 加载本地元数据与图谱路径...")
    meta_csv_path = os.path.join(data_dir, 'neurovault_fMRI_Zmaps_meta.csv')
    weights_csv_path = os.path.join(data_dir, 'neurosynth_weights_aligned.csv')

    if not os.path.exists(meta_csv_path) or not os.path.exists(weights_csv_path):
        logging.error("找不到元数据 CSV，请确认 data_dir 路径正确。")
        return

    meta_df = pd.read_csv(meta_csv_path)
    weights_df = pd.read_csv(weights_csv_path)
    if (len(meta_df) != len(weights_df) or
            not np.array_equal(meta_df['final_z_map_path'].astype(str),
                               weights_df['final_z_map_path'].astype(str))):
        raise ValueError('Metadata and semantic weights must have aligned image paths and row order')

    # 提取有效的本地 NIfTI 路径和对齐的 NeuroSynth 矩阵
    valid_paths = meta_df['final_z_map_path'].values

    # 提取词汇表 (跳过前两列 'original_image_path', 'final_z_map_path')
    vocabulary = weights_df.columns[2:].values
    term_weights = weights_df.iloc[:, 2:].values
    term_weights[term_weights < 0] = 0  # 清理负权重

    total_images = len(valid_paths)
    logging.info(f"✅ 成功读取 {total_images} 张对齐的 Z 图谱记录。")

    # =========================================================================
    # 阶段 2：空间掩膜与矩阵展平 (Masking)
    # =========================================================================
    logging.info("🧠 [阶段 2] 构建 MNI 掩膜并将 3D 图像打平为 2D 矩阵...")
    # 使用 3mm 模板，自带 4mm FWHM 平滑，极大降低噪音和内存占用
    mask_img = nilearn.datasets.load_mni152_brain_mask(resolution=3)
    masker = nilearn.maskers.NiftiMasker(
        mask_img=mask_img,
        smoothing_fwhm=4.0, # 直接在加载时进行空间平滑
        standardize=False,
        memory=os.path.join(output_dir, 'nilearn_cache'),
        memory_level=1
    )
    masker.fit()

    X, term_weights, is_usable = mask_aligned_images(valid_paths, term_weights,
        lambda path: masker.transform(nilearn.image.smooth_img(path, fwhm=None)),
        on_error=lambda idx, path, error: logging.warning('Failed to mask %s: %s', path, error))
    write_provenance(os.path.join(output_dir, 'cluster_provenance.json'), seed=seed,
        arrays=dict(X=X, term_weights=term_weights, vocabulary=vocabulary.astype(str),
                    image_paths=valid_paths.astype(str), is_usable=is_usable),
        input_files=[meta_csv_path, weights_csv_path],
        settings=dict(iterations=200, components=20, sample_fraction=.8, smoothing_fwhm_mm=4.,
                      clustering='average linkage cosine on semantic profiles', clusters=15))
    logging.info(f"✅ 矩阵展平完成。特征矩阵形状: {X.shape}")

    # =========================================================================
    # 阶段 3：自适应 Bootstrap PCA 降维 (基于 N=400 的黄金比例)
    # =========================================================================
    logging.info("⚙️ [阶段 3] 启动 Bootstrap PCA 提取空间特征...")

    num_iterations = 200
    n_components = 20
    # 动态抽样：抽取 80% 的数据
    num_samples_to_select = int(len(X) * 0.8)
    if num_samples_to_select < 2 * n_components:
        raise ValueError('Motif bootstrap requires at least 50 usable images for two PCA batches')

    all_pca_maps = np.empty((num_iterations * n_components, X.shape[1]))
    all_term_weights = np.empty((num_iterations * n_components, term_weights.shape[1]))

    for i in tqdm(range(num_iterations), desc="Bootstrap PCA"):
        # 1. 无放回抽样
        selected_indices = rng.choice(len(X), num_samples_to_select, replace=False)
        X_selected = X[selected_indices]
        term_weights_selected = term_weights[selected_indices]

        # 2. 标准化 (消除不同实验绝对激活值的尺度差异)
        scaler = sklearn.preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X_selected).astype(np.float32)

        # 3. 增量 PCA 拟合
        ipca = sklearn.decomposition.IncrementalPCA(n_components=n_components)
        for batch in np.array_split(X_scaled, max(2, len(X_scaled)//50)):
            ipca.partial_fit(batch)

        # 获得本轮的空间基底
        pca_maps = ipca.components_  # 形状: (n_components, n_voxels)
        image_scores = ipca.transform(X_scaled).T

        # 4. 投影到语义空间
        term_weights_for_components = np.dot(image_scores, term_weights_selected)

        start_idx = i * n_components
        end_idx = start_idx + n_components
        all_pca_maps[start_idx:end_idx, :] = pca_maps
        all_term_weights[start_idx:end_idx, :] = term_weights_for_components

    # 保存降维后的中间结果，方便日后直接调用
    np.savez_compressed(
        os.path.join(dirs['arrays'], 'bootstrap_pca_ensemble.npz'),
        pca_maps=all_pca_maps,
        term_weights=all_term_weights,
        seed=seed
    )

    # =========================================================================
    # 阶段 4：语义聚类与均值网络生成
    # =========================================================================
    logging.info("📊 [阶段 4] 基于 NeuroSynth 语义特征进行层次聚类...")

    n_cluster = 15  # 提取 15 个大尺度功能网络

    # 使用余弦距离聚类语义剖面
    clustering = sklearn.cluster.AgglomerativeClustering(
        metric='cosine',
        n_clusters=n_cluster,
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(all_term_weights)

    average_pca_maps = np.zeros((n_cluster, all_pca_maps.shape[1]))
    average_term_weights = np.zeros((n_cluster, all_term_weights.shape[1]))

    # =========================================================================
    # 阶段 5：可视化与结果持久化
    # =========================================================================
    logging.info("💾 [阶段 5] 正在聚合网络并保存可视化结果...")

    for i in range(n_cluster):
        indices = np.where(cluster_labels == i)[0]
        if len(indices) == 0: continue

        # 1. 均值化去噪
        pc_map = np.mean(all_pca_maps[indices], axis=0)
        pc_terms = np.mean(all_term_weights[indices], axis=0)

        # 2. 符号对齐 (确保主激活区是正值，便于可视化)
        if -pc_map.min() > pc_map.max():
            pc_map = -pc_map
            pc_terms = -pc_terms

        # 3. 提取前三个关键认知词汇
        important_terms = vocabulary[np.argsort(pc_terms)[-3:]][::-1]
        cluster_name = f"Motif_{i+1:02d}_{'_'.join(important_terms)}"

        logging.info(f"  -> 生成 {cluster_name}")

        # 4. 逆变换回 3D NIfTI 并保存
        pc_img = masker.inverse_transform(pc_map)
        nib.save(pc_img, os.path.join(dirs['nifti'], f"{cluster_name}.nii.gz"))

        # 5. 绘制脑图 (PNG)
        pc_threshold = scipy.stats.scoreatpercentile(np.abs(pc_map), 90) # 截取前10%的高亮区域
        nilearn.plotting.plot_stat_map(
            pc_img,
            threshold=pc_threshold,
            colorbar=True,
            title=f"Network {i+1}: {', '.join(important_terms)}",
            output_file=os.path.join(dirs['figs'], f"{cluster_name}_brain.png")
        )

        # 6. 绘制词云 (PNG)
        create_wordcloud(
            pc_terms,
            vocabulary,
            output_path=os.path.join(dirs['figs'], f"{cluster_name}_wordcloud.png"),
            title=f"Network {i+1} Semantic Profile", seed=seed
        )

    logging.info("🎉 恭喜！端到端图谱萃取管线全部执行完毕。请前往 output_dir 查看你的专属大尺度脑网络。")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build legacy motifs from aligned local NeuroVault maps.')
    parser.add_argument('--data-dir', default='./neurovault_production')
    parser.add_argument('--output-dir', default='./neuro_motifs_results')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, seed=args.seed)
