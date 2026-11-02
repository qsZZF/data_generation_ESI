# -*- coding: utf-8 -*-

import os
import re

import numpy as np
from network_cluster_utils import mask_aligned_images, write_provenance

def main(path_prefix: str = '.', *, seed=0, img_count=25000, download=False):
    import matplotlib.pyplot as plt
    import nibabel as nib
    import nilearn.datasets
    import nilearn.image
    import nilearn.maskers
    import nilearn.plotting
    import scipy.cluster.hierarchy
    import scipy.stats
    import sklearn.cluster
    import sklearn.decomposition
    import sklearn.preprocessing
    import tqdm
    import wordcloud
    import pandas as pd

    rng = np.random.default_rng(seed)

    base_path = f'{path_prefix}/assets'

    nilearn_data_path = os.path.join(base_path, 'nilearn')
    if not os.path.exists(nilearn_data_path):
        os.makedirs(nilearn_data_path)

    # Use the local cache unless downloads were explicitly requested.
    nv_data = nilearn.datasets.fetch_neurovault(max_images=img_count, fetch_neurosynth_words=True,
        data_dir=nilearn_data_path, mode='download_new' if download else 'offline', verbose=1)

    term_weights = nv_data['word_frequencies']
    vocabulary = np.asarray(nv_data['vocabulary'], dtype=str)

    # Clean and report term scores
    term_weights[term_weights < 0] = 0
    total_scores = np.mean(term_weights, axis=0)

    print('Top 10 neurosynth terms from downloaded images:')
    for term_idx in np.argsort(total_scores)[-10:][::-1]:
        print(vocabulary[term_idx])

    # Remove anatomical stop words before building the word cloud.
    vocabulary_clean = re.sub(r'\bcortex\b', '', ' '.join(nv_data['vocabulary']))
    vocabulary_clean = re.sub(r'\bgyrus\b', '', vocabulary_clean)

    stopwords = set(wordcloud.STOPWORDS)
    # Extra anatomical stop word keeps the cloud focused on functional descriptors.
    stopwords.add('cortex')

    # Generate a word cloud to visualize most frequent Neurosynth descriptors.
    word_cloud = wordcloud.WordCloud(scale=4, collocations=False, background_color='white',
        max_words=16, random_state=seed, stopwords=stopwords).generate(vocabulary_clean)

    # Display the generated Word Cloud
    _ = plt.figure(figsize=(12, 6))
    _ = plt.imshow(word_cloud)
    _ = plt.axis('off')
    plt.show()

    # Build a masker to vectorize all maps into a common brain mask.
    mask_img = nilearn.datasets.load_mni152_brain_mask(resolution=3) # resolution=2 is memory consuming
    masker = nilearn.maskers.NiftiMasker(mask_img=mask_img, memory=os.path.join(base_path, 'nilearn_cache'), memory_level=1)
    # Fit caches the mask transform so repeated inverse/forward transforms remain cheap.
    masker = masker.fit()

    # Images may fail to be transformed, and are of different shapes,
    # so we need to transform one-by-one and keep track of failures.
    X, term_weights, is_usable = mask_aligned_images(nv_data['images'], term_weights,
        lambda path: masker.transform(nilearn.image.smooth_img(path, fwhm=None)).astype(np.float32),
        on_error=lambda idx, path, error: print(f'Failed to mask image {idx}: {path}: {error}'))
    if len(X) < 3500:
        raise ValueError(f'CLASS bootstrap requires 3500 usable images; got {len(X)}')
    write_provenance(os.path.join(base_path, 'cluster_provenance.json'), seed=seed,
        arrays=dict(X=X, term_weights=term_weights, vocabulary=vocabulary,
                    image_paths=np.asarray(nv_data['images'], dtype=str), is_usable=is_usable),
        settings=dict(requested_images=img_count, actual_images=len(is_usable),
                      iterations=200, components=70, samples_per_iteration=3500,
                      clustering='average linkage cosine on semantic profiles', clusters=200))

    # Persist transformed maps and term weights for reuse across runs.
    np.savez_compressed(os.path.join(base_path, f'X_and_term_weights_{len(X)}.npz'),
                        X=X, term_weights=term_weights, vocabulary=vocabulary, seed=seed)

    # ICA on full dataset
    n_components = 10
    fast_ica = sklearn.decomposition.FastICA(n_components=n_components, random_state=seed)
    ica_maps = fast_ica.fit_transform(X.T).T

    term_weights_for_components_ICA = np.dot(fast_ica.components_, term_weights)

    # Exploratory PCA: flattened voxel indices are not spatial neighbours.
    scaler = sklearn.preprocessing.StandardScaler()
    # Standardize each image over its masked voxels in this exploratory branch.
    X_scaled = scaler.fit_transform(X.T)  # Caution for X.T
    X_scaled = X_scaled.astype(np.float32)

    n_components = 5  # modify to find better result
    ipca = sklearn.decomposition.IncrementalPCA(n_components=n_components)

    for batch in np.array_split(X_scaled, 10):
        ipca.partial_fit(batch)

    pca_maps = ipca.transform(X_scaled).T

    term_weights_for_components_PCA = np.dot(ipca.components_, term_weights)

    ic_imgs = {}

    show_all = False

    for index, (ic_map, ic_terms) in enumerate(zip(ica_maps, term_weights_for_components_ICA)):
        # Flip sign so that positive peaks dominate visualization.
        if -ic_map.min() > ic_map.max():
            ic_map = -ic_map
            ic_terms = -ic_terms

        ic_threshold = scipy.stats.scoreatpercentile(np.abs(ic_map), 90)
        ic_img = masker.inverse_transform(ic_map)
        important_terms = vocabulary[np.argsort(ic_terms)[-3:]]
        title = f"IC{int(index)}  {', '.join(important_terms[::-1])}"

        ic_imgs[f'IC{int(index)}'] = ic_img
        ic_imgs[f'IC{int(index)}_terms'] = important_terms

        if index == 0 or (show_all and index > 0):
            _ = nilearn.plotting.plot_stat_map(ic_img, threshold=ic_threshold, colorbar=False, title=title)
            plt.show()

    pc_imgs = {}

    show_all = False

    for index, (pc_map, pc_terms) in enumerate(zip(pca_maps, term_weights_for_components_PCA)):
        # Align signs to keep positive loadings for interpretation.
        if -pc_map.min() > pc_map.max():
            pc_map = -pc_map
            pc_terms = -pc_terms

        pc_threshold = scipy.stats.scoreatpercentile(np.abs(pc_map), 90)
        pc_img = masker.inverse_transform(pc_map)
        important_terms = vocabulary[np.argsort(pc_terms)[-3:]]
        title = f"PC{int(index)}  {', '.join(important_terms[::-1])}"

        pc_imgs[f'PC{int(index)}'] = pc_img
        pc_imgs[f'PC{int(index)}_terms'] = important_terms

        if index == 0 or (show_all and index > 0):
            _ = nilearn.plotting.plot_stat_map(pc_img, threshold=pc_threshold, colorbar=False, title=title)
            plt.show()

    # Bootstrap PCA ensembles
    num_samples_to_select = 3500
    num_iterations = 200
    n_components = 70

    all_pca_maps = np.empty((num_iterations * n_components, X.shape[1]))
    all_term_weights = np.empty((num_iterations * n_components, term_weights.shape[1]))

    for i in tqdm.tqdm(range(num_iterations)):
        # Sample without replacement so each bootstrap covers diverse studies.
        selected_indices = rng.choice(len(X), num_samples_to_select, replace=False)
        X_selected = X[selected_indices] # (nsamples, n_voxels)
        term_weights_selected = term_weights[selected_indices]

        # fast standardization
        means = X_selected.mean(axis=1, keepdims=True)
        stds = X_selected.std(axis=1, keepdims=True)
        # 加上 1e-8 防止除以 0 的极小概率事件
        X_standardized = (X_selected - means) / (stds + 1e-8)

        ipca = sklearn.decomposition.IncrementalPCA(n_components=n_components)
        for batch in np.array_split(X_standardized, 10):
            # Chunked fitting keeps memory usage predictable even with many voxels.
            ipca.partial_fit(batch)
        image_scores = ipca.transform(X_standardized).T
        pca_maps = ipca.components_

        # Project term weights onto each bootstrap component for semantic labelling.
        term_weights_for_components_PCA = np.dot(image_scores, term_weights_selected)

        start_idx = i * n_components
        end_idx = start_idx + n_components
        all_pca_maps[start_idx:end_idx, :] = pca_maps
        all_term_weights[start_idx:end_idx, :] = term_weights_for_components_PCA

    # Cache ensemble outputs for downstream notebooks.
    np.savez_compressed(os.path.join(base_path, f'all_pca_maps_and_term_weights_{len(X)}.npz'),
                        all_pca_maps=all_pca_maps, all_term_weights=all_term_weights, seed=seed)
    n_cluster = 200 # Cluster ensemble components
    # Cosine metric emphasises similarity in semantic profiles rather than absolute power.
    clustering = sklearn.cluster.AgglomerativeClustering(metric='cosine', n_clusters=n_cluster, linkage='average')
    clustering.fit(all_term_weights)

    words_label = clustering.labels_  # corresponding to the cluster in n_clusters

    average_pca_maps = np.zeros((n_cluster, all_pca_maps.shape[1]))
    average_term_weights = np.zeros((n_cluster, all_term_weights.shape[1]))

    # Average ensemble members within each cluster to obtain robust motifs.
    for i in range(n_cluster):
        indices = np.where(words_label == i)[0]
        if len(indices) > 0:
            average_pca_maps[i] = np.mean(all_pca_maps[indices], axis=0)
            average_term_weights[i] = np.mean(all_term_weights[indices], axis=0)

    fig_path = os.path.join(base_path, 'fig')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    pc_imgs = {}

    for index, (pc_map, pc_terms) in enumerate(zip(average_pca_maps, average_term_weights)):
        # Reorient map for visualization and save to disk per cluster.
        if -pc_map.min() > pc_map.max():
            pc_map = -pc_map
            pc_terms = -pc_terms

        pc_threshold = scipy.stats.scoreatpercentile(np.abs(pc_map), 90)
        pc_img = masker.inverse_transform(pc_map)
        important_terms = vocabulary[np.argsort(pc_terms)[-3:]]
        title = f"CLASS{int(index) + 1}  {', '.join(important_terms[::-1])}"

        pc_imgs[f'CLASS{int(index) + 1}'] = pc_img
        pc_imgs[f'CLASS{int(index) + 1}_terms'] = important_terms

        nilearn.plotting.plot_stat_map(pc_img, threshold=pc_threshold, colorbar=False, title=title,
                                       output_file=os.path.join(fig_path, f'CLASS{int(index) + 1}.png'))

    nii_path = os.path.join(base_path, 'SIM_data')
    if not os.path.exists(nii_path):
        os.makedirs(nii_path)

    # Export averaged maps as NIfTI images for further analysis.
    for key, img in pc_imgs.items():
        if 'terms' not in key:
            file_path = os.path.join(nii_path, f'{key}.nii.gz')
            nib.save(img, file_path)

    # Calculate the linkage matrix
    linkage_matrix = scipy.cluster.hierarchy.linkage(all_pca_maps, 'average')

    # Plot the dendrogram
    plt.figure(figsize=(15, 10))
    scipy.cluster.hierarchy.dendrogram(
        linkage_matrix,
        orientation='top',
        labels=np.arange(1, len(all_pca_maps) + 1),  # Assuming you want to label clusters by index
        distance_sort='descending',
        show_leaf_counts=True,
    )

    _ = plt.title('Hierarchical Clustering Dendrogram')
    _ = plt.xlabel('Index of Terms')
    _ = plt.ylabel('Distance')
    plt.show()



    # 建立一个字典来收集数据
    metadata_records = []

    for index, pc_terms in enumerate(average_term_weights):
        # 提取前 5 个最重要的词及其权重得分
        top_5_indices = np.argsort(pc_terms)[-5:][::-1]
        top_5_words = vocabulary[top_5_indices]
        top_5_scores = pc_terms[top_5_indices]

        record = {
            'Cluster_ID': f"CLASS{index + 1}",
            'Top1_Term': top_5_words[0], 'Top1_Score': round(top_5_scores[0], 4),
            'Top2_Term': top_5_words[1], 'Top2_Score': round(top_5_scores[1], 4),
            'Top3_Term': top_5_words[2], 'Top3_Score': round(top_5_scores[2], 4),
        }
        metadata_records.append(record)

    # 保存为 CSV
    df_meta = pd.DataFrame(metadata_records)
    csv_path = os.path.join(base_path, 'Cluster_Semantic_Metadata.csv')
    df_meta.to_csv(csv_path, index=False)
    print(f"📝 网络的语义身份档案已保存至: {csv_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build legacy CLASS maps from cached NeuroVault maps.')
    parser.add_argument('--path-prefix', default='.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-images', type=int, default=25000)
    parser.add_argument('--download', action='store_true', help='Explicitly allow downloading additional maps')
    args = parser.parse_args()
    main(args.path_prefix, seed=args.seed, img_count=args.max_images, download=args.download)
