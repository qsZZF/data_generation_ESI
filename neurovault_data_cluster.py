# -*- coding: utf-8 -*-

import os
import re

import matplotlib.pyplot as plt
import nibabel as nib
import nilearn.plotting
import numpy as np
import scipy
import sklearn.cluster
import tqdm
import wordcloud
import pandas as pd

def main(path_prefix: str = '.'):
    img_count = 25000  # Number of images to download, set as what you want

    base_path = f'{path_prefix}/assets'

    nilearn_data_path = os.path.join(base_path, 'nilearn')
    if not os.path.exists(nilearn_data_path):
        os.makedirs(nilearn_data_path)

    # Download statistical maps and Neurosynth term weights.
    # Setting mode='download_new' avoids re-downloading existing files on reruns.
    # NeuroVault fetch also grabs Neurosynth metadata so semantic summaries stay aligned with maps.
    nv_data = nilearn.datasets.fetch_neurovault(max_images=img_count, fetch_neurosynth_words=True, data_dir=nilearn_data_path, mode='download_new', verbose=1)

    term_weights = nv_data['word_frequencies']
    vocabulary = nv_data['vocabulary']

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
    word_cloud = wordcloud.WordCloud(scale=4, collocations=False, background_color='white', max_words=16).generate(vocabulary_clean)

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
    X = []
    # Track per-image usability so metadata and term weights stay aligned.
    is_usable = np.ones((img_count,), dtype=bool)

    for idx, image_path in enumerate(nv_data['images']):
        # load image and remove nan and inf values.
        # applying smooth_img to an image with fwhm=None simply cleans up
        # non-finite values but otherwise doesn't modify the image.
        try:
            # smooth_img with fwhm=None just ensures finite voxels before masking.
            X.append(masker.transform(nilearn.image.smooth_img(image_path, fwhm=None)).astype(np.float32))
        except Exception as e:
            meta = nv_data['images_meta'][idx]
            print(
                f"Failed to mask/reshape image: id: {meta.get('id')}; "
                f"name: {meta.get('name')}; "
                f"collection: {meta.get('collection_id')}; error: {e}"
            )
            is_usable[idx] = False

    # Stack valid maps and drop metadata rows for failed downloads.
    X = np.vstack(X)
    term_weights = term_weights[is_usable, :]

    # Persist transformed maps and term weights for reuse across runs.
    np.savez_compressed(os.path.join(base_path, f'X_and_term_weights_{img_count}.npz'), X=X, term_weights=term_weights)

    # ICA on full dataset
    n_components = 10
    fast_ica = sklearn.decomposition.FastICA(n_components=n_components, random_state=0)
    ica_maps = fast_ica.fit_transform(X.T).T

    term_weights_for_components_ICA = np.dot(fast_ica.components_, term_weights)

    # PCA on smoothed dataset
    scaler = sklearn.preprocessing.StandardScaler()
    # Transpose yields voxels as features; scaling prevents high-variance voxels from dominating.
    X_scaled = scaler.fit_transform(X.T)  # Caution for X.T
    X_scaled = X_scaled.astype(np.float32)
    # Optional gaussian filter
    X_smoothed = scipy.ndimage.gaussian_filter(X_scaled, sigma=1)  # sigma can be modified

    n_components = 5  # modify to find better result
    ipca = sklearn.decomposition.IncrementalPCA(n_components=n_components)

    for batch in np.array_split(X_smoothed, 10):
        ipca.partial_fit(batch)

    pca_maps = ipca.transform(X_smoothed).T

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

    # 快速获取词汇表 (获取 1 张图的元数据即可)
    tmp_data = nilearn.datasets.fetch_neurovault(max_images=1, fetch_neurosynth_words=True, data_dir=nilearn_data_path)
    vocabulary = tmp_data['vocabulary']
    # Reload from disk so ensembles later reuse the persisted arrays.
    with np.load(os.path.join(base_path, f'X_and_term_weights_{img_count}.npz')) as npz_file:
        X = npz_file['X']
        term_weights = npz_file['term_weights']
        print(f"Loaded X shape: {X.shape}, term_weights shape: {term_weights.shape}")
        

    # Bootstrap PCA ensembles
    num_samples_to_select = 3500
    num_iterations = 200
    n_components = 70

    all_pca_maps = np.empty((num_iterations * n_components, X.shape[1]))
    all_term_weights = np.empty((num_iterations * n_components, term_weights.shape[1]))

    for i in tqdm.tqdm(range(num_iterations)):
        # Sample without replacement so each bootstrap covers diverse studies.
        selected_indices = np.random.choice(len(X), num_samples_to_select, replace=False)
        X_selected = X[selected_indices] # (nsamples, n_voxels)
        term_weights_selected = term_weights[selected_indices]

        # # Standardize and smooth data -> PCA
        # scaler = sklearn.preprocessing.StandardScaler()
        # X_scaled = scaler.fit_transform(X_selected.T)
        # X_scaled = X_scaled.astype(np.float32)
        # X_smoothed = scipy.ndimage.gaussian_filter(X_scaled, sigma=1.5)
        
        # fast standardization
        means = X_selected.mean(axis=1, keepdims=True)
        stds = X_selected.std(axis=1, keepdims=True)
        # 加上 1e-8 防止除以 0 的极小概率事件
        X_smoothed = (X_selected - means) / (stds + 1e-8)

        ipca = sklearn.decomposition.IncrementalPCA(n_components=n_components)
        for batch in np.array_split(X_smoothed, 10):
            # Chunked fitting keeps memory usage predictable even with many voxels.
            ipca.partial_fit(batch)
        image_scores = ipca.transform(X_smoothed).T
        pca_maps = ipca.components_

        # Project term weights onto each bootstrap component for semantic labelling.
        term_weights_for_components_PCA = np.dot(image_scores, term_weights_selected)

        start_idx = i * n_components
        end_idx = start_idx + n_components
        all_pca_maps[start_idx:end_idx, :] = pca_maps
        all_term_weights[start_idx:end_idx, :] = term_weights_for_components_PCA

    # Cache ensemble outputs for downstream notebooks.
    np.savez_compressed(os.path.join(base_path, f'all_pca_maps_and_term_weights_{len(X)}.npz'), all_pca_maps=all_pca_maps, all_term_weights=all_term_weights)
    with np.load(os.path.join(base_path, 'all_pca_maps_and_term_weights_25000.npz')) as data:
        # 注意：这里的键名 ('all_term_weights' 或 'term_weights') 
        # 取决于你上次保存 np.savez_compressed 时用的变量名
        # 请根据实际情况修改，如果你用的是上一版的代码，通常是 'all_term_weights'
        all_term_weights = data['all_term_weights']
        all_pca_maps = data['all_pca_maps']

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
    os.environ['NO_PROXY'] = 'neurovault.org'
    main()