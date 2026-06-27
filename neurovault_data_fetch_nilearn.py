import os
import gc
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import t, norm
from nilearn.datasets import fetch_neurovault
import logging
import time
from requests.exceptions import RequestException
os.environ['NO_PROXY'] = 'neurovault.org'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_t_to_z(t_scores, df):
    """极度稳健的 T-to-Z 转换函数"""
    signs = np.sign(t_scores)
    p_values = np.clip(t.sf(np.abs(t_scores), df=df), 1e-300, 1.0)
    z_scores = signs * norm.isf(p_values)
    return np.nan_to_num(z_scores)

def append_to_csv(dataframe, filepath):
    """增量写入 CSV"""
    if not os.path.exists(filepath):
        dataframe.to_csv(filepath, index=False)
    else:
        dataframe.to_csv(filepath, mode='a', header=False, index=False)

def stream_process_neurovault(data_dir='./neurovault_data_stream', max_images=None, batch_size=200):
    """
    终极管线：一次性全量拉取元数据，随后在本地进行分批流式图像处理。
    """
    os.makedirs(data_dir, exist_ok=True)
    meta_csv_path = os.path.join(data_dir, 'neurovault_fMRI_Zmaps_meta.csv')
    weights_csv_path = os.path.join(data_dir, 'neurosynth_weights_aligned.csv')
    
    # ---------------------------------------------------------
    # 阶段 1：网络同步 (Data Sync) - 仅执行一次！
    # ---------------------------------------------------------
    logging.info("🌐 [阶段 1] 开始同步 NeuroVault API 数据 (底层文件自带缓存)...")
    
    # 这里直接请求你想要的总体数量 (例如 None 代表全库，或 10000)
    # nilearn 会从第 1 页拉到最后一页，但对于硬盘上已存在的 nii.gz 它会自动跳过下载
    max_retries = 10  # 最大重试次数
    retry_delay = 5   # 断线后等待 5 秒再重试
    nv_data = None

    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"🚀 第 {attempt} 次尝试拉取数据...")
            
            nv_data = fetch_neurovault(
                max_images=max_images,
                fetch_neurosynth_words=True,
                image_terms={'modality': 'fMRI-BOLD'},
                data_dir=data_dir,
                verbose=1
            )
            
            # 如果成功执行到这里，说明没有报错，直接跳出循环
            logging.info("✅ 网络同步顺利完成！")
            break
            
        except Exception as e:
            logging.warning(f"⚠️ 网络连接中断 (尝试 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                logging.info(f"⏳ 等待 {retry_delay} 秒后自动重试...")
                time.sleep(retry_delay)
            else:
                logging.error("❌ 达到最大重试次数，拉取彻底失败。请检查网络或代理设置！")
                return # 彻底失败则退出函数
                
    # 检查是否真的拉到了数据
    if nv_data is None or len(nv_data.images) == 0:
        logging.warning("⚠️ 最终未获取到任何图谱数据。")
        return

    # ---------------------------------------------------------
    # 阶段 2：数据结构化与断点续传检测
    # ---------------------------------------------------------
    logging.info("🧠 [阶段 2] 解析元数据并进行断点续传检查...")
    meta_df = pd.DataFrame(nv_data.images_meta)
    meta_df['original_image_path'] = nv_data.images
    
    # 获取 NeuroSynth 标签权重
    weights_df = pd.DataFrame(nv_data.word_frequencies, columns=nv_data.vocabulary)
    weights_df['original_image_path'] = nv_data.images
    
    valid_maps = meta_df[meta_df['map_type'].isin(['Z map', 'T map'])].copy()
    
    # 断点续传检查
    processed_paths = set()
    if os.path.exists(meta_csv_path):
        processed_paths = set(pd.read_csv(meta_csv_path)['original_image_path'].dropna())
    
    unprocessed_maps = valid_maps[~valid_maps['original_image_path'].isin(processed_paths)].copy()
    
    if unprocessed_maps.empty:
        logging.info("🎉 所有 Z/T 图像均已处理完毕。")
        return

    # ---------------------------------------------------------
    # 阶段 3：本地分批图像处理 (Local Batch Processing)
    # ---------------------------------------------------------
    logging.info(f"⚙️ [阶段 3] 启动本地分批处理 (Batch Size = {batch_size}) ...")
    
    # 将待处理的数据 DataFrame 切分为多个 Batch
    chunks = [unprocessed_maps[i : i + batch_size] for i in range(0, len(unprocessed_maps), batch_size)]
    
    for batch_idx, current_chunk in enumerate(chunks):
        logging.info("=" * 50)
        logging.info(f"📦 正在处理 Batch {batch_idx + 1} / {len(chunks)} (包含 {len(current_chunk)} 张图像)")
        
        final_image_paths = []
        
        # 逐个对 NIfTI 图像进行重运算
        for idx, row in current_chunk.iterrows():
            img_path = row['original_image_path']
            map_type = row['map_type']
            
            if map_type == 'Z map':
                final_image_paths.append(img_path)
                
            elif map_type == 'T map':
                # 提取自由度
                df_val = None
                if 't_statistic_df' in row and pd.notnull(row['t_statistic_df']):
                    df_val = float(row['t_statistic_df'])
                elif 'number_of_subjects' in row and pd.notnull(row['number_of_subjects']):
                    df_val = float(row['number_of_subjects']) - 1.0
                
                if df_val is None or df_val <= 0:
                    final_image_paths.append(None)
                    continue
                
                try:
                    # 读取、转换并保存新 NIfTI
                    img = nib.load(img_path)
                    t_data = img.get_fdata()
                    z_data = safe_t_to_z(t_data, df_val)
                    
                    new_img_path = img_path.replace('.nii.gz', '_converted_Z.nii.gz')
                    z_img = nib.Nifti1Image(z_data, img.affine, img.header)
                    nib.save(z_img, new_img_path)
                    
                    final_image_paths.append(new_img_path)
                    
                    # 极速释放内存
                    del img, t_data, z_data, z_img
                except Exception as e:
                    logging.error(f"  ❌ 转换出错 {os.path.basename(img_path)}: {e}")
                    final_image_paths.append(None)
        
        # 将结果合并回本批次的 DataFrame
        current_chunk = current_chunk.copy()
        current_chunk['final_z_map_path'] = final_image_paths
        
        # 从 weights_df 中抽出对应的行
        current_weights = weights_df[weights_df['original_image_path'].isin(current_chunk['original_image_path'])].copy()
        current_weights = pd.merge(current_chunk[['original_image_path', 'final_z_map_path']], current_weights, on='original_image_path', how='inner')
        
        # 清除转换失败（None）的行
        current_chunk = current_chunk.dropna(subset=['final_z_map_path'])
        current_weights = current_weights.dropna(subset=['final_z_map_path'])
        
        # 追加保存到 CSV
        if not current_chunk.empty:
            append_to_csv(current_chunk, meta_csv_path)
            append_to_csv(current_weights, weights_csv_path)
            logging.info(f"💾 Batch {batch_idx + 1} 完成，{len(current_chunk)} 条有效记录已追加至硬盘。")
        
        # 主动触发垃圾回收，确保内存在下个 batch 前回落
        del current_chunk, current_weights
        gc.collect()

    logging.info("========================================")
    logging.info("🏁 恭喜！全部分批处理管线已顺利执行完毕。")

if __name__ == "__main__":
    # max_images=None 时，将在第一阶段拉取整个 NeuroVault 库的元数据
    # batch_size=200 保证了无论总量多少，你的工作站内存在处理阶段都不会崩溃
    stream_process_neurovault(data_dir='./neurovault_production', max_images=None, batch_size=200)