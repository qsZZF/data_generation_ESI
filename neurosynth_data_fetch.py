import os
from nimare import extract, io, decode

# --- 第1步：定义本地存储路径 ---
# 指定一个你希望存放所有数据的根目录
local_data_dir = './neurosyn_nimare_data' 
os.makedirs(local_data_dir, exist_ok=True)

# --- 第2步：下载NeuroSynth的原始数据 ---
print("正在从NeuroSynth下载原始数据...")
neurosynth_db = extract.fetch_neurosynth(
    data_dir=local_data_dir,
    version='7',
    source='abstract',
    vocab='terms',
    overwrite=False
)[0]
print("下载完成")
# 所有文件现在都保存在 local_data_dir/neurosynth/ 目录下

# --- 第3步：将原始数据转换为NiMARE的Dataset对象并保存 ---
print("正在转换为NiMARE Dataset...")
ns_dir = os.path.join(local_data_dir, 'neurosynth')

# # 根据第一步的输出，找到对应文件的实际路径（可能需要调整）
# coordinates_file = os.path.join(ns_dir, 'data-neurosynth_version-7_coordinates.tsv.gz')   # 或 .tsv
# metadata_file    = os.path.join(ns_dir, 'data-neurosynth_version-7_metadata.tsv.gz')      # 或 .tsv
# features_file    = os.path.join(ns_dir, 'data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_features.npz')
# vocab_file       = os.path.join(ns_dir, 'data-neurosynth_version-7_vocab-terms_vocabulary.txt')

# # 构建 annotations_files 字典
# annotations = {
#     "features": features_file,
#     "vocabulary": vocab_file
# }
# neurosynth_dset = io.convert_neurosynth_to_dataset(
#     coordinates_file=coordinates_file,
#     metadata_file=metadata_file,
#     annotations_files=annotations,
#     target='mni152_2mm'
# )
neurosynth_dset = io.convert_neurosynth_to_dataset(
    coordinates_file=neurosynth_db['coordinates'],
    metadata_file=neurosynth_db['metadata'],
    annotations_files=neurosynth_db['features'],
)
# 保存Dataset，方便后续直接加载
dataset_save_path = os.path.join(local_data_dir, 'neurosynth_dataset.pkl.gz')
neurosynth_dset.save(dataset_save_path)
print(f"数据集已保存至: {dataset_save_path}")
# from nimare import dataset
# dset = dataset.Dataset.load(dataset_save_path)
# --- 第4步：训练解码器并保存模型 ---
print("正在训练解码器模型...")
decoder = decode.continuous.CorrelationDecoder(
    frequency_threshold=0.001,
    n_cores=8 
)
decoder.fit(neurosynth_dset)

# 将训练好的解码器保存为.pkl文件
decoder_save_path = os.path.join(local_data_dir, 'neurosynth_decoder.pkl')
decoder.save(decoder_save_path)
print(f"解码器模型已保存至: {decoder_save_path}")

