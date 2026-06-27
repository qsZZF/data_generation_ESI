from nimare import dataset, decode
import os
local_data_dir = './neurosyn_nimare_data' 
dataset_save_path = os.path.join(local_data_dir, 'neurosynth_dataset.pkl.gz')
dset = dataset.Dataset.load(dataset_save_path)
# --- 第4步：训练解码器并保存模型 ---
print("正在训练解码器模型...")
decoder = decode.continuous.CorrelationDecoder(
    frequency_threshold=0.001,
    n_cores=4 
)
decoder.fit(dset)

# 将训练好的解码器保存为.pkl文件
decoder_save_path = os.path.join(local_data_dir, 'neurosynth_decoder.pkl')
decoder.save(decoder_save_path)
print(f"解码器模型已保存至: {decoder_save_path}")