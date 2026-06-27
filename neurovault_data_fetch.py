import os
import json
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from nimare.io import convert_neurovault_to_dataset
from nimare.dataset import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def create_robust_session(retries=5, backoff=2, timeout=30):
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.timeout = timeout
    return session

def fetch_all_public_collection_ids(output_file='neurovault_ids.json'):
    """
    使用 NeuroVault API 的 next 链接遍历，获取所有公开且非空的收藏集 ID。
    结果逐页追加保存到 JSON 文件，支持断点续传。
    """
    session = create_robust_session()
    initial_url = "https://neurovault.org/api/collections/?limit=100"
    
    # 读取已保存的数据，获取上次停止的 next URL
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            saved = json.load(f)
            all_ids = saved.get('ids', [])
            next_url = saved.get('next_url', initial_url)
            logging.info(f"发现已有文件，已保存 {len(all_ids)} 个 ID，从 {next_url} 继续")
    else:
        all_ids = []
        next_url = initial_url
    
    page_count = 0
    while next_url:
        logging.info(f"正在请求: {next_url}")
        try:
            resp = session.get(next_url)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"请求失败: {e}，等待 10 秒后重试...")
            import time
            time.sleep(10)
            continue  # 再次尝试相同的 URL
        
        # 筛选公开且图像数 > 0 的收藏集
        for col in data.get('results', []):
            # 公开收藏集的 private 字段为 False 或不存在
            is_public = not col.get('private', False)
            has_images = col.get('number_of_images', 0) > 0
            if is_public and has_images:
                all_ids.append(col['id'])
        
        # 获取下一页链接
        next_url = data.get('next')
        page_count += 1
        
        # 保存当前进度和 next_url
        with open(output_file, 'w') as f:
            json.dump({
                'ids': all_ids,
                'next_url': next_url,
                'page_count': page_count,
                'total_ids': len(all_ids)
            }, f)
        logging.info(f"第 {page_count} 页完成，累计 {len(all_ids)} 个 ID")
    
    logging.info(f"遍历完成，共 {page_count} 页，最终收集 {len(all_ids)} 个公开非空收藏集")
    return all_ids

from nimare.generate import create_neurovault_studyset
from nimare.transforms import ImageTransformer
import os, logging

def build_neurovault_studyset_from_ids(
    id_list,
    data_dir='./neurovault_images',
    output_file='neurovault_studyset.pkl.gz',
    batch_size=100,
    contrasts=None,
    map_type_conversion=None,
    target='mni152_2mm'
):
    """
    使用新版 Studyset API 分批下载 NeuroVault 图像。
    """
    os.makedirs(data_dir, exist_ok=True)

    # 默认不筛选，下载所有图像
    if contrasts is None:
        contrasts = {}
    if map_type_conversion is None:
        map_type_conversion = {
            "Z map": "z", "T map": "t", "F map": "f",
            "p map": "p", "U map": "u", "R map": "r",
            "Chi2 map": "chi2", "Other": "other",
        }

    n_batches = (len(id_list) + batch_size - 1) // batch_size
    all_studs = []

    for i in range(n_batches):
        batch_ids = id_list[i*batch_size : (i+1)*batch_size]
        batch_file = os.path.join(data_dir, f'neurovault_batch_{i:04d}.pkl.gz')

        # 断点续传：如果批次文件已存在且有效，直接加载
        if os.path.exists(batch_file):
            try:
                import pandas as pd
                batch_stud = pd.read_pickle(batch_file)
                all_studs.append(batch_stud)
                logging.info(f"批次 {i+1}/{n_batches} 已存在，跳过下载")
                continue
            except Exception as e:
                logging.warning(f"批次文件损坏，重新下载: {e}")

        try:
            logging.info(f"下载批次 {i+1}/{n_batches} （{len(batch_ids)} 个集合）...")
            # 核心调用：创建 Studyset
            batch_stud = create_neurovault_studyset(
                collection_ids=batch_ids,
                contrasts=contrasts,
                img_dir=data_dir,
                map_type_conversion=map_type_conversion
            )
            # 保存 Studyset（它本身是一个可序列化的对象）
            batch_stud.save(batch_file)
            all_studs.append(batch_stud)
            logging.info(f"批次 {i+1} 保存成功")
        except Exception as e:
            logging.error(f"批次 {i+1} 失败: {e}")
            continue

    if not all_studs:
        raise RuntimeError("没有成功下载任何批次")

    # 合并所有 Studyset
    final_stud = all_studs[0]
    for s in all_studs[1:]:
        final_stud = final_stud.merge(s)

    final_stud.save(output_file)
    logging.info(f"最终 Studyset 已保存至 {output_file}")
    return final_stud


all_ids = fetch_all_public_collection_ids('neurovault_ids.json')
# 加载 ID 列表
with open('neurovault_ids.json') as f:
    ids_data = json.load(f)
    all_ids = ids_data['ids']

# 测试：仅取前 200 个 ID 进行流程验证
test_dset = build_neurovault_studyset_from_ids(
    all_ids[:200],
    data_dir='./test_nv_images',
    output_studyset='neurovault_test.pkl.gz',
    batch_size=50
)
