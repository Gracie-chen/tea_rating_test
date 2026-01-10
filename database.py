import json
import pandas as pd
import os
import time
import shutil
import numpy as np               # [新增]
import dashscope                 # [新增]
from dashscope import TextEmbedding # [新增]
# ==========================================
# 1. 路径与字段定义
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data") 

# 定义两个数据库文件名
FILE_INITIAL = "initial_case.csv"   # 原始判例库
FILE_ADJUSTED = "adjusted_case.csv" # 修正后的数据
FILE_MANUAL = "knowledge_manual.txt" # 行业权威手册

PATH_INITIAL = os.path.join(DATA_DIR, FILE_INITIAL)
PATH_ADJUSTED = os.path.join(DATA_DIR, FILE_ADJUSTED)
PATH_MANUAL = os.path.join(DATA_DIR, FILE_MANUAL)
PATH_VECTORS = os.path.join(DATA_DIR, "vectors.npy")       # [新增] 存向量数据
PATH_VECTOR_META = os.path.join(DATA_DIR, "vectors_meta.json") # [新增] 存对应的ID映射
# 定义统一的字段格式 (严格按照你的要求)
COLUMNS_SCHEMA = [
    "name", "type", "input_review", "input_context", "expert_summary", "timestamp",
    "score_优雅性", "reason_优雅性", 
    "score_辨识度", "reason_辨识度", 
    "score_协调性", "reason_协调性", 
    "score_饱和度", "reason_饱和度", 
    "score_持久性", "reason_持久性", 
    "score_苦涩度", "reason_苦涩度"
]

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ==========================================
# 2. 核心功能函数
# ==========================================

def _init_csv_if_not_exists(file_path):
    """初始化指定的 CSV 文件"""
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=COLUMNS_SCHEMA)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')

def load_all_cases():
    """
    读取逻辑：同时读取【原始库】和【修正库】，合并后供 AI 参考 (RAG)
    """
    _init_csv_if_not_exists(PATH_INITIAL)
    _init_csv_if_not_exists(PATH_ADJUSTED)
    
    dfs = []
    for p in [PATH_INITIAL, PATH_ADJUSTED]:
        try:
            # 尝试读取，如果文件损坏或为空则跳过
            d = pd.read_csv(p, encoding='utf-8-sig')
            # 简单的列校验，防止空文件报错
            if not d.empty:
                dfs.append(d)
        except Exception:
            pass
    
    if dfs:
        # 合并两个数据源，reset_index 防止索引冲突
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=COLUMNS_SCHEMA)

def flatten_case_data(nested_data):
    """将嵌套字典拍平为符合 Schema 的格式"""
    # 基础字段
    flat_row = {
        "name": nested_data.get("name", "未命名"),
        "type": nested_data.get("type", "通用"),
        "input_review": nested_data.get("input_review", ""),
        "input_context": nested_data.get("input_context", ""),
        "expert_summary": nested_data.get("expert_summary", ""),
        "timestamp": nested_data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    }

    # 分数与理由 (动态提取，防止 Key 报错)
    scores = nested_data.get("scores", {})
    reasons = nested_data.get("reasons", {})
    
    factors = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]
    for f in factors:
        flat_row[f"score_{f}"] = scores.get(f, 0.0) # 默认为 0
        flat_row[f"reason_{f}"] = reasons.get(f, "")  # 默认为空

    return flat_row

def insert_case(case_data_dict, target="adjusted"):
    """
    插入数据
    :param target: "initial" (原始库) 或 "adjusted" (修正库)
    """
    # 1. 确定目标文件
    if target == "initial":
        target_path = PATH_INITIAL
    else:
        target_path = PATH_ADJUSTED

    try:
        # 2. 初始化文件确保表头存在
        _init_csv_if_not_exists(target_path)

        # 3. 数据扁平化处理
        flat_row = flatten_case_data(case_data_dict)
        
        # 4. 读取旧数据并追加
        # 注意：这里只读取目标文件，不读取全部，避免混淆
        try:
            current_df = pd.read_csv(target_path, encoding='utf-8-sig')
        except:
            current_df = pd.DataFrame(columns=COLUMNS_SCHEMA)
            
        new_row_df = pd.DataFrame([flat_row])
        
        # 严格按照 COLUMNS_SCHEMA 排序，防止列乱序
        # 缺失的列会自动填充 NaN，我们用 fillna("") 处理好看点
        new_row_df = new_row_df.reindex(columns=COLUMNS_SCHEMA).fillna("")
        
        updated_df = pd.concat([current_df, new_row_df], ignore_index=True)
        
        # 5. 写入
        updated_df.to_csv(target_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 数据已保存至 {target}: {target_path}")
        return True, f"保存成功 (库: {target})"
        
    except Exception as e:
        return False, f"保存失败: {str(e)}"

# 保持 JSON 兼容性 (如果你还想保留 JSON 备份功能的话，不保留可删除)
KB_FILE = os.path.join(DATA_DIR, "tea_knowledge_base.json")
def load_json_kb():
    # ... (保持原样)
    if not os.path.exists(KB_FILE): return []
    try:
        with open(KB_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except: return []

def save_json_kb(record):
    # ... (保持原样)
    data = load_json_kb()
    data.append(record)
    with open(KB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# [新增代码块]

def get_embedding(text):
    """调用阿里云生成 Embedding (单条)"""
    try:
        # 注意：调用此函数前需确保 dashscope.api_key 已在 logic.py 或 app.py 中设置
        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v1,
            input=text
        )
        if resp.status_code == 200:
            return resp.output.embeddings[0].embedding
        else:
            print(f"Embedding API Error: {resp}")
            return None
    except Exception as e:
        print(f"Embedding Exception: {e}")
        return None

def refresh_vector_index(all_cases):
    """
    全量/增量刷新向量库
    逻辑：遍历所有案例 -> 检查是否有向量 -> 没有则计算 -> 保存
    """
    # 1. 加载旧数据
    if os.path.exists(PATH_VECTORS) and os.path.exists(PATH_VECTOR_META):
        try:
            vectors = np.load(PATH_VECTORS)
            with open(PATH_VECTOR_META, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except:
            vectors = np.array([])
            meta = []
    else:
        vectors = None
        meta = []

    # 2. 找出新数据 (简单的去重逻辑)
    existing_reviews = set([m.get('input_review', '') for m in meta])
    
    new_vectors = []
    new_meta = []
    dirty = False 
    
    # 遍历传入的 DataFrame
    for _, row in all_cases.iterrows():
        txt = str(row.get('input_review', ''))
        # 如果文本太短或已存在，跳过
        if len(txt) < 2 or txt in existing_reviews:
            continue
            
        # 计算新向量
        # print(f"生成向量: {txt[:10]}...") 
        emb = get_embedding(txt)
        if emb:
            new_vectors.append(emb)
            # 记录元数据，方便检索时找回
            # 将 series 转为 dict，并处理可能的 NaN
            r_dict = row.to_dict()
            new_meta.append(r_dict) 
            dirty = True
            
    # 3. 合并保存
    if dirty:
        if vectors is not None and len(vectors) > 0:
            final_vectors = np.vstack([vectors, np.array(new_vectors)])
            final_meta = meta + new_meta
        else:
            final_vectors = np.array(new_vectors)
            final_meta = new_meta
            
        np.save(PATH_VECTORS, final_vectors)
        with open(PATH_VECTOR_META, 'w', encoding='utf-8') as f:
            json.dump(final_meta, f, ensure_ascii=False)
        print(f"向量库已更新，当前共 {len(final_meta)} 条。")
    
    return True

def load_vector_store():
    """读取向量库供 logic.py 使用"""
    if os.path.exists(PATH_VECTORS) and os.path.exists(PATH_VECTOR_META):
        try:
            vectors = np.load(PATH_VECTORS)
            with open(PATH_VECTOR_META, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            return vectors, meta
        except:
            return None, []
    return None, []