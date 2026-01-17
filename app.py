import streamlit as st
import os
import json
import numpy as np
import faiss
import time
import pickle
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from docx import Document
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from openai import OpenAI

# ==========================================
# 0. 基础配置与持久化路径
# ==========================================
st.set_page_config(
    page_title="茶饮六因子AI评分器 (Local Pro)",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 定义记忆存储目录
DATA_DIR = Path("./tea_data")
DATA_DIR.mkdir(exist_ok=True) 

# 定义文件路径
PATHS = {
    "kb_index": DATA_DIR / "kb.index",
    "kb_chunks": DATA_DIR / "kb_chunks.pkl",
    "case_index": DATA_DIR / "cases.index",
    "case_data": DATA_DIR / "cases.json",
    # 注意：这里改为 LLaMA-Factory 兼容的训练数据路径
    "training_data": DATA_DIR / "tea_finetune.json", 
    "prompt": DATA_DIR / "prompts.json"
}

# 样式
st.markdown("""
    <style>
    .main-title {font-size: 2.5em; font-weight: bold; text-align: center; color: #2E7D32; margin-bottom: 0.5em;}
    .slogan {font-size: 1.2em; font-style: italic; text-align: center; color: #558B2F; margin-bottom: 30px; font-family: "KaiTi", "楷体", serif;}
    .factor-card {background-color: #F1F8E9; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #4CAF50;}
    .score-header {display:flex; justify-content:space-between; font-weight:bold; color:#2E7D32;}
    .advice-tag {font-size: 0.85em; padding: 2px 6px; border-radius: 4px; margin-top: 5px; background-color: #fff; border: 1px dashed #4CAF50; color: #388E3C; display: inline-block;}
    .master-comment {background-color: #FFFDE7; border: 1px solid #FFF9C4; padding: 15px; border-radius: 8px; font-family: "KaiTi", serif; font-size: 1.1em; color: #5D4037; margin-bottom: 20px; line-height: 1.6;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心数据管理
# ==========================================

class DataManager:
    @staticmethod
    def save(index, data, idx_path, data_path, is_json=False):
        if index: faiss.write_index(index, str(idx_path))
        with open(data_path, "w" if is_json else "wb") as f:
            if is_json: json.dump(data, f, ensure_ascii=False, indent=2)
            else: pickle.dump(data, f)
    
    @staticmethod
    def load(idx_path, data_path, is_json=False):
        if idx_path.exists() and data_path.exists():
            try:
                index = faiss.read_index(str(idx_path))
                with open(data_path, "r" if is_json else "rb") as f:
                    data = json.load(f) if is_json else pickle.load(f)
                return index, data
            except: pass
        # 默认返回 384 维索引 (适配 all-MiniLM-L6-v2)
        # 如果你之前运行过旧代码，建议删除 ./tea_data 下的 .index 文件重新生成，否则维度不匹配会报错
        return faiss.IndexFlatL2(384), [] 
    
    @staticmethod
    def append_to_finetune_dataset(user_input, scores, system_prompt, master_comment):
        """
        核心微调逻辑：将校准后的数据保存为 LLaMA-Factory 兼容的 Alpaca 格式 (JSON List)
        """
        try:
            # 1. 构造期望的模型输出 (JSON)
            target_output = json.dumps({
                "master_comment": master_comment,
                "scores": scores
            }, ensure_ascii=False)
            
            # 2. 构造一条训练数据
            new_entry = {
                "instruction": system_prompt,
                "input": user_input,
                "output": target_output
            }
            
            # 3. 读取现有文件或创建新列表
            current_data = []
            if PATHS['training_data'].exists():
                try:
                    with open(PATHS['training_data'], "r", encoding="utf-8") as f:
                        current_data = json.load(f)
                        if not isinstance(current_data, list): current_data = []
                except: current_data = []
            
            # 4. 追加并保存
            current_data.append(new_entry)
            with open(PATHS['training_data'], "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=2)
            
            return len(current_data)
        except Exception as e:
            print(f"[ERROR] append_to_finetune 失败: {str(e)}")
            return 0

# 本地 Embedder，使用 sentence-transformers
class LocalEmbedder:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            # 使用轻量级模型，速度快，适合 CPU/单卡
            # 第一次运行会自动下载模型 (~80MB)
            self.model = SentenceTransformer('all-MiniLM-L6-v2') 
            self.dim = 384
        except Exception as e:
            st.error(f"Embedding 模型加载失败，请确保安装了 sentence-transformers: {e}")
            self.model = None
            self.dim = 384

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts or not self.model: return np.zeros((0, self.dim), dtype="float32")
        if isinstance(texts, str): texts = [texts]
        try:
            embeddings = self.model.encode(texts)
            return np.array(embeddings).astype("float32")
        except: 
            return np.zeros((len(texts), self.dim), dtype="float32")

# 默认 Prompt (保持不变)
DEFAULT_PROMPT_CONFIG = {
    "system_template": """你是一名资深的茶饮产品研发与感官分析专家。
请基于给定的产品描述、参考资料和相似历史判例，严格按照"罗马测评法2.0"进行专业评分。

====================
一、评分方法
====================
六因子（0-9分）：
1. 优雅性：香气愉悦感
2. 辨识度：香气记忆点
3. 协调性：融合度
4. 饱和度：浓厚度
5. 持久性：余韵
6. 苦涩度：舒适度（分数越高越舒适，越不苦）

====================
二、输出约束
====================
请直接输出 JSON 格式，包含 "master_comment" 和 "scores" 两个字段。不要输出任何 Markdown 标记或多余的解释。""",
    
    "user_template": """【待评分产品】
{product_desc}

【参考标准】
{context_text}

【历史判例】
{case_text}

请输出JSON结果："""
}

# ==========================================
# 2. 逻辑函数
# ==========================================

# 核心评分函数
def run_scoring(text, kb_res, case_res, prompt_cfg, embedder, client, model_id, r_num, c_num): 
    vec = embedder.encode([text])
    
    # RAG 检索
    ctx_txt, hits = "（无资料）", []
    if kb_res[0].ntotal > 0: 
        _, idx = kb_res[0].search(vec, r_num)
        hits = [kb_res[1][i] for i in idx[0] if i < len(kb_res[1]) and i >= 0]
        if hits: ctx_txt = "\n".join([f"- {h[:150]}..." for h in hits])
        
    # 判例检索
    case_txt, found_cases = "（无判例）", []
    if case_res[0].ntotal > 0: 
        _, idx = case_res[0].search(vec, c_num)
        for i in idx[0]:
            if i < len(case_res[1]) and i >= 0:
                c = case_res[1][i]
                found_cases.append(c)
                # 简化判例展示，节省 Context Window
                sc = c.get('scores', {})
                u_sc = sc.get('优雅性',{}).get('score', '-')
                case_txt += f"\n- {c['text'][:30]}... (优雅:{u_sc})"

    sys_p = prompt_cfg.get('system_template', DEFAULT_PROMPT_CONFIG['system_template'])
    user_p = prompt_cfg.get('user_template', DEFAULT_PROMPT_CONFIG['user_template']).format(
        product_desc=text, context_text=ctx_txt, case_text=case_txt
    )

    try:
        # 调用本地 vLLM
        resp = client.chat.completions.create(
            model=model_id, 
            messages=[{"role":"system", "content":sys_p}, {"role":"user", "content":user_p}],
            temperature=0.3,
            max_tokens=1024,
            # Qwen2.5 支持 json_object 模式，确保输出格式稳定
            response_format={"type": "json_object"} 
        )
        content = resp.choices[0].message.content
        return json.loads(content), hits, found_cases
    except Exception as e:
        st.error(f"推理错误 (请检查 vLLM 是否启动): {e}")
        return None, [], []

# 风味形态图
def calculate_section_scores(scores):
    s = scores["scores"]
    def g(k): return s.get(k, {}).get("score", 0)
    top  = (g("优雅性") + g("辨识度")) / 2
    mid  = (g("协调性") + g("饱和度")) / 2
    base = (g("持久性") + g("苦涩度")) / 2
    return top, mid, base

def plot_flavor_shape(scores_data):
    top, mid, base = calculate_section_scores(scores_data)
    fig, ax = plt.subplots(figsize=(4, 5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    y = np.array([1, 2, 3]) 
    x = np.array([base, mid, top])
    y_new = np.linspace(1, 3, 300)
    try:
        spl = make_interp_spline(y, x, k=2)
        x_smooth = spl(y_new)
    except:
        x_smooth = np.interp(y_new, y, x)
    x_smooth = np.maximum(x_smooth, 0.1)
    
    # 简单的可视化填充
    ax.fill_betweenx(y_new, -x_smooth, x_smooth, color='#4CAF50', alpha=0.6)
    ax.text(0, 2.7, f"前调 {top:.1f}", ha='center', color='white', fontweight='bold')
    ax.text(0, 2.0, f"中调 {mid:.1f}", ha='center', color='white', fontweight='bold')
    ax.text(0, 1.3, f"后调 {base:.1f}", ha='center', color='white', fontweight='bold')
    ax.axis('off')
    ax.set_xlim(-10, 10)
    return fig

# ==========================================
# 3. 页面初始化
# ==========================================

if'loaded' not in st.session_state:
    # 第一次加载时，如果发现 index 维度不匹配（例如之前是1024，现在是384），需要处理
    # 这里简单处理：如果报错就重建空的
    try:
        kb_idx, kb_data = DataManager.load(PATHS['kb_index'], PATHS['kb_chunks'])
    except:
        kb_idx, kb_data = faiss.IndexFlatL2(384), []
        
    try:
        case_idx, case_data = DataManager.load(PATHS['case_index'], PATHS['case_data'], is_json=True)
    except:
        case_idx, case_data = faiss.IndexFlatL2(384), []

    st.session_state.kb = (kb_idx, kb_data)
    st.session_state.cases = (case_idx, case_data)
    
    if PATHS['prompt'].exists():
        try:
            with open(PATHS['prompt'], 'r') as f: st.session_state.prompt_config = json.load(f)
        except: st.session_state.prompt_config = DEFAULT_PROMPT_CONFIG.copy()
    else:
        st.session_state.prompt_config = DEFAULT_PROMPT_CONFIG.copy()
    
    # 初始化 Embedder
    st.session_state.embedder = LocalEmbedder()
    
    st.session_state.loaded = True

# 初始化 OpenAI Client (指向 vLLM)
# 请确保你的 vLLM 正在运行于 port 8000
client = OpenAI(
    api_key="EMPTY", 
    base_url="http://localhost:8000/v1"
)

# 侧边栏
with st.sidebar:
    st.header("⚙️ 本地配置")
    st.success("🟢 已连接本地 vLLM")
    
    model_name = "Qwen2.5-7B-Instruct" # 必须与 vLLM 启动参数一致
    st.caption(f"当前模型: {model_name}")
    
    st.markdown("---")
    st.markdown("**数据统计**")
    st.caption(f"RAG片段: {len(st.session_state.kb[1])} 条")
    st.caption(f"历史判例: {len(st.session_state.cases[1])} 条")
    
    if PATHS['training_data'].exists():
        try:
            with open(PATHS['training_data'], 'r') as f:
                d = json.load(f)
            st.caption(f"💪 **待微调数据: {len(d)} 条**")
        except: pass
    
    if st.button("🗑️ 清空所有数据 (慎点)"):
        import shutil
        shutil.rmtree(DATA_DIR)
        DATA_DIR.mkdir()
        st.warning("数据已清空，请刷新页面")

st.markdown('<div class="main-title">🍵 茶品 AI 评分器 (vLLM版)</div>', unsafe_allow_html=True)

# ==========================================
# 4. 功能标签页
# ==========================================
tab1, tab2 = st.tabs(["💡 交互评分与校准", "🚀 微调数据中心"])

# --- Tab 1: 交互评分 ---
with tab1:
    st.info("💡 流程：输入茶评 -> AI 评分 -> **专家人工校准** -> 存入训练库")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # 使用 Session State 保持输入
        if'user_input' not in st.session_state: st.session_state.user_input = ""
        user_input = st.text_area("输入茶评描述:", value=st.session_state.user_input, height=120)
        st.session_state.user_input = user_input
        
        if st.button("🚀 开始评分", type="primary"):
            if not user_input: st.warning("请输入内容")
            else:
                with st.spinner(f"AI 正在思考..."):
                    scores, kb_hits, case_hits = run_scoring(
                        user_input, st.session_state.kb, st.session_state.cases,
                        st.session_state.prompt_config, st.session_state.embedder, client, model_name, 3, 2
                    )
                    if scores:
                        st.session_state.last_scores = scores
                        st.session_state.last_master = scores.get("master_comment", "")
                        st.rerun() # 刷新页面显示结果

    # 显示结果区域
    if'last_scores' in st.session_state and st.session_state.last_scores:
        scores = st.session_state.last_scores
        
        st.markdown("---")
        st.subheader("📊 评分结果 (请专家校准)")
        
        # 左右分栏：左边是可视化，右边是校准表单
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown(f"**AI 生成总评:**\n\n> {st.session_state.last_master}")
            fig = plot_flavor_shape(scores)
            st.pyplot(fig)
        
        with res_col2:
            with st.form("calibration_form"):
                st.markdown("#### ✍️ 专家校准面板")
                st.caption("请修正 AI 的评分，您的修正将成为模型变强的养料。")
                
                # 1. 校准总评
                new_master = st.text_area("宗师总评 (校准)", value=st.session_state.last_master, height=80)
                
                # 2. 校准六因子
                factors = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]
                s_dict = scores.get("scores", {})
                new_scores = {}
                
                c1, c2 = st.columns(2)
                for i, f in enumerate(factors):
                    with (c1 if i % 2 == 0 else c2):
                        current_data = s_dict.get(f, {})
                        val = st.slider(f"{f}", 0, 9, int(current_data.get("score", 5)))
                        cmt = st.text_input(f"评语 ({f})", current_data.get("comment", ""))
                        sug = st.text_input(f"建议 ({f})", current_data.get("suggestion", ""))
                        
                        new_scores[f] = {"score": val, "comment": cmt, "suggestion": sug}
                
                submitted = st.form_submit_button("✅ 确认校准并保存到训练库", type="primary")
                
                if submitted:
                    # 保存到微调数据文件
                    sys_p = st.session_state.prompt_config['system_template']
                    count = DataManager.append_to_finetune_dataset(
                        user_input, new_scores, sys_p, new_master
                    )
                    
                    # 同时也保存到判例库 (RAG)
                    new_case = {"text": user_input, "scores": new_scores, "master_comment": new_master, "tags": "人工校准"}
                    st.session_state.cases[1].append(new_case)
                    vec = st.session_state.embedder.encode([user_input])
                    st.session_state.cases[0].add(vec)
                    DataManager.save(st.session_state.cases[0], st.session_state.cases[1], PATHS['case_index'], PATHS['case_data'], is_json=True)
                    
                    st.success(f"🎉 保存成功！当前训练数据量: {count} 条")
                    time.sleep(1)
                    st.rerun()

# --- Tab 2: 微调数据中心 ---
with tab2:
    st.header("🏭 微调数据工厂")
    st.markdown("""
    这里存放了你在前台校准过的所有数据。
    **使用步骤:**
    1. 点击下方按钮下载 `dataset.json`。
    2. 将文件放入服务器 `LLaMA-Factory/data` 文件夹。
    3. 启动 LLaMA-Factory WebUI 进行微调。
    """)
    
    if PATHS['training_data'].exists():
        with open(PATHS['training_data'], 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        st.write(f"📊 当前已积累优质数据: **{len(raw_data)}** 条")
        
        # 数据预览
        with st.expander("🔍 预览最后 3 条数据"):
            st.json(raw_data[-3:] if len(raw_data) > 3 else raw_data)
        
        # 下载按钮
        json_str = json.dumps(raw_data, ensure_ascii=False, indent=2)
        st.download_button(
            label="⬇️ 下载 dataset.json (LLaMA-Factory专用)",
            data=json_str,
            file_name="tea_finetune.json",
            mime="application/json"
        )
    else:
        st.warning("暂无数据，请去【交互评分与校准】页面进行打标。")

    st.markdown("---")
    st.subheader("📚 RAG 知识库管理")
    up_files = st.file_uploader("上传 PDF/TXT 补充知识库", accept_multiple_files=True)
    if up_files and st.button("更新知识库"):
        with st.spinner("正在向量化..."):
            raw_text = ""
            for f in up_files:
                if f.name.endswith(".txt"): raw_text += f.read().decode("utf-8")
                elif f.name.endswith(".pdf"): 
                    reader = PdfReader(f)
                    for page in reader.pages: raw_text += page.extract_text()
            
            # 简单切分
            chunk_size = 300
            chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
            
            # 向量化
            vecs = st.session_state.embedder.encode(chunks)
            st.session_state.kb[0].add(vecs)
            st.session_state.kb[1].extend(chunks)
            
            # 保存
            DataManager.save(st.session_state.kb[0], st.session_state.kb[1], PATHS['kb_index'], PATHS['kb_chunks'])
            st.success(f"已新增 {len(chunks)} 条知识片段！")
