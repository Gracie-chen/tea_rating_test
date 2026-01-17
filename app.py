import streamlit as st
import os
import json
import numpy as np
import faiss
import time
import pickle
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional
from PyPDF2 import PdfReader
from http import HTTPStatus
import dashscope
from dashscope import TextEmbedding
from openai import OpenAI
from docx import Document
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ==========================================
# [SECTION 0] 基础配置与路径定义
# ==========================================

st.set_page_config(
    page_title="茶饮六因子AI评分器 Pro",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 样式定义
st.markdown("""
    <style>
    .main-title {font-size: 2.5em; font-weight: bold; text-align: center; color: #2E7D32; margin-bottom: 0.5em;}
    .slogan {font-size: 1.2em; font-style: italic; text-align: center; color: #558B2F; margin-bottom: 30px; font-family: "KaiTi", "楷体", serif;}
    .factor-card {background-color: #F1F8E9; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #4CAF50;}
    .score-header {display:flex; justify-content:space-between; font-weight:bold; color:#2E7D32;}
    .advice-tag {font-size: 0.85em; padding: 2px 6px; border-radius: 4px; margin-top: 5px; background-color: #fff; border: 1px dashed #4CAF50; color: #388E3C; display: inline-block;}
    .master-comment {background-color: #FFFDE7; border: 1px solid #FFF9C4; padding: 15px; border-radius: 8px; font-family: "KaiTi", serif; font-size: 1.1em; color: #5D4037; margin-bottom: 20px; line-height: 1.6;}
    .ft-card {border: 1px solid #ddd; padding: 15px; border-radius: 8px; background-color: #f8f9fa; margin-top: 10px;}
    </style>
""", unsafe_allow_html=True)

class PathConfig:
    """路径管理类"""
    # 外部资源文件（位于同级目录）
    SRC_SYS_PROMPT = Path("sys_p.txt")
    SRC_SEED_CASES = Path("seed_case.json")

    # 运行时数据目录
    DATA_DIR = Path("./tea_data")
    
    def __init__(self):
        self.DATA_DIR.mkdir(exist_ok=True)
        # 向量库与持久化数据
        self.kb_index = self.DATA_DIR / "kb.index"
        self.kb_chunks = self.DATA_DIR / "kb_chunks.pkl"
        self.case_index = self.DATA_DIR / "cases.index"
        self.case_data = self.DATA_DIR / "cases.json"
        
        # 微调与Prompt配置
        self.training_file = self.DATA_DIR / "deepseek_finetune.jsonl"
        self.ft_status = self.DATA_DIR / "ft_status.json"
        self.prompt_config_file = self.DATA_DIR / "prompts.json"

PATHS = PathConfig()

# 默认的用户Prompt模板（System Prompt将从文件读取）
DEFAULT_USER_TEMPLATE = """【待评分产品】
{product_desc}

【参考标准（知识库）】
{context_text}

【相似判例得分参考（案例库）】
{case_text}

请严格输出以下JSON格式（不含Markdown）：
{{
  "master_comment": "约100字的宗师级总评，富含文化意蕴...",
  "scores": {{
    "优雅性": {{"score": 0-9, "comment": "...", "suggestion": "..."}},
    "辨识度": {{"score": 0-9, "comment": "...", "suggestion": "..."}},
    "协调性": {{"score": 0-9, "comment": "...", "suggestion": "..."}},
    "饱和度": {{"score": 0-9, "comment": "...", "suggestion": "..."}},
    "持久性": {{"score": 0-9, "comment": "...", "suggestion": "..."}},
    "苦涩度": {{"score": 0-9, "comment": "...", "suggestion": "..."}}
  }}
}}"""

# ==========================================
# [SECTION 1] 资源与数据管理
# ==========================================

class ResourceManager:
    """负责外部文件加载、数据持久化及格式转换"""

    @staticmethod
    def load_external_text(path: Path, fallback: str = "") -> str:
        """读取外部文本文件 (如 sys_p.txt)"""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception as e:
                st.error(f"加载文件 {path} 失败: {e}")
        return fallback

    @staticmethod
    def load_external_json(path: Path, fallback: Any = None) -> Any:
        """读取外部JSON文件 (如 seed_case.json)"""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"加载文件 {path} 失败: {e}")
        return fallback if fallback is not None else []

    @staticmethod
    def save(index: Any, data: Any, idx_path: Path, data_path: Path, is_json: bool = False):
        """保存 FAISS 索引和数据文件"""
        if index: faiss.write_index(index, str(idx_path))
        with open(data_path, "w" if is_json else "wb") as f:
            if is_json: json.dump(data, f, ensure_ascii=False, indent=2)
            else: pickle.dump(data, f)
    
    @staticmethod
    def load(idx_path: Path, data_path: Path, is_json: bool = False) -> Tuple[Any, List]:
        """加载 FAISS 索引和数据文件"""
        if idx_path.exists() and data_path.exists():
            try:
                index = faiss.read_index(str(idx_path))
                with open(data_path, "r" if is_json else "rb") as f:
                    data = json.load(f) if is_json else pickle.load(f)
                return index, data
            except: pass
        return faiss.IndexFlatL2(1024), []

# 以下三个方法用于微调
    @staticmethod
    def append_to_finetune(case_text: str, scores: Dict, sys_prompt: str, user_tpl: str, master_comment: str = "（人工校准）") -> bool:
        """将判例写入微调数据集 (.jsonl)"""
        try:
            user_content = user_tpl.format(product_desc=case_text, context_text="", case_text="")
            assistant_content = json.dumps({"master_comment": master_comment, "scores": scores}, ensure_ascii=False)
            entry = {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
            with open(PATHS.training_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            print(f"[ERROR] Finetune append failed: {e}")
            return False

    @staticmethod
    def save_ft_status(job_id, status, fine_tuned_model=None):
        data = {"job_id": job_id, "status": status, "timestamp": time.time()}
        if fine_tuned_model: data["fine_tuned_model"] = fine_tuned_model
        with open(PATHS.ft_status, 'w') as f: json.dump(data, f)

    @staticmethod
    def load_ft_status():
        if PATHS.ft_status.exists():
            try: return json.load(open(PATHS.ft_status, 'r'))
            except: pass
        return None

# ==========================================
# [SECTION 2] AI 服务 (Embedding & LLM)
# ==========================================

class AliyunEmbedder:
    def __init__(self, api_key):
        self.model_name = "text-embedding-v4"
        dashscope.api_key = api_key 

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts: return np.zeros((0, 1024), dtype="float32")
        if isinstance(texts, str): texts = [texts]
        try:
            resp = TextEmbedding.call(model=self.model_name, input=texts)
            if resp.status_code == HTTPStatus.OK:
                return np.array([i['embedding'] for i in resp.output['embeddings']]).astype("float32")
        except: pass
        return np.zeros((len(texts), 1024), dtype="float32")

def run_scoring(text: str, kb_res: Tuple, case_res: Tuple, prompt_cfg: Dict, embedder: AliyunEmbedder, client: OpenAI, model_id: str, k_num: int, c_num: int):
    """执行 RAG 检索与 LLM 评分"""
    # 1. 向量化与 RAG 检索
    vec = embedder.encode([text]) 
    
    ctx_txt, hits = "（无手册资料）", []
    if kb_res[0].ntotal > 0:
        _, idx = kb_res[0].search(vec, k_num)
        hits = [kb_res[1][i] for i in idx[0] if i < len(kb_res[1])]
        ctx_txt = "\n".join([f"- {h[:200]}..." for h in hits])

    # 如果后续用Lora微调方法的话是否是考虑删除这一段few-shot    
    case_txt, found_cases = "（无相似判例）", []
    if case_res[0].ntotal > 0:
        _, idx = case_res[0].search(vec, c_num)
        for i in idx[0]:
            if i < len(case_res[1]) and i >= 0:
                c = case_res[1][i]
                found_cases.append(c)
                sc = c.get('scores', {})
                u_sc = sc.get('优雅性',{}).get('score', 0) if isinstance(sc,dict) and '优雅性' in sc else 0
                k_sc = sc.get('苦涩度',{}).get('score', 0) if isinstance(sc,dict) and '苦涩度' in sc else 0
                case_txt += f"\n参考案例: {c['text'][:30]}... -> 优雅性:{u_sc} 苦涩度:{k_sc}"

    # 2. 组装 Prompt
    sys_p = prompt_cfg.get('system_template', "")
    user_p = prompt_cfg.get('user_template', "").format(product_desc=text, context_text=ctx_txt, case_text=case_txt)

    # 3. 调用 LLM
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role":"system", "content":sys_p}, {"role":"user", "content":user_p}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        return json.loads(resp.choices[0].message.content), hits, found_cases
    except Exception as e:
        st.error(f"Inference Error: {e}")
        return None, [], []

# ==========================================
# [SECTION 3] 辅助与可视化
# ==========================================

def parse_file(uploaded_file) -> str:
    """解析上传文件"""
    try:
        if uploaded_file.name.endswith('.txt'): return uploaded_file.read().decode("utf-8")
        if uploaded_file.name.endswith('.pdf'): return "".join([p.extract_text() for p in PdfReader(uploaded_file).pages])
        if uploaded_file.name.endswith('.docx'): return "\n".join([p.text for p in Document(uploaded_file).paragraphs])
    except: return ""
    return ""

def create_word_report(results: List[Dict]) -> BytesIO:
    """生成Word报告"""
    doc = Document()
    doc.add_heading("茶评批量评分报告", 0)
    for item in results:
        doc.add_heading(f"条目 {item['id']}", 1)
        doc.add_paragraph(f"原文：{item['text']}")
        s = item.get('scores', {}).get('scores', {})
        mc = item.get('scores', {}).get('master_comment', '')
        if mc: doc.add_paragraph(f"总评：{mc}", style="Intense Quote")
        
        table = doc.add_table(rows=1, cols=4)
        table.style = 'Table Grid'
        hdr = table.rows[0].cells
        hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = '因子', '分数', '评语', '建议'
        for k, v in s.items():
            r = table.add_row().cells
            r[0].text = k
            r[1].text = str(v.get('score',''))
            r[2].text = v.get('comment','')
            r[3].text = v.get('suggestion','')
        doc.add_paragraph("_"*20)
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

def plot_flavor_shape(scores_data: Dict):
    """绘制风味形态图"""
    s = scores_data["scores"]
    top = (s["优雅性"]["score"] + s["辨识度"]["score"]) / 2
    mid = (s["协调性"]["score"] + s["饱和度"]["score"]) / 2
    base = (s["持久性"]["score"] + s["苦涩度"]["score"]) / 2
    
    fig, ax = plt.subplots(figsize=(4, 5))
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)

    y = np.array([1, 2, 3]) 
    x = np.array([base, mid, top])
    y_new = np.linspace(1, 3, 300)
    try:
        spl = make_interp_spline(y, x, k=2)
        x_smooth = spl(y_new)
    except:
        x_smooth = np.interp(y_new, y, x)
    x_smooth = np.maximum(x_smooth, 0.1)

    colors = {'base': '#8B4513', 'mid': '#D2691E', 'top': '#FFD700'}
    for mask, col in [((y_new>=1.0)&(y_new<=1.6), colors['base']), 
                      ((y_new>1.6)&(y_new<=2.4), colors['mid']), 
                      ((y_new>2.4)&(y_new<=3.0), colors['top'])]:
        ax.fill_betweenx(y_new[mask], -x_smooth[mask], x_smooth[mask], color=col, alpha=0.9, edgecolor=None)

    ax.plot(x_smooth, y_new, 'k', linewidth=1, alpha=0.2)
    ax.plot(-x_smooth, y_new, 'k', linewidth=1, alpha=0.2)
    ax.axhline(y=1.6, color='w', linestyle=':', alpha=0.5)
    ax.axhline(y=2.4, color='w', linestyle=':', alpha=0.5)
    
    font = {'ha': 'center', 'va': 'center', 'color': 'white', 'fontweight': 'bold', 'fontsize': 12}
    ax.text(0, 2.7, f"Top\n{top:.1f}", **font)
    ax.text(0, 2.0, f"Mid\n{mid:.1f}", **font)
    ax.text(0, 1.3, f"Base\n{base:.1f}", **font)
    ax.axis('off'); ax.set_xlim(-10, 10); ax.set_ylim(0.8, 3.2)
    return fig

def bootstrap_seed_cases(embedder: AliyunEmbedder):
    """
    初始化判例库：如果内存/磁盘中为空，则从 seed_case.json 文件读取。
    """
    case_idx, case_data = st.session_state.cases
    if len(case_data) > 0: return

    # 从外部 JSON 加载
    seed_cases = ResourceManager.load_external_json(PATHS.SRC_SEED_CASES)
    if not seed_cases:
        st.warning("seed_case.json 未找到或为空，判例库初始化跳过。")
        return

    texts = [c["text"] for c in seed_cases]
    vecs = embedder.encode(texts)

    if case_idx.ntotal == 0: case_idx = faiss.IndexFlatL2(1024)
    if len(vecs) > 0:
        case_idx.add(vecs)
        case_data.extend(seed_cases)
        st.session_state.cases = (case_idx, case_data)
        ResourceManager.save(case_idx, case_data, PATHS.case_index, PATHS.case_data, is_json=True)

# ==========================================
# [SECTION 4] 主程序逻辑
# ==========================================

# A. 初始化 Session
if'loaded' not in st.session_state:
    # 1. 加载RAG与判例数据
    kb_idx, kb_data = ResourceManager.load(PATHS.kb_index, PATHS.kb_chunks)
    case_idx, case_data = ResourceManager.load(PATHS.case_index, PATHS.case_data, is_json=True)
    st.session_state.kb = (kb_idx, kb_data)
    st.session_state.cases = (case_idx, case_data)
    
    # 2. 加载 Prompt 配置
    # 优先读取持久化的 prompts.json，如果没有，则从 sys_p.txt 构建默认配置 - 实现prompts修改永久化
    if PATHS.prompt_config_file.exists():
        try:
            with open(PATHS.prompt_config_file, 'r') as f:
                st.session_state.prompt_config = json.load(f)
        except: pass
    
    if'prompt_config' not in st.session_state:
        # 从 sys_p.txt 读取 System Prompt，使用硬编码的 User Prompt
        sys_prompt_content = ResourceManager.load_external_text(PATHS.SRC_SYS_PROMPT, fallback="你是一名茶评专家...")
        st.session_state.prompt_config = {
            "system_template": sys_prompt_content,
            "user_template": DEFAULT_USER_TEMPLATE
        }
    

    st.session_state.loaded = True

# B. 侧边栏
with st.sidebar:
    st.header("⚙️ 系统配置")
    st.markdown("**🔐 API 配置**")
    aliyun_key = os.getenv("ALIYUN_API_KEY") or st.secrets.get("ALIYUN_API_KEY", "")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY", "")

    if not aliyun_key or not deepseek_key:
        st.warning("⚠️ 未配置 API Key")
        st.stop()
    else:
        st.success("✅ API 就绪")

    st.markdown("---")
    st.markdown(f"**预处理模型：** `Deepseek-chat`")
    st.markdown(f"**评分模型：** `Qwen2.5-7B-Instruct`")
    model_id = "Qwen2.5-7B-Instruct"
    # 加载微调模型（如有）
    ft_status = ResourceManager.load_ft_status()
    if ft_status and ft_status.get("status") == "succeeded":
        st.info(f"🎉 发现微调模型：`{ft_status.get('fine_tuned_model')}`")

    embedder = AliyunEmbedder(aliyun_key)
    client = OpenAI(api_key="dummy", base_url="http://117.50.89.74:8000/v1")
    # 确保初始化判例
    bootstrap_seed_cases(embedder)
    # 展示当前RAG与判例容量
    st.markdown("---")
    st.markdown(f"知识库: {len(st.session_state.kb[1])} | 判例库: {len(st.session_state.cases[1])}")
    st.caption("快速上传仅支持.zip文件格式。")
    st.caption("少量文件上传请至\"模型调优\"板块。")
    # 
    if st.button("📤 导出数据"):
        import zipfile, shutil
        temp_dir = Path("./temp_export"); temp_dir.mkdir(exist_ok=True)
        for p in [PATHS.kb_index, PATHS.kb_chunks, PATHS.case_index, PATHS.case_data, PATHS.prompt_config_file]:
            if p.exists(): shutil.copy2(p, temp_dir / p.name)
        zip_path = Path("./rag_export.zip")
        with zipfile.ZipFile(zip_path, 'w') as z:
            for f in temp_dir.iterdir(): z.write(f, f.name)
        with open(zip_path, 'rb') as f:
            st.download_button("⬇️ 下载ZIP", f, "tea_data.zip", "application/zip")
        shutil.rmtree(temp_dir); zip_path.unlink()

    if st.button("📥 导入数据"):
        u_zip = st.file_uploader("上传ZIP", type=['zip'])
        if u_zip:
            import zipfile, tempfile
            with tempfile.TemporaryDirectory() as td:
                zp = Path(td)/"u.zip"
                with open(zp,'wb') as f: f.write(u_zip.getvalue())
                with zipfile.ZipFile(zp,'r') as z: z.extractall(PATHS.DATA_DIR)
                st.success("导入成功，请刷新"); st.rerun()

# C. 主界面
st.markdown('<div class="main-title">🍵 茶品六因子 AI 评分器 Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="slogan">“一片叶子落入水中，改变了水的味道...”</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💡 交互评分", "🚀 批量评分", "🛠️ 模型调优"])

# --- Tab 1: 交互评分 ---
with tab1:
    st.info("将参考知识库与判例库进行评分。确认结果可一键更新判例库。")
    c1, c2, c3, c4, c5 = st.columns([1, 3, 1, 3, 1])
    r_num = c2.number_input("参考知识库条目数量", 1, 20, 3, key="r1")
    c_num = c4.number_input("参考判例库条目数量", 1, 20, 2, key="c1")
    # 使用会话状态存储用户输入，避免刷新后丢失
    if'current_user_input' not in st.session_state: st.session_state.current_user_input = ""
    user_input = st.text_area("请输入茶评描述:", value=st.session_state.current_user_input, height=150, key="ui")
    st.session_state.current_user_input = user_input
    # 使用会话状态存储评分结果
    if'last_scores' not in st.session_state: 
        st.session_state.last_scores = None
        st.session_state.last_master_comment = ""
    
    if st.button("开始评分", type="primary", use_container_width=True):
        if not user_input: st.warning("请输入内容")
        else:
            with st.spinner(f"正在使用 {model_id} 品鉴..."):
                scores, kb_h, case_h = run_scoring(user_input, st.session_state.kb, st.session_state.cases, st.session_state.prompt_config, embedder, client, "Qwen2.5-7B-Instruct", r_num, c_num)
                if scores:
                    st.session_state.last_scores = scores
                    st.session_state.last_master_comment = scores.get("master_comment", "")
                    st.rerun()
    
    if st.session_state.last_scores:
        s = st.session_state.last_scores["scores"]
        mc = st.session_state.last_master_comment
        st.markdown(f'<div class="master-comment"><b>👵 宗师总评：</b><br>{mc}</div>', unsafe_allow_html=True)
        # 展示评分结果（包含可视化）
        left_col, right_col = st.columns([3, 7]) 
        with left_col:
            st.subheader("📊 风味形态")
            st.pyplot(plot_flavor_shape(st.session_state.last_scores), use_container_width=True)
        with right_col:
            cols = st.columns(2)
            factors = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]
            for i, f in enumerate(factors):
                if f in s:
                    d = s[f]
                    with cols[i%3]:
                        st.markdown(f"""<div class="factor-card"><div class="score-header"><span>{f}</span><span>{d['score']}/9</span></div><div>{d['comment']}</div><div class="advice-tag">💡 {d.get('suggestion','')}</div></div>""", unsafe_allow_html=True)
        
        st.subheader("📝 得分校准与保存")
        if st.button("💾 评分准确！一键保存！"):
            nc = {"text": user_input, "scores": s, "tags": "交互-原始", "master_comment": mc, "created_at": time.strftime("%Y-%m-%d")}
            st.session_state.cases[1].append(nc)
            st.session_state.cases[0].add(embedder.encode([user_input]))
            ResourceManager.save(st.session_state.cases[0], st.session_state.cases[1], PATHS.case_index, PATHS.case_data, is_json=True)
            st.success("已保存"); st.rerun()

        st.markdown("---")
        st.subheader("🛠️ 评分有误！需要校准！")
        cal_master = st.text_area("校准总评", mc)
        cal_scores = {}
        st.write("###### 分项调整") # 加个小标题提示
        for f in factors:
            if f in s:
                # 使用 container(border=True) 形成卡片式布局，视觉更整洁
                with st.container(border=True):
                    # 标题与分数放在一起
                    st.markdown(f"**📌 {f}**") 
                    
                    cal_scores[f] = {
                        # 将分数滑块放在最上方
                        "score": st.number_input("分数", 0, 9, int(s[f]['score']), 1, key=f"s_{f}", label_visibility="collapsed"),
                        # 评语和建议直接列在下方
                        # height=68 约为两行的高度，节省空间，用户输入多时会自动滚动
                        "comment": st.text_area(f"{f} 评语", s[f]['comment'], key=f"c_{f}", height=68),
                        "suggestion": st.text_area(f"{f} 建议", s[f].get('suggestion',''), key=f"sg_{f}", height=68)
                    }
        
        if st.button("💾 保存校准评分", type="primary"):
            nc = {"text": user_input, "scores": cal_scores, "tags": "交互-校准", "master_comment": cal_master, "created_at": time.strftime("%Y-%m-%d")}
            st.session_state.cases[1].append(nc)
            st.session_state.cases[0].add(embedder.encode([user_input]))
            ResourceManager.save(st.session_state.cases[0], st.session_state.cases[1], PATHS.case_index, PATHS.case_data, is_json=True)
            ResourceManager.append_to_finetune(user_input, cal_scores, st.session_state.prompt_config['system_template'], st.session_state.prompt_config['user_template'], cal_master)
            st.success("校准已保存"); st.rerun()

# --- Tab 2: 批量评分 ---
with tab2:
    f = st.file_uploader("上传文件 (.txt/.docx)")
    c1, c2 = st.columns(2)
    r_n = c1.number_input("RAG数", 1, 20, 3, key="rb")
    c_n = c2.number_input("Case数", 1, 20, 2, key="cb")
    if f and st.button("批量处理"):
        lines = [l.strip() for l in parse_file(f).split('\n') if len(l)>10]
        res, bar = [], st.progress(0)
        for i, l in enumerate(lines):
            s, _, _ = run_scoring(l, st.session_state.kb, st.session_state.cases, st.session_state.prompt_config, embedder, client, "Qwen2.5-7B-Instruct", r_n, c_n)
            res.append({"id":i+1, "text":l, "scores":s})
            bar.progress((i+1)/len(lines))
        st.success("完成")
        st.download_button("下载Word", create_word_report(res), "report.docx")

# --- Tab 3: 模型调优 ---
with tab3:
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("📚 知识库")
        up = st.file_uploader("上传PDF", accept_multiple_files=True)
        if up and st.button("更新知识库"):
            raw = "".join([parse_file(u) for u in up])
            cks = [raw[i:i+600] for i in range(0,len(raw),500)]
            idx = faiss.IndexFlatL2(1024); idx.add(embedder.encode(cks))
            st.session_state.kb = (idx, cks)
            ResourceManager.save(idx, cks, PATHS.kb_index, PATHS.kb_chunks)
            st.success("已更新"); st.rerun()

    with c2:
        st.subheader("⚖️ 判例与微调")
        st.info(f"现有判例: {len(st.session_state.cases[1])}")
        
        if st.button("将判例转为微调数据"):
            cnt = 0
            for c in st.session_state.cases[1]:
                if ResourceManager.append_to_finetune(c["text"], c["scores"], st.session_state.prompt_config.get('system_template',''), st.session_state.prompt_config.get('user_template','')): cnt += 1
            st.success(f"导入 {cnt} 条")

        st.markdown("#### DeepSeek 微调")
        if st.button("启动微调"):
            try:
                with open(PATHS.training_file, "rb") as f: file_obj = client.files.create(file=f, purpose="fine-tune")
                # 注意：此处 Model ID 可能需根据 DeepSeek 实际 API 调整
                job = client.fine_tuning.jobs.create(training_file=file_obj.id, model="deepseek-chat", suffix="tea-v1")
                ResourceManager.save_ft_status(job.id, "queued")
                st.success(f"任务ID: {job.id}")
            except Exception as e:
                st.error(f"失败: {e}")
                if PATHS.training_file.exists():
                    with open(PATHS.training_file, "rb") as f: st.download_button("下载数据", f, "train.jsonl")

        fts = ResourceManager.load_ft_status()
        if fts:
            st.code(f"Job: {fts.get('job_id')}\nStatus: {fts.get('status')}")
            if st.button("刷新状态"):
                try:
                    job = client.fine_tuning.jobs.retrieve(fts['job_id'])
                    ResourceManager.save_ft_status(job.id, job.status, getattr(job,'fine_tuned_model',None))
                    st.rerun()
                except: pass

        with st.expander("➕ 添加精细判例"):
            with st.form("case_form"):
                f_txt = st.text_area("判例描述", height=80)
                f_tag = st.text_input("标签", "人工录入")
                st.markdown("**因子评分详情**")
                fc1, fc2 = st.columns(2)
                factors = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]
                input_scores = {}
                for i, f in enumerate(factors):
                    with (fc1 if i%2==0 else fc2):
                        val = st.number_input(f"{f}分数", 0,9,7, key=f"s_{i}")
                        cmt = st.text_input(f"{f}评语", key=f"c_{i}")
                        sug = st.text_input(f"{f}建议", key=f"a_{i}")
                        input_scores[f] = {"score": val, "comment": cmt, "suggestion": sug}
                
                if st.form_submit_button("保存"):
                    if not embedder: st.error("需 API Key")
                    else:
                        new_c = {"text": f_txt, "tags": f_tag, "scores": input_scores}
                        st.session_state.cases[1].append(new_c)
                        vec = embedder.encode([f_txt])
                        st.session_state.cases[0].add(vec)
                        ResourceManager.save(st.session_state.cases[0], st.session_state.cases[1], PATHS['case_index'], PATHS['case_data'], is_json=True)
                        
                        sys_p = st.session_state.prompt_config['system_template']
                        ResourceManager.append_to_finetune(f_txt, input_scores, sys_p, st.session_state.prompt_config['user_template'])
                        
                        st.success("已保存！")
                        time.sleep(1); st.rerun()
    
    with c3:
        st.subheader("📝 Prompt 配置")
        pc = st.session_state.prompt_config
        st.caption("系统提示词 (system_template) 默认加载自 sys_p.txt")
        st.caption("用户提示词 (user_template) 默认使用内置代码配置")
        
        sys_t = st.text_area("系统提示词", pc.get('system_template',''), height=200)
        user_t = st.text_area("用户提示词", pc.get('user_template',''), height=200)
        
        if st.button("保存 Prompt 到文件"):
            new_cfg = {"system_template": sys_t, "user_template": user_t}
            st.session_state.prompt_config = new_cfg
            with open(PATHS.prompt_config_file, 'w', encoding='utf-8') as f:
                json.dump(new_cfg, f, ensure_ascii=False, indent=2)
            st.success("Prompt 已更新并保存到 prompts.json")
