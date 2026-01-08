import streamlit as st
import os
import json
import numpy as np
import faiss
import time
import pickle
import re
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from http import HTTPStatus
import dashscope
from dashscope import TextEmbedding
from openai import OpenAI
from docx import Document

# ==========================================
# 0. 基础配置与持久化路径
# ==========================================
st.set_page_config(
    page_title="茶饮六因子AI评分器 Pro",
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
    "training_file": DATA_DIR / "deepseek_finetune.jsonl", # 微调数据
    "ft_status": DATA_DIR / "ft_status.json", # 记录微调任务ID和状态
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
    .ft-card {border: 1px solid #ddd; padding: 15px; border-radius: 8px; background-color: #f8f9fa; margin-top: 10px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心数据管理
# ==========================================

class DataManager:
    @staticmethod
    # 将FAISS和原始数据一起存盘 NOTE: FAISS index only stores vectors, raw texts/cases must be persisted separately.
    def save(index, data, idx_path, data_path, is_json=False):
        if index: faiss.write_index(index, str(idx_path))
        with open(data_path, "w" if is_json else "wb") as f:
            if is_json: json.dump(data, f, ensure_ascii=False, indent=2)
            else: pickle.dump(data, f)

    @staticmethod
    # 把“已确认判例”变成微调样本 
    def append_to_finetune(case_text, scores, system_prompt, user_template):
        try:
            user_content = user_template.format(product_desc=case_text, context_text="", case_text="")
            assistant_content = json.dumps({"master_comment": "（人工校准）", "scores": scores}, ensure_ascii=False)
            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
            with open(PATHS['training_file'], "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return True
        except: return False

    @staticmethod
    # 从磁盘恢复FAISS和数据
    def load(idx_path, data_path, is_json=False):
        if idx_path.exists() and data_path.exists():
            try:
                index = faiss.read_index(str(idx_path))
                with open(data_path, "r" if is_json else "rb") as f:
                    data = json.load(f) if is_json else pickle.load(f)
                return index, data
            except: pass
        return faiss.IndexFlatL2(1024), [] # 这里括号内的1024是由于text-embedding是1024维的，如果更换embedding模型则需要一起调整。
    
    @staticmethod
    def save_ft_status(job_id, status, fine_tuned_model=None):
        """保存微调任务状态"""
        data = {"job_id": job_id, "status": status, "timestamp": time.time()}
        if fine_tuned_model: data["fine_tuned_model"] = fine_tuned_model
        with open(PATHS['ft_status'], 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load_ft_status():
        if PATHS['ft_status'].exists():
            try: return json.load(open(PATHS['ft_status'], 'r'))
            except: pass
        return None

# 一层薄封装，把embedding API 包成一个统一的 encode() 方法
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

# 默认 Prompt
DEFAULT_PROMPT_CONFIG = {
    "system_template": """你是一名资深的茶饮产品研发与感官分析专家，精通《中国茶感官品鉴手册》等已上传的权威文献及手册。
请基于给定的产品描述、参考资料和相似历史判例，严格按照"罗马测评法2.0"进行专业评分。

====================
一、评分方法（必须严格遵守）
====================

罗马测评法2.0
三段（段位）与六因子如下（每个因子 0–9 分，整数）：

【前段：香】
1) ①优雅性：香气引发的愉悦感
2) ②辨识度：香气可被识别记忆

【中段：味】
3) ③协调性：茶汤内含物的融合度
4) ④饱和度：整体茶汤的浓厚度

【后段：韵】
5) ⑤持久性：茶汤在口腔中的余韵
6) ⑥苦涩度：苦味、收敛拉扯感

重要：你只能评这六项；不要添加任何额外维度（例如产地、工艺、树龄、品牌、价格、包装等）。

====================
二、信息来源约束（非常重要）
====================

1) 评分只能来自“用户输入的茶评文本”中明确表达或可直接对应的描述。
2) 不能使用外部常识、茶类刻板印象、产地/品种推断、或任何“脑补联想”来补齐信息。
   - 即使用户说的是“铁观音/龙井/普洱”，也不允许因为茶名而默认香气或滋味特征。
3) 若某因子在茶评中“未提及或描述极其模糊”，你仍必须给出 0–9 分，但必须：
   - 在该因子的 evidence 写“未提及/证据不足”
   - 将 confidence 标为 low
   - 分数采用“中性保守分 4”（除非用户明确表达负面/正面到足以改变分数）
4) 不要写长篇感想；不要扩写用户没有说过的细节。

====================
三、0–9 分通用标尺（用于六因子）
====================

采用“质量/体验好坏”的方向：分数越高，体验越好（包括苦涩度也是“越舒适越高分”，不是“越苦越高分”）。

通用锚点（按用户措辞强度做保守映射）：
- 9：极佳/惊艳/非常高级/几乎无可挑剔（用户表达非常强烈的肯定）
- 8：优秀/很喜欢/明显高水平
- 7：很好/清晰明显的优点
- 6：好/满意/整体不错
- 5：中等偏上/还可以
- 4：一般/中性/证据不足时的默认保守分
- 3：偏弱/有明显不足
- 2：较差/缺点突出
- 1：很差/几乎不可接受
- 0：严重缺陷/明显不适/难以下咽（用户表达极端负面）

【苦涩度特别说明（必须执行）】
- 9：几乎不苦不涩，或苦涩极轻微且很快化开，口腔无拉扯收敛不适
- 6–7：有轻微苦/涩但可接受，化得快，不影响整体舒适
- 4–5：苦/涩存在且较明显，但仍能喝，舒适度一般
- 0–3：苦涩强烈、锁喉、拉扯感重、收敛明显、难受（按用户描述强度给低分）

====================
四、因子解释口径（用于抓取证据与打分）
====================

你需要从茶评里提取与每个因子“直接相关”的语句作为证据（尽量短，最多 2 段原句/短语）。

①优雅性（香气愉悦感）关注：
- 正向：清雅、幽雅、舒服、干净、细腻、愉悦、高级、柔和不刺鼻、闻着很享受
- 负向：杂、闷、刺鼻、霉、馊、焦、烟、青臭、压迫感、不舒服

②辨识度（香气可识别与记忆点）关注：
- 正向：香型具体可指认（如兰花香/蜜香/果香/木质香等）、特征鲜明、有记忆点、一闻就知道
- 负向：香气平、糊、淡、说不清、不突出、混杂难辨

③协调性（融合度/平衡度）关注：
- 正向：协调、平衡、圆润、融合好、不突兀、前后统一、顺口
- 负向：割裂、失衡、某味突兀（酸/苦/涩/甜腻/青味等顶出来）、冲突感

④饱和度（茶汤浓厚度/充实度）关注：
- 正向：浓厚、饱满、厚度、稠滑、胶质感、物质感强、丰富
- 负向：寡淡、水薄、空、轻飘、没内容、像水

⑤持久性（余韵/回甘/余香/生津持续）关注：
- 正向：回甘持久、余香久、喉韵长、口腔留香、生津持续、咽下后还在
- 负向：散得快、余味短、回甘一闪而过、喝完没感觉

⑥苦涩度（苦味/收敛/拉扯感的舒适度）关注：
- 高分：不苦不涩、微苦即化、涩不拉扯、很顺
- 低分：苦涩重、锁喉、刮口、拉扯强、收敛明显且久

====================
五、工作流程（必须按步骤执行）
====================

Step 1：通读用户茶评，仅提取与六因子相关的句子/短语（不要扩写）。
Step 2：对每个因子：
- 找证据（evidence）
- 给 0–9 整数分（score）
- 写 2–3 句详细的解释（reason），解释必须能被证据直接支撑
- 给出置信度：high / medium / low
Step 3：计算段位小结（可计算但不得替代六因子）：
- 前段（香）= (优雅性 + 辨识度) / 2
- 中段（味）= (协调性 + 饱和度) / 2
- 后段（韵）= (持久性 + 苦涩度) / 2
并输出 overall（总分 sum=6项之和，avg=平均分）。
Step 4：列出“信息不足项”（哪些因子 evidence=未提及/证据不足））。
Step 5：列出帮助提升茶饮评分的建议（suggestion）。

====================
六、严格禁止事项
====================

- 禁止出现：根据茶类/产地/工艺“推测”香气滋味；禁止“想象”没写的体验。
- 禁止把“耐泡次数/价格/包装/品牌故事”当作任何因子的证据。
- 禁止输出非 JSON 内容。
- 禁止输出你的思考过程（只给结果 JSON）。
""",
    
    "user_template": """【待评分产品】
{product_desc}

【参考标准（知识库）】
{context_text}

【历史判例参考（案例库）】
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
}

# 内置判例，AI生成，可能后续需要根据实际情况精细评分。
# SEED_CASES = 


# ==========================================
# 2. 逻辑函数
# ==========================================

# 最核心***的评分函数；流程：用户文本 → 向量检索 → RAG + 判例拼 Prompt → 调用模型 → 解析 JSON
def run_scoring(text, kb_res, case_res, prompt_cfg, embedder, client, model_id): # 输入：茶评、知识库、案例库、prompt配置等
    vec = embedder.encode([text]) # 文本通过阿里云embedder转为向量
    ctx_txt, hits = "（无手册资料）", [] # RAG初始
    if kb_res[0].ntotal > 0: # 如果RAG非空，找到最相似的3个片段
        _, idx = kb_res[0].search(vec, 3)
        hits = [kb_res[1][i] for i in idx[0] if i < len(kb_res[1])]
        ctx_txt = "\n".join([f"- {h[:200]}..." for h in hits])
        
    case_txt, found_cases = "（无相似判例）", [] # 判例初始
    if case_res[0].ntotal > 0: # 如果判例库非空，找到最相似的2个片段
        _, idx = case_res[0].search(vec, 2)
        for i in idx[0]:
            if i < len(case_res[1]) and i >= 0:
                c = case_res[1][i]
                found_cases.append(c)
                sc = c.get('scores', {})
                u_sc = sc.get('优雅性',{}).get('score', 0) if isinstance(sc,dict) and '优雅性' in sc else 0
                k_sc = sc.get('苦涩度',{}).get('score', 0) if isinstance(sc,dict) and '苦涩度' in sc else 0
                # 挑了两个因子教模型相似的文本大致落在哪个区间
                case_txt += f"\n参考案例: {c['text'][:30]}... -> 优雅性:{u_sc} 苦涩度:{k_sc}"

    # 系统prompt无改动，用户prompt随着茶评、知识库内容、判例库内容相应变化
    sys_p = prompt_cfg.get('system_template', DEFAULT_PROMPT_CONFIG['system_template'])
    user_p = prompt_cfg.get('user_template', DEFAULT_PROMPT_CONFIG['user_template']).format(product_desc=text, context_text=ctx_txt, case_text=case_txt)

    try:
        resp = client.chat.completions.create(
            model=model_id, # 使用用户指定的 Model ID
            messages=[{"role":"system", "content":sys_p}, {"role":"user", "content":user_p}],
            response_format={"type": "json_object"},
            temperature=0.3 # 温度设置较低，减少模型自由发挥的空间
        )
        return json.loads(resp.choices[0].message.content), hits, found_cases #返回评分 JSON、命中的手册片段、命中的判例对象
    except Exception as e: #UI 友好，不会炸页面
        st.error(f"Inference Error: {e}")
        return None, [], []

# 上传文件解析
def parse_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.txt'): return uploaded_file.read().decode("utf-8")
        if uploaded_file.name.endswith('.pdf'): return "".join([p.extract_text() for p in PdfReader(uploaded_file).pages])
        # 扫描版本的PDF将无法识别其中内容
        if uploaded_file.name.endswith('.docx'): return "\n".join([p.text for p in Document(uploaded_file).paragraphs])
    except: return ""
    return ""

# 批量评分导出（仅适用于批量评分模式）
def create_word_report(results):
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
    bio = BytesIO() # Output is returned as BytesIO for direct download.
    doc.save(bio)
    bio.seek(0)
    return bio

def bootstrap_seed_cases_if_empty(embedder):
    """
    Inject built-in SEED_CASES into case library
    ONLY when local case library is empty.
    """
    case_idx, case_data = st.session_state.cases

    # 如果已经有判例，什么都不做
    if len(case_data) > 0:
        return

    texts = [c["text"] for c in SEED_CASES]
    vecs = embedder.encode(texts)

    # 确保 index 是空的、维度正确
    if case_idx.ntotal == 0 and case_idx.d == 1024:
        case_idx.add(vecs)
    else:
        case_idx = faiss.IndexFlatL2(1024)
        case_idx.add(vecs)

    case_data.extend(SEED_CASES)

    # 更新 session_state 并落盘
    st.session_state.cases = (case_idx, case_data)
    DataManager.save(
        case_idx,
        case_data,
        PATHS["case_index"],
        PATHS["case_data"],
        is_json=True
    )

# ==========================================
# 3. 页面初始化
# ==========================================

# Session State 首次加载：只做一次“冷启动恢复”
if 'loaded' not in st.session_state:
    kb_idx, kb_data = DataManager.load(PATHS['kb_index'], PATHS['kb_chunks'])
    case_idx, case_data = DataManager.load(PATHS['case_index'], PATHS['case_data'], is_json=True)
    st.session_state.kb = (kb_idx, kb_data)
    st.session_state.cases = (case_idx, case_data)
    
    if PATHS['prompt'].exists():
        try:
            with open(PATHS['prompt'], 'r') as f: st.session_state.prompt_config = json.load(f)
        except: st.session_state.prompt_config = DEFAULT_PROMPT_CONFIG.copy()
    else:
        st.session_state.prompt_config = DEFAULT_PROMPT_CONFIG.copy()
    
    st.session_state.loaded = True
with st.sidebar:
    st.header("⚙️ 系统配置")
    st.markdown("**🔐 API 配置（默认使用环境变量）**")

    # 从环境变量 / secrets 读取
    aliyun_key = os.getenv("ALIYUN_API_KEY") or st.secrets.get("ALIYUN_API_KEY", "")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY", "")
    

    if not aliyun_key or not deepseek_key:
        st.warning("⚠️ 当前未配置 API Key，系统将无法运行")
        st.stop()
    else:
        # ✅ API Key 存在，视为“调用可用”
        st.success("✅ API 调用成功")

    st.markdown("---")
    st.markdown("**🧠 模型设定**")

    # 固定模型
    model_name = "deepseek-chat"
    st.markdown(f"**当前模型：** `{model_name}`")

    # 如存在微调模型，仅展示提示（不允许切换）
    ft_status = DataManager.load_ft_status()
    if ft_status and ft_status.get("status") == "succeeded":
        ft_model = ft_status.get("fine_tuned_model")
        st.info(f"🎉 已检测到微调模型：`{ft_model}`（当前未启用）")

    model_id = model_name   # model_id 和 model_name在此处（deepseek）是一样的 model_id kept for future extension (e.g., switching to fine-tuned model), currently fixed.

    embedder = AliyunEmbedder(aliyun_key)
    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
    bootstrap_seed_cases_if_empty(embedder)

    
    st.markdown("---")
    st.markdown("**📚 RAG 知识库管理**")
    
    # 显示当前 RAG 状态
    st.caption(f"知识库片段: {len(st.session_state.kb[1])} 条")
    st.caption(f"判例库案例: {len(st.session_state.cases[1])} 条")
    
    if st.button("📤 导出 RAG 数据"):
        # 创建压缩包
        import zipfile, shutil
        
        # 创建临时目录
        temp_dir = Path("./temp_export")
        temp_dir.mkdir(exist_ok=True)
        
        # 复制所有 RAG 文件
        for key, path in PATHS.items():
            if path.exists():
                shutil.copy2(path, temp_dir / path.name)
        
        # 创建 zip 文件
        zip_path = Path("./rag_export.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in temp_dir.iterdir():
                zipf.write(file, file.name)
        
        # 提供下载
        with open(zip_path, 'rb') as f:
            st.download_button(
                label="⬇️ 下载 RAG 数据包",
                data=f,
                file_name="tea_rag_data.zip",
                mime="application/zip"
            )
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        zip_path.unlink()
    
    if st.button("📥 导入 RAG 数据"):
        uploaded_zip = st.file_uploader("上传 RAG 数据包", type=['zip'])
        if uploaded_zip:
            with st.spinner("导入中..."):
                # 解压到临时目录
                import tempfile, zipfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = Path(tmpdir) / "uploaded.zip"
                    with open(zip_path, 'wb') as f:
                        f.write(uploaded_zip.getvalue())
                    
                    # 解压
                    with zipfile.ZipFile(zip_path, 'r') as zipf:
                        zipf.extractall(DATA_DIR)
                    
                    # 重新加载数据
                    kb_idx, kb_data = DataManager.load(PATHS['kb_index'], PATHS['kb_chunks'])
                    case_idx, case_data = DataManager.load(PATHS['case_index'], PATHS['case_data'], is_json=True)
                    st.session_state.kb = (kb_idx, kb_data)
                    st.session_state.cases = (case_idx, case_data)
                    
                    st.success("✅ RAG 数据导入成功！")
                    st.rerun()
st.markdown('<div class="main-title">🍵 茶饮六因子 AI 评分器 Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="slogan">“一片叶子落入水中，改变了水的味道...”</div>', unsafe_allow_html=True)

# ==========================================
# 4. 功能标签页
# ==========================================
tab1, tab2, tab3 = st.tabs(["💡 交互评分", "🚀 批量评分", "🛠️ 模型调优"])

# --- Tab 1: 交互评分 ---
with tab1:
    st.info("AI 将参考知识库与判例库进行评分。确认结果后将自动更新 RAG 库和后台微调数据。")
    user_input = st.text_area("输入茶评描述:", height=120)
    
    if st.button("开始评分", type="primary", use_container_width=True):
        if not user_input or not client: st.warning("请检查输入或 API Key")
        else:
            with st.spinner(f"正在使用模型 {model_id} 品鉴..."):
                scores, kb_hits, case_hits = run_scoring( # 评分json，命中知识库手册的chunks，命中的相似判例
                    user_input, st.session_state.kb, st.session_state.cases,
                    st.session_state.prompt_config, embedder, client, model_id
                )
                if scores:
                    mc = scores.get("master_comment", "暂无总评")
                    st.markdown(f'<div class="master-comment"><b>👵 宗师总评：</b><br>{mc}</div>', unsafe_allow_html=True)
                    
                    cols = st.columns(3)
                    factors = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]
                    s_dict = scores.get("scores", {})
                    
                    for i, fname in enumerate(factors):
                        if fname in s_dict:
                            data = s_dict[fname]
                            with cols[i%3]:
                                st.markdown(f"""<div class="factor-card"><div class="score-header"><span>{fname}</span><span>{data.get('score')}/9</span></div><div style="margin:5px 0; font-size:0.9em;">{data.get('comment')}</div><div class="advice-tag">💡 {data.get('suggestion','')}</div></div>""", unsafe_allow_html=True)

                    with st.expander("📥 认可此评分？可保存或修改评分结果！"):
                        # ---- 1) 提供可编辑的“人工校准区” ----
                        factors = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]
                        edited_scores = {}

                        # master_comment 也允许编辑（可选）
                        edited_master = st.text_area(
                            "✍️ 宗师总评（可选：不改则沿用模型输出）",
                            value=scores.get("master_comment", ""),
                            height=120
                        )

                        st.markdown("#### 🛠️ 六因子校准（可修改后再保存）")

                        # 用 form 避免每改一个输入就触发保存逻辑混乱
                        with st.form("adjust_scores_form"):
                            c1, c2 = st.columns(2)
                            for i, f in enumerate(factors):
                                src = s_dict.get(f, {})
                                col = c1 if i % 2 == 0 else c2
                                with col:
                                    st.markdown(f"**{f}**")

                                    # 分数（0-9）
                                    score_val = st.number_input(
                                        f"{f} 分数",
                                        min_value=0, max_value=9,
                                        value=int(src.get("score", 4)),
                                        step=1,
                                        key=f"edit_score_{f}"
                                    )

                                    # 评语/建议
                                    comment_val = st.text_input(
                                        f"{f} 评语",
                                        value=str(src.get("comment", "")),
                                        key=f"edit_comment_{f}"
                                    )
                                    suggestion_val = st.text_input(
                                        f"{f} 建议",
                                        value=str(src.get("suggestion", "")),
                                        key=f"edit_suggestion_{f}"
                                    )

                                    edited_scores[f] = {
                                        "score": int(score_val),
                                        "comment": comment_val,
                                        "suggestion": suggestion_val
                                    }

                            # ---- 2) 保存按钮：以“编辑后的结果”为准落盘 & 入训练集 ----
                            submitted = st.form_submit_button("✅ 使用校准后的评分保存（加入判例库 & 训练集）")

                        if submitted:
                            # 保存判例库用“校准后的 scores”
                            new_case = {"text": user_input, "scores": edited_scores, "tags": "交互生成-人工校准"}
                            st.session_state.cases[1].append(new_case)

                            vec = embedder.encode([user_input])
                            st.session_state.cases[0].add(vec)
                            DataManager.save(
                                st.session_state.cases[0],
                                st.session_state.cases[1],
                                PATHS['case_index'],
                                PATHS['case_data'],
                                is_json=True
                            )

                            # 训练集也使用校准后的 scores（建议把 master_comment 也写入训练集）
                            sys_p = st.session_state.prompt_config['system_template']

                            # 这里沿用 append_to_finetune，但它目前 master_comment 固定“（人工校准）”
                            # 如果希望把 edited_master 写入训练集，建议升级 append_to_finetune
                            DataManager.append_to_finetune(
                                user_input,
                                edited_scores,
                                sys_p,
                                st.session_state.prompt_config['user_template']
                            )

                            st.success("✅ 已用人工校准结果存档！数据已加入判例库和微调队列。")
                            time.sleep(1)
                            st.rerun()

                        # ---- 3) 同时保留原“直接认可保存”快捷入口（可选）----
                        st.markdown("---")
                        if st.button("⚡ 直接认可模型评分并保存（不校准）"):
                            new_case = {"text": user_input, "scores": s_dict, "tags": "交互生成-未校准"}
                            st.session_state.cases[1].append(new_case)

                            vec = embedder.encode([user_input])
                            st.session_state.cases[0].add(vec)
                            DataManager.save(st.session_state.cases[0], st.session_state.cases[1], PATHS['case_index'], PATHS['case_data'], is_json=True)

                            sys_p = st.session_state.prompt_config['system_template']
                            DataManager.append_to_finetune(user_input, s_dict, sys_p, st.session_state.prompt_config['user_template'])

                            st.success("已按模型原评分存档！")
                            time.sleep(1)
                            st.rerun()


# --- Tab 2: 批量评分 ---
with tab2:
    up_file = st.file_uploader("上传文件 (支持 .txt / .docx)", type=['txt','docx'])
    if up_file and st.button("开始批量处理"):
        if not client: st.error("请配置 Key")
        else:
            txt = parse_file(up_file)
            lines = [l.strip() for l in txt.split('\n') if len(l)>10]
            results = []
            bar = st.progress(0)
            for i, line in enumerate(lines):
                s, _, _ = run_scoring(line, st.session_state.kb, st.session_state.cases, st.session_state.prompt_config, embedder, client, model_id)
                results.append({"id": i+1, "text": line, "scores": s})
                bar.progress((i+1)/len(lines))
            st.success("完成！")
            doc_io = create_word_report(results)
            st.download_button("📥 下载 Word 报告", doc_io, "茶评报告.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# --- Tab 3: 模型调优 (自动化微调流程) ---
with tab3:
    c1, c2, c3 = st.columns(3)
    
    # Column 1: RAG 知识库
    with c1:
        st.subheader("📚 RAG 知识库")
        files = st.file_uploader("上传PDF", accept_multiple_files=True, key="kb_up")
        st.info(f"💾 当前存储: {len(st.session_state.kb[1])} 片段")
        if files and st.button("更新知识库"):
            if not embedder: st.error("需 API Key")
            else:
                with st.spinner("处理并存盘..."):
                    raw = "".join([parse_file(f) for f in files])
                    chunks = [raw[i:i+600] for i in range(0,len(raw),500)]
                    vecs = embedder.encode(chunks)
                    idx = faiss.IndexFlatL2(1024)
                    idx.add(vecs)
                    st.session_state.kb = (idx, chunks)
                    DataManager.save(idx, chunks, PATHS['kb_index'], PATHS['kb_chunks'])
                    st.success("知识库已更新！"); time.sleep(1); st.rerun()

    # Column 2: 判例库 & 微调控制台
    with c2:
        st.subheader("⚖️ 判例库 & 微调")
        st.caption("你录入的判例将自动积累为微调数据")
        
        # 修复点：先定义 case_count
        case_count = len(st.session_state.cases[1])
        st.info(f"💾 当前判例: {case_count} 条")

        # === 微调控制面板 ===
        st.markdown("#### ☁️ 云端微调控制台")
        
        line_count = 0
        if PATHS['training_file'].exists():
            try: line_count = sum(1 for _ in open(PATHS['training_file'], 'r', encoding='utf-8'))
            except: pass
        
        st.write(f"可用微调数据: **{line_count} 条**")
        
        if line_count >= 10:
            if st.button("🚀 一键启动微调 (DeepSeek)"):
                if not client: st.error("请先配置 API Key")
                else:
                    try:
                        with open(PATHS['training_file'], "rb") as f:
                            file_obj = client.files.create(file=f, purpose="fine-tune")
                        job = client.fine_tuning.jobs.create(
                            training_file=file_obj.id,
                            model="deepseek-chat",
                            suffix="tea-expert"
                        )
                        DataManager.save_ft_status(job.id, "queued", fine_tuned_model=None)
                        st.success(f"微调任务已启动！Job ID: {job.id}")
                        time.sleep(1); st.rerun()
                    except Exception as e:
                        st.error(f"启动微调失败: {e}")
        else:
            st.warning("⚠️ 建议积累至少 10 条判例后进行微调。")

        ft_status = DataManager.load_ft_status()
        if ft_status:
            st.markdown(f"""
            <div class="ft-card">
                <b>🔄 最近任务状态</b><br>
                Job ID: <code>{ft_status.get('job_id', 'N/A')}</code><br>
                状态: <b>{ft_status.get('status', 'N/A')}</b><br>
                模型: {ft_status.get('fine_tuned_model', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
            
            if ft_status.get('status') in ['queued', 'running']:
                if st.button("🔄 刷新状态"):
                    try:
                        job = client.fine_tuning.jobs.retrieve(ft_status['job_id'])
                        new_status = job.status
                        ft_info = {"job_id": job.id, "status": new_status}
                        if new_status == 'succeeded':
                            ft_info["fine_tuned_model"] = job.fine_tuned_model
                            st.success(f"训练完成！模型: {ft_info['fine_tuned_model']}")
                            st.balloons()
                        elif new_status == 'failed':
                            ft_info["error"] = job.error.message
                            st.error(f"训练失败: {job.error.message}")
                        
                        DataManager.save_ft_status(ft_info['job_id'], ft_info['status'], ft_info.get('fine_tuned_model'))
                        time.sleep(1); st.rerun()
                    except Exception as e:
                        st.error(f"查询状态失败: {e}")

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
                        DataManager.save(st.session_state.cases[0], st.session_state.cases[1], PATHS['case_index'], PATHS['case_data'], is_json=True)
                        
                        sys_p = st.session_state.prompt_config['system_template']
                        DataManager.append_to_finetune(f_txt, input_scores, sys_p, st.session_state.prompt_config['user_template'])
                        
                        st.success("已保存！")
                        time.sleep(1); st.rerun()

        st.write(f"现有判例预览:")
        for i, c in enumerate(st.session_state.cases[1][-5:]):
            with st.expander(f"#{case_count-i} {c.get('tags','')}"):
                st.write(c['text'][:50]+"...")
                st.json(c['scores'])

    # Column 3: Prompt
    with c3:
        st.subheader("📝 Prompt 提示词模板")
        current_sys = st.session_state.prompt_config.get('system_template', '')
        current_user = st.session_state.prompt_config.get('user_template', '')
        
        if "{case_text}" not in current_user: st.warning("用户输入模板 缺少 {case_text}")
        
        sys_t = st.text_area("系统提示词模板", current_sys, height=200)
        user_t = st.text_area("用户输入提示词模板", current_user, height=200, disabled=True)
        
        if st.button("💾 保存 Prompt 提示词"):
            new_cfg = {"system_template": sys_t, "user_template": user_t}
            st.session_state.prompt_config = new_cfg
            with open(PATHS['prompt'], 'w') as f: json.dump(new_cfg, f, ensure_ascii=False)

            st.success("Prompt 已保存！"); time.sleep(1); st.rerun()








