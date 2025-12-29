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

deepseek_key = os.environ.get("DEEPSEEK_KEY")
aliyun_key = os.environ.get("ALIYUN_KEY")
if not (deepseek_key AND aliyun_key):
    raise RuntimeError("OPENAI_API_KEY is not set")

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
    def save(index, data, idx_path, data_path, is_json=False):
        if index: faiss.write_index(index, str(idx_path))
        with open(data_path, "w" if is_json else "wb") as f:
            if is_json: json.dump(data, f, ensure_ascii=False, indent=2)
            else: pickle.dump(data, f)

    @staticmethod
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
    def load(idx_path, data_path, is_json=False):
        if idx_path.exists() and data_path.exists():
            try:
                index = faiss.read_index(str(idx_path))
                with open(data_path, "r" if is_json else "rb") as f:
                    data = json.load(f) if is_json else pickle.load(f)
                return index, data
            except: pass
        return faiss.IndexFlatL2(1024), []
    
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
    "system_template": """你是一名资深的茶饮产品研发与感官分析专家，精通《中国茶感官品鉴手册》。
请基于给定的产品描述、参考资料和相似历史判例，严格按照"罗马测评法2.0"进行专业评分。

【评分模型】
三段六因子：前香(优雅/辨识)、中味(协调/饱和)、后韵(持久/苦涩)。

【思维链要求】
1. 先判断苦涩度的转化（是化开回甘还是锁喉焦苦）。
2. 结合香气和口感给予客观分数（0-9分）。
3. 针对每个因子给出具体的【鉴赏建议】（高分）或【改进建议】（低分）。

{model_description}""",
    
    "user_template": """【待评分产品】
{product_desc}

【参考标准（手册）】
{context_text}

【历史判例参考（Few-Shot）】
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

# ==========================================
# 2. 逻辑函数
# ==========================================

def get_model_desc(): return "优雅性/辨识度/协调性/饱和度/持久性/苦涩度，关注各阶段感官表现。"

def run_scoring(text, kb_res, case_res, prompt_cfg, embedder, client, model_id):
    vec = embedder.encode([text])
    ctx_txt, hits = "（无手册资料）", []
    if kb_res[0].ntotal > 0:
        _, idx = kb_res[0].search(vec, 3)
        hits = [kb_res[1][i] for i in idx[0] if i < len(kb_res[1])]
        ctx_txt = "\n".join([f"- {h[:200]}..." for h in hits])
        
    case_txt, found_cases = "（无相似判例）", []
    if case_res[0].ntotal > 0:
        _, idx = case_res[0].search(vec, 2)
        for i in idx[0]:
            if i < len(case_res[1]) and i >= 0:
                c = case_res[1][i]
                found_cases.append(c)
                sc = c.get('scores', {})
                u_sc = sc.get('优雅性',{}).get('score', 0) if isinstance(sc,dict) and '优雅性' in sc else 0
                k_sc = sc.get('苦涩度',{}).get('score', 0) if isinstance(sc,dict) and '苦涩度' in sc else 0
                case_txt += f"\n参考案例: {c['text'][:30]}... -> 优雅性:{u_sc} 苦涩度:{k_sc}"

    sys_p = prompt_cfg.get('system_template', DEFAULT_PROMPT_CONFIG['system_template']).replace("{model_description}", get_model_desc())
    user_p = prompt_cfg.get('user_template', DEFAULT_PROMPT_CONFIG['user_template']).format(product_desc=text, context_text=ctx_txt, case_text=case_txt)

    try:
        resp = client.chat.completions.create(
            model=model_id, # 使用用户指定的 Model ID
            messages=[{"role":"system", "content":sys_p}, {"role":"user", "content":user_p}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        return json.loads(resp.choices[0].message.content), hits, found_cases
    except Exception as e:
        st.error(f"Inference Error: {e}")
        return None, [], []

def parse_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.txt'): return uploaded_file.read().decode("utf-8")
        if uploaded_file.name.endswith('.pdf'): return "".join([p.extract_text() for p in PdfReader(uploaded_file).pages])
        if uploaded_file.name.endswith('.docx'): return "\n".join([p.text for p in Document(uploaded_file).paragraphs])
    except: return ""
    return ""

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
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# ==========================================
# 3. 页面初始化
# ==========================================

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

    st.markdown("---")
    st.markdown("**🧠 模型设定**")
    
    # 自动获取微调后的模型 ID
    ft_status = DataManager.load_ft_status()
    default_model = "deepseek-chat" # 默认使用通用模型
    if ft_status and ft_status.get('status') == 'succeeded' and 'fine_tuned_model' in ft_status:
        default_model = ft_status['fine_tuned_model']
        st.toast(f"已自动加载微调模型: {default_model}", icon="🎉")
        
    model_id = st.text_input("Model ID", value=default_model)
    
    embedder = None
    client = None
    if aliyun_key and deepseek_key:
        embedder = AliyunEmbedder(aliyun_key)
        client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")

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
                scores, kb_hits, case_hits = run_scoring(
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
                    
                    with st.expander("📥 认可此评分？(点击保存)"):
                        if st.button("✅ 确认保存 (自动加入训练集)"):
                            new_case = {"text": user_input, "scores": s_dict, "tags": "交互生成"}
                            st.session_state.cases[1].append(new_case)
                            vec = embedder.encode([user_input])
                            st.session_state.cases[0].add(vec)
                            DataManager.save(st.session_state.cases[0], st.session_state.cases[1], PATHS['case_index'], PATHS['case_data'], is_json=True)
                            
                            sys_p = st.session_state.prompt_config['system_template'].replace("{model_description}", get_model_desc())
                            DataManager.append_to_finetune(user_input, s_dict, sys_p, st.session_state.prompt_config['user_template'])
                            
                            st.success("已存档！数据已加入 RAG 库和微调队列。")
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
                        
                        sys_p = st.session_state.prompt_config['system_template'].replace("{model_description}", get_model_desc())
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
        st.subheader("📝 Prompt 模板")
        current_sys = st.session_state.prompt_config.get('system_template', '')
        current_user = st.session_state.prompt_config.get('user_template', '')
        
        if "{case_text}" not in current_user: st.warning("User Template 缺少 {case_text}")
        
        sys_t = st.text_area("System Template", current_sys, height=200)
        user_t = st.text_area("User Template", current_user, height=200)
        
        if st.button("💾 保存 Prompt"):
            new_cfg = {"system_template": sys_t, "user_template": user_t}
            st.session_state.prompt_config = new_cfg
            with open(PATHS['prompt'], 'w') as f: json.dump(new_cfg, f, ensure_ascii=False)

            st.success("Prompt 已保存！"); time.sleep(1); st.rerun()
