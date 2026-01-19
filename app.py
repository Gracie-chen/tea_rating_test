import streamlit as st
import os
import json
import requests
import numpy as np
import faiss
import time
import pickle
from github import Github, GithubException, Auth  # 新增 Auth
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
import base64

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
    .case-card {border: 1px solid #e0e0e0; padding: 12px; border-radius: 8px; margin-bottom: 10px; background-color: #fafafa;}
    </style>
""", unsafe_allow_html=True)

class PathConfig:
    """路径管理类"""
    # 外部资源文件（位于同级目录）
    SRC_SYS_PROMPT = Path("sys_p.txt")
    # 修改：将 seed_case.json 改为 tea_data/case.json
    SRC_CASES = Path("tea_data/case.json")  # GitHub上的路径

    # 运行时数据目录
    DATA_DIR = Path("./tea_data")
    RAG_DIR = Path("./tea_data/RAG")  # 新增：RAG文件存储目录
    
    def __init__(self):
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RAG_DIR.mkdir(exist_ok=True)  # 确保RAG目录存在
        # 向量库与持久化数据
        self.kb_index = self.DATA_DIR / "kb.index"
        self.kb_chunks = self.DATA_DIR / "kb_chunks.pkl"
        self.kb_files = self.DATA_DIR / "kb_files.json"  # 新增：记录RAG文件列表
        self.case_index = self.DATA_DIR / "cases.index"
        self.case_data = self.DATA_DIR / "case.json"  # 修改：与GitHub保持一致
        
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
        """读取外部JSON文件"""
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
        with open(data_path, "w" if is_json else "wb", encoding="utf-8" if is_json else None) as f:
            if is_json: json.dump(data, f, ensure_ascii=False, indent=2)
            else: pickle.dump(data, f)
    
    @staticmethod
    def load(idx_path: Path, data_path: Path, is_json: bool = False) -> Tuple[Any, List]:
        """加载 FAISS 索引和数据文件"""
        if idx_path.exists() and data_path.exists():
            try:
                index = faiss.read_index(str(idx_path))
                with open(data_path, "r" if is_json else "rb", encoding="utf-8" if is_json else None) as f:
                    data = json.load(f) if is_json else pickle.load(f)
                return index, data
            except: pass
        return faiss.IndexFlatL2(1024), []

    # ===== 微调相关方法 =====
    @staticmethod
    def overwrite_finetune(cases: List[Dict], sys_prompt: str, user_tpl: str) -> int:
        """覆盖写入微调数据集 (.jsonl) - 修改为覆盖逻辑"""
        try:
            count = 0
            with open(PATHS.training_file, "w", encoding="utf-8") as f:
                for c in cases:
                    case_text = c.get("text", "")
                    scores = c.get("scores", {})
                    master_comment = c.get("master_comment", "（人工校准）")
                    
                    user_content = user_tpl.format(product_desc=case_text, context_text="", case_text="")
                    assistant_content = json.dumps({"master_comment": master_comment, "scores": scores}, ensure_ascii=False)
                    entry = {
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content}
                        ]
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    count += 1
            return count
        except Exception as e:
            print(f"[ERROR] Finetune overwrite failed: {e}")
            return 0

    @staticmethod
    def append_to_finetune(case_text: str, scores: Dict, sys_prompt: str, user_tpl: str, master_comment: str = "（人工校准）") -> bool:
        """将单个判例追加到微调数据集"""
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

    # ===== RAG文件管理 =====
    @staticmethod
    def save_kb_files(file_list: List[str]):
        """保存知识库文件列表"""
        with open(PATHS.kb_files, "w", encoding="utf-8") as f:
            json.dump(file_list, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_kb_files() -> List[str]:
        """加载知识库文件列表"""
        if PATHS.kb_files.exists():
            try:
                with open(PATHS.kb_files, "r", encoding="utf-8") as f:
                    return json.load(f)
            except: pass
        return []

# ==========================================
# [SECTION 1.5] Github 同步工具 (增强版)
# ==========================================

class GithubSync:
    """负责将数据同步回 Github 仓库"""
    
    @staticmethod
    def _get_github_config():
        """获取GitHub配置"""
        token = st.secrets.get("GITHUB_TOKEN")
        repo_name = st.secrets.get("GITHUB_REPO")
        branch = st.secrets.get("GITHUB_BRANCH", "main")
        return token, repo_name, branch
    
    @staticmethod
    def _get_github_client():
        """获取 GitHub 客户端（使用新的认证方式）"""
        token, repo_name, branch = GithubSync._get_github_config()
        if not token or not repo_name:
            return None, None, None
        # 使用新的认证方式，避免 DeprecationWarning
        g = Github(auth=Auth.Token(token))
        return g, repo_name, branch
    
    @staticmethod
    def push_json(file_path_in_repo: str, data_dict: Dict, commit_msg: str = "Update via Streamlit") -> bool:
        """推送 JSON 数据到 Github"""
        g, repo_name, branch = GithubSync._get_github_client()
        
        if not g or not repo_name:
            st.error("❌ 未配置 Github Token 或 仓库名 (GITHUB_TOKEN / GITHUB_REPO)")
            return False

        try:
            repo = g.get_repo(repo_name)
            content_str = json.dumps(data_dict, ensure_ascii=False, indent=2)
            
            try:
                contents = repo.get_contents(file_path_in_repo, ref=branch)
                repo.update_file(
                    path=contents.path,
                    message=commit_msg,
                    content=content_str,
                    sha=contents.sha,
                    branch=branch
                )
            except GithubException as e:
                if e.status == 404:
                    repo.create_file(
                        path=file_path_in_repo,
                        message=f"Create {file_path_in_repo}",
                        content=content_str,
                        branch=branch
                    )
                else:
                    raise e
            return True

        except Exception as e:
            st.error(f"Github 同步失败: {str(e)}")
            return False

    @staticmethod
    def push_binary_file(file_path_in_repo: str, file_content: bytes, commit_msg: str = "Upload file") -> bool:
        """推送二进制文件到 Github (如PDF, DOCX等)"""
        g, repo_name, branch = GithubSync._get_github_client()
        
        if not g or not repo_name:
            st.error("❌ 未配置 Github Token 或 仓库名")
            return False

        try:
            repo = g.get_repo(repo_name)
            content_b64 = base64.b64encode(file_content).decode('utf-8')
            
            try:
                contents = repo.get_contents(file_path_in_repo, ref=branch)
                repo.update_file(
                    path=contents.path,
                    message=commit_msg,
                    content=content_b64,
                    sha=contents.sha,
                    branch=branch
                )
            except GithubException as e:
                if e.status == 404:
                    repo.create_file(
                        path=file_path_in_repo,
                        message=f"Create {file_path_in_repo}",
                        content=content_b64,
                        branch=branch
                    )
                else:
                    raise e
            return True

        except Exception as e:
            st.error(f"Github 文件上传失败: {str(e)}")
            return False

    @staticmethod
    def delete_file(file_path_in_repo: str, commit_msg: str = "Delete file") -> bool:
        """从 Github 删除文件"""
        g, repo_name, branch = GithubSync._get_github_client()
        
        if not g or not repo_name:
            return False

        try:
            repo = g.get_repo(repo_name)
            
            try:
                contents = repo.get_contents(file_path_in_repo, ref=branch)
                repo.delete_file(
                    path=contents.path,
                    message=commit_msg,
                    sha=contents.sha,
                    branch=branch
                )
                return True
            except GithubException as e:
                if e.status == 404:
                    return True  # 文件本来就不存在
                raise e

        except Exception as e:
            st.error(f"Github 删除文件失败: {str(e)}")
            return False

    @staticmethod
    def sync_rag_folder(current_files: List[str], uploaded_files: List, rag_folder: str = "tea_data/RAG") -> bool:
        """
        同步RAG文件夹到GitHub
        - current_files: 当前应该存在的文件名列表
        - uploaded_files: Streamlit上传的文件对象列表
        - rag_folder: GitHub上的RAG文件夹路径
        """
        g, repo_name, branch = GithubSync._get_github_client()
        
        if not g or not repo_name:
            st.error("❌ 未配置 Github Token 或 仓库名")
            return False

        try:
            repo = g.get_repo(repo_name)
            
            # 1. 获取GitHub上RAG文件夹中现有的文件
            existing_files = []
            try:
                contents = repo.get_contents(rag_folder, ref=branch)
                existing_files = [c.name for c in contents if c.type == "file"]
            except GithubException as e:
                if e.status != 404:
                    raise e
                # 404表示文件夹不存在，这是OK的
            
            # 2. 计算需要删除的文件（在GitHub上有但不在当前列表中）
            files_to_delete = set(existing_files) - set(current_files)
            
            # 3. 删除多余的文件
            for fname in files_to_delete:
                file_path = f"{rag_folder}/{fname}"
                GithubSync.delete_file(file_path, f"Delete RAG file: {fname}")
            
            # 4. 上传/更新当前的文件
            for uf in uploaded_files:
                file_path = f"{rag_folder}/{uf.name}"
                uf.seek(0)
                file_content = uf.read()
                GithubSync.push_binary_file(file_path, file_content, f"Update RAG file: {uf.name}")
            
            return True

        except Exception as e:
            st.error(f"RAG同步失败: {str(e)}")
            return False

    @staticmethod
    def sync_cases(cases: List[Dict], file_path: str = "tea_data/case.json") -> bool:
        """同步判例库到GitHub"""
        return GithubSync.push_json(file_path, cases, "Update case.json from App")

    @staticmethod
    def pull_rag_folder(rag_folder: str = "tea_data/RAG") -> List[Tuple[str, bytes]]:
        """
        从 GitHub 拉取 RAG 文件夹中的所有文件
        返回: [(文件名, 文件内容bytes), ...]
        
        重要修复：
        - 对于大文件(>1MB)，GitHub API 的 content.content 会是 None
        - 需要通过 Git Blob API 或 download_url 带认证头下载
        """
        token, repo_name, branch = GithubSync._get_github_config()
        
        if not token or not repo_name:
            print("[WARN] GitHub config not found, skip pulling RAG")
            return []

        try:
            g = Github(auth=Auth.Token(token))
            repo = g.get_repo(repo_name)
            
            files = []
            try:
                print(f"[INFO] Getting contents of {rag_folder}...")
                contents = repo.get_contents(rag_folder, ref=branch)
                print(f"[INFO] Found {len(contents)} items in {rag_folder}")
                
                for content in contents:
                    if content.type != "file":
                        print(f"[INFO] Skipping non-file: {content.name}")
                        continue
                    
                    print(f"[INFO] Processing file: {content.name}, size: {content.size} bytes")
                    file_content = None
                    
                    # 方法1: 小文件直接从 content.content 获取
                    if content.content is not None:
                        try:
                            file_content = base64.b64decode(content.content)
                            print(f"[INFO] ✓ Loaded {content.name} via content.content ({len(file_content)} bytes)")
                        except Exception as e:
                            print(f"[WARN] Failed to decode {content.name}: {e}")
                    
                    # 方法2: 大文件通过 Git Blob API 获取
                    if file_content is None and content.sha:
                        try:
                            print(f"[INFO] Trying git blob for {content.name}...")
                            blob = repo.get_git_blob(content.sha)
                            if blob.encoding == "base64":
                                file_content = base64.b64decode(blob.content)
                                print(f"[INFO] ✓ Loaded {content.name} via git blob ({len(file_content)} bytes)")
                        except Exception as e:
                            print(f"[WARN] Failed to get blob for {content.name}: {e}")
                    
                    # 方法3: 通过 download_url 带认证头下载
                    if file_content is None and content.download_url:
                        try:
                            print(f"[INFO] Trying download_url for {content.name}...")
                            headers = {
                                "Authorization": f"token {token}",
                                "Accept": "application/vnd.github.v3.raw"
                            }
                            response = requests.get(content.download_url, headers=headers, timeout=120)
                            if response.status_code == 200:
                                file_content = response.content
                                print(f"[INFO] ✓ Loaded {content.name} via download_url ({len(file_content)} bytes)")
                            else:
                                print(f"[WARN] Failed to download {content.name}: HTTP {response.status_code}")
                        except Exception as e:
                            print(f"[WARN] Failed to download {content.name}: {e}")
                    
                    # 方法4: 使用 Raw URL 直接下载
                    if file_content is None:
                        try:
                            print(f"[INFO] Trying raw URL for {content.name}...")
                            # 构造 raw URL
                            raw_url = f"https://raw.githubusercontent.com/{repo_name}/{branch}/{rag_folder}/{content.name}"
                            headers = {"Authorization": f"token {token}"}
                            response = requests.get(raw_url, headers=headers, timeout=120)
                            if response.status_code == 200:
                                file_content = response.content
                                print(f"[INFO] ✓ Loaded {content.name} via raw URL ({len(file_content)} bytes)")
                            else:
                                print(f"[WARN] Failed to get raw {content.name}: HTTP {response.status_code}")
                        except Exception as e:
                            print(f"[WARN] Failed to get raw {content.name}: {e}")
                    
                    if file_content:
                        files.append((content.name, file_content))
                    else:
                        print(f"[ERROR] ✗ Could not load {content.name} with any method")
                            
            except GithubException as e:
                if e.status == 404:
                    print(f"[INFO] RAG folder not found: {rag_folder}")
                    return []
                print(f"[ERROR] GithubException: {e}")
                raise e
            
            print(f"[INFO] Total loaded: {len(files)} files")
            return files

        except Exception as e:
            print(f"[ERROR] Pull RAG folder failed: {e}")
            import traceback
            traceback.print_exc()
            return []


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

def llm_normalize_user_input(raw_query: str, client: OpenAI) -> str:
    """使用 LLM 对用户输入做语义规范化 / 去噪"""
    system_prompt = (
        """
          A. 角色与目标
          你是"茶评清洗器"。你的任务是从输入文本中提取并输出只与茶评相关的信息，删除无关内容，保持原意与原有表述风格，只能删减不能修改。
          B. 什么算"相关信息"（保留）
          仅保留与以下内容有关的句子/短语：
          茶的基本信息：茶名/品类、产地、年份、工艺、等级、原料、香型等
          干茶/茶汤/叶底：外观、色泽、条索、汤色、叶底描述
          香气与滋味：香气类型、强弱、层次、回甘、生津、涩感、苦感、甜度、醇厚度、喉韵、体感等
          冲泡信息与表现：器具、投茶量、水温、时间、出汤、几泡变化、耐泡度、适饮建议
          主观评价与结论：好喝/一般/缺点/性价比
          C. 什么算"无关信息"（删除）
          删除与茶评无直接关系的内容，例如：
          与茶无关的生活日常、情绪宣泄、社交聊天、段子
          店铺/物流/客服/包装破损/发货慢（除非"包装异味影响茶"这类直接影响品饮）
          广告、价格链接、优惠券、引流话术、品牌吹水（除非是"性价比"且与品饮结论相关）
          与其它产品/话题无关的对比闲聊
          凑字数内容
          D. 输出格式
          只输出清洗后的茶评正文，不要解释、不加标题、不输出"删除了什么"
          如果输入中没有任何茶评相关信息，则输出："无相关茶评信息"
          E. 操作原则
          尽量保留原句；只做删除/少量拼接
          不要补充不存在的细节，不要推测        
          """
    )

    resp = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_query}
        ]
    )
    return resp.choices[0].message.content.strip()

def run_scoring(text: str, kb_res: Tuple, case_res: Tuple, prompt_cfg: Dict, embedder: AliyunEmbedder, client: OpenAI, model_id: str, k_num: int, c_num: int):
    """执行 RAG 检索与 LLM 评分"""
    vec = embedder.encode([text]) 
    
    ctx_txt, hits = "（无手册资料）", []
    if kb_res[0].ntotal > 0:
        _, idx = kb_res[0].search(vec, k_num)
        hits = [kb_res[1][i] for i in idx[0] if i < len(kb_res[1])]
        ctx_txt = "\n".join([f"- {h[:200]}..." for h in hits])

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

    sys_p = prompt_cfg.get('system_template', "")
    user_p = prompt_cfg.get('user_template', "").format(product_desc=text, context_text=ctx_txt, case_text=case_txt)

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

def parse_file_bytes(filename: str, content: bytes) -> str:
    """解析文件内容 (从bytes) - 用于从GitHub拉取的文件"""
    try:
        if filename.lower().endswith('.txt'):
            return content.decode('utf-8')
        elif filename.lower().endswith('.pdf'):
            reader = PdfReader(BytesIO(content))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        elif filename.lower().endswith('.docx'):
            doc = Document(BytesIO(content))
            return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"[WARN] Failed to parse {filename}: {e}")
        import traceback
        traceback.print_exc()
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
    """初始化判例库：如果内存/磁盘中为空，则从 case.json 文件读取"""
    case_idx, case_data = st.session_state.cases
    if len(case_data) > 0: return

    # 从外部 JSON 加载 (修改路径)
    seed_cases = ResourceManager.load_external_json(PATHS.SRC_CASES)
    if not seed_cases:
        # 兼容旧路径
        old_path = Path("seed_case.json")
        seed_cases = ResourceManager.load_external_json(old_path)
    
    if not seed_cases:
        st.warning("case.json 未找到或为空，判例库初始化跳过。")
        return

    texts = [c["text"] for c in seed_cases]
    vecs = embedder.encode(texts)

    if case_idx.ntotal == 0: case_idx = faiss.IndexFlatL2(1024)
    if len(vecs) > 0:
        case_idx.add(vecs)
        case_data.extend(seed_cases)
        st.session_state.cases = (case_idx, case_data)
        ResourceManager.save(case_idx, case_data, PATHS.case_index, PATHS.case_data, is_json=True)


def load_rag_from_github(aliyun_key: str) -> Tuple[bool, str]:
    """
    从 GitHub 加载 RAG 文件
    返回: (是否成功, 消息)
    """
    try:
        rag_files = GithubSync.pull_rag_folder("tea_data/RAG")
        
        if not rag_files:
            return False, "GitHub 上没有找到 RAG 文件"
        
        # 解析所有文件内容
        all_text = ""
        file_names = []
        
        for fname, fcontent in rag_files:
            file_names.append(fname)
            parsed_text = parse_file_bytes(fname, fcontent)
            if parsed_text:
                all_text += parsed_text + "\n"
                print(f"[INFO] Parsed {fname}: {len(parsed_text)} chars")
            else:
                print(f"[WARN] No text extracted from {fname}")
        
        if not all_text.strip():
            return False, "无法从 RAG 文件中提取文本"
        
        # 切片并构建索引
        chunks = [all_text[i:i+600] for i in range(0, len(all_text), 500)]
        
        if not chunks:
            return False, "切片失败"
        
        temp_embedder = AliyunEmbedder(aliyun_key)
        kb_idx = faiss.IndexFlatL2(1024)
        vecs = temp_embedder.encode(chunks)
        kb_idx.add(vecs)
        
        st.session_state.kb = (kb_idx, chunks)
        st.session_state.kb_files = file_names
        
        # 保存到本地缓存
        ResourceManager.save(kb_idx, chunks, PATHS.kb_index, PATHS.kb_chunks)
        ResourceManager.save_kb_files(file_names)
        
        return True, f"成功加载 {len(chunks)} 条知识片段，来自 {len(file_names)} 个文件"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"加载失败: {str(e)}"


# ==========================================
# [SECTION 3.5] 判例管理弹窗
# ==========================================

@st.dialog("📋 判例库管理", width="large")
def show_cases_dialog(embedder: AliyunEmbedder):
    """展示并管理所有判例的弹窗"""
    cases = st.session_state.cases[1]
    
    if not cases:
        st.info("当前判例库为空")
        return
    
    st.write(f"共 **{len(cases)}** 条判例")
    
    # 用于追踪需要删除的判例索引
    if 'cases_to_delete' not in st.session_state:
        st.session_state.cases_to_delete = set()
    
    # 用于追踪编辑状态
    if 'editing_case_idx' not in st.session_state:
        st.session_state.editing_case_idx = None
    
    for idx, case in enumerate(cases):
        with st.container(border=True):
            col1, col2, col3 = st.columns([6, 1, 1])
            
            with col1:
                # 显示判例摘要
                text_preview = case.get('text', '')[:100] + ('...' if len(case.get('text', '')) > 100 else '')
                st.markdown(f"**#{idx+1}** {text_preview}")
                
                # 显示分数摘要
                scores = case.get('scores', {})
                if scores:
                    score_str = " | ".join([f"{k}:{v.get('score', '?')}" for k, v in scores.items()])
                    st.caption(score_str)
            
            with col2:
                if st.button("✏️", key=f"edit_{idx}", help="编辑此判例"):
                    st.session_state.editing_case_idx = idx
                    st.rerun()
            
            with col3:
                if st.button("🗑️", key=f"del_{idx}", help="删除此判例"):
                    st.session_state.cases_to_delete.add(idx)
                    st.rerun()
    
    # 如果有待删除的判例
    if st.session_state.cases_to_delete:
        st.warning(f"将删除 {len(st.session_state.cases_to_delete)} 条判例")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 确认删除并同步", type="primary"):
                # 执行删除
                new_cases = [c for i, c in enumerate(cases) if i not in st.session_state.cases_to_delete]
                
                # 重建FAISS索引
                new_idx = faiss.IndexFlatL2(1024)
                if new_cases:
                    texts = [c["text"] for c in new_cases]
                    vecs = embedder.encode(texts)
                    new_idx.add(vecs)
                
                st.session_state.cases = (new_idx, new_cases)
                ResourceManager.save(new_idx, new_cases, PATHS.case_index, PATHS.case_data, is_json=True)
                
                # 同步到GitHub
                with st.spinner("同步到GitHub..."):
                    GithubSync.sync_cases(new_cases)
                
                st.session_state.cases_to_delete = set()
                st.success("删除完成！")
                time.sleep(1)
                st.rerun()
        with col2:
            if st.button("❌ 取消"):
                st.session_state.cases_to_delete = set()
                st.rerun()


@st.dialog("✏️ 编辑判例", width="large")
def edit_case_dialog(case_idx: int, embedder: AliyunEmbedder):
    """编辑单个判例的弹窗"""
    cases = st.session_state.cases[1]
    if case_idx >= len(cases):
        st.error("判例不存在")
        return
    
    case = cases[case_idx]
    factors = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]
    
    st.subheader(f"编辑判例 #{case_idx + 1}")
    
    # 编辑文本
    new_text = st.text_area("判例描述", case.get("text", ""), height=100)
    new_master = st.text_area("总评", case.get("master_comment", ""), height=60)
    new_tags = st.text_input("标签", case.get("tags", ""))
    
    # 编辑各因子分数
    st.markdown("**因子评分**")
    new_scores = {}
    cols = st.columns(3)
    
    old_scores = case.get("scores", {})
    for i, f in enumerate(factors):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{f}**")
                old_f = old_scores.get(f, {})
                new_scores[f] = {
                    "score": st.number_input(f"分数", 0, 9, int(old_f.get("score", 5)), key=f"edit_s_{f}"),
                    "comment": st.text_input(f"评语", old_f.get("comment", ""), key=f"edit_c_{f}"),
                    "suggestion": st.text_input(f"建议", old_f.get("suggestion", ""), key=f"edit_sg_{f}")
                }
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 保存修改并同步", type="primary"):
            # 更新判例
            cases[case_idx] = {
                "text": new_text,
                "scores": new_scores,
                "tags": new_tags,
                "master_comment": new_master,
                "created_at": case.get("created_at", time.strftime("%Y-%m-%d"))
            }
            
            # 重建FAISS索引（因为文本可能变了）
            new_idx = faiss.IndexFlatL2(1024)
            texts = [c["text"] for c in cases]
            vecs = embedder.encode(texts)
            new_idx.add(vecs)
            
            st.session_state.cases = (new_idx, cases)
            ResourceManager.save(new_idx, cases, PATHS.case_index, PATHS.case_data, is_json=True)
            
            # 同步到GitHub
            with st.spinner("同步到GitHub..."):
                GithubSync.sync_cases(cases)
            
            st.session_state.editing_case_idx = None
            st.success("保存成功！")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("❌ 取消"):
            st.session_state.editing_case_idx = None
            st.rerun()


# ==========================================
# [SECTION 4] 主程序逻辑
# ==========================================
# A. 初始化 Session
if 'loaded' not in st.session_state:
    print("[INFO] ========== Initializing Session ==========")
    # 1. 加载RAG与判例数据
    kb_idx, kb_data = ResourceManager.load(PATHS.kb_index, PATHS.kb_chunks)
    case_idx, case_data = ResourceManager.load(PATHS.case_index, PATHS.case_data, is_json=True)
    st.session_state.kb = (kb_idx, kb_data)
    st.session_state.cases = (case_idx, case_data)
    st.session_state.kb_files = ResourceManager.load_kb_files()
    
    print(f"[INFO] Local KB: {len(kb_data)} chunks, Cases: {len(case_data)} items")
    
    # 2. 如果本地RAG为空，尝试从GitHub拉取
    if len(kb_data) == 0:
        print("[INFO] Local KB is empty, trying to pull from GitHub...")
        temp_aliyun_key = os.getenv("ALIYUN_API_KEY") or st.secrets.get("ALIYUN_API_KEY", "")
        
        if temp_aliyun_key:
            success, msg = load_rag_from_github(temp_aliyun_key)
            print(f"[INFO] Load RAG result: {success}, {msg}")
        else:
            print("[WARN] No ALIYUN_API_KEY found, skip RAG loading")
    else:
        print(f"[INFO] Using local KB cache: {len(kb_data)} chunks")
    
    # 3. 加载 Prompt 配置
    if PATHS.prompt_config_file.exists():
        try:
            with open(PATHS.prompt_config_file, 'r', encoding='utf-8') as f:
                st.session_state.prompt_config = json.load(f)
        except: pass
        
    if 'prompt_config' not in st.session_state:
        sys_prompt_content = ResourceManager.load_external_text(PATHS.SRC_SYS_PROMPT, fallback="你是一名茶评专家...")
        st.session_state.prompt_config = {
            "system_template": sys_prompt_content,
            "user_template": DEFAULT_USER_TEMPLATE
        }
    
    st.session_state.loaded = True
    print("[INFO] ========== Session Initialized ==========")


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
    
    ft_status = ResourceManager.load_ft_status()
    if ft_status and ft_status.get("status") == "succeeded":
        st.info(f"🎉 发现微调模型：`{ft_status.get('fine_tuned_model')}`")

    embedder = AliyunEmbedder(aliyun_key)
    client = OpenAI(api_key="dummy", base_url="http://117.50.89.74:8000/v1")
    client_d = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
    
    bootstrap_seed_cases(embedder)
    
    st.markdown("---")
    kb_files = st.session_state.get('kb_files', [])
    kb_count = len(st.session_state.kb[1])
    case_count = len(st.session_state.cases[1])
    st.markdown(f"知识库: **{kb_count}** 条 | 判例库: **{case_count}** 条")
    if kb_files:
        st.caption(f"RAG文件: {', '.join(kb_files)}")
    
    # ===== 新增：手动刷新 RAG 按钮 =====
    st.markdown("---")
    if st.button("🔄 从GitHub刷新RAG", use_container_width=True):
        with st.spinner("正在从 GitHub 拉取 RAG 文件..."):
            success, msg = load_rag_from_github(aliyun_key)
            if success:
                st.success(f"✅ {msg}")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"❌ {msg}")
    
    st.caption("快速上传仅支持.zip文件格式。")
    st.caption("少量文件上传请至\"模型调优\"板块。")
    
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
st.markdown('<div class="slogan">"一片叶子落入水中，改变了水的味道..."</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["💡 交互评分", "🚀 批量评分", "🛠️ 模型调优", "📲 提示词（Prompt）配置"])

# --- Tab 1: 交互评分 ---
with tab1:
    st.info("将参考知识库与判例库进行评分。确认结果可一键更新判例库。")
    c1, c2, c3, c4, c5 = st.columns([1, 3, 1, 3, 1])
    r_num = c2.number_input("参考知识库条目数量", 1, 20, 3, key="r1")
    c_num = c4.number_input("参考判例库条目数量", 1, 20, 2, key="c1")
    
    if 'current_user_input' not in st.session_state: st.session_state.current_user_input = ""
    user_input = st.text_area("请输入茶评描述:", value=st.session_state.current_user_input, height=150, key="ui")
    st.session_state.current_user_input = user_input
    
    if 'last_scores' not in st.session_state: 
        st.session_state.last_scores = None
        st.session_state.last_master_comment = ""
    
    if st.button("开始评分", type="primary", use_container_width=True):
        if not user_input: st.warning("请输入内容")
        else:
            with st.spinner(f"正在使用 {model_id} 品鉴..."):
                user_input = llm_normalize_user_input(user_input, client_d)
                scores, kb_h, case_h = run_scoring(user_input, st.session_state.kb, st.session_state.cases, st.session_state.prompt_config, embedder, client, "Qwen2.5-7B-Instruct", r_num, c_num)
                if scores:
                    st.session_state.last_scores = scores
                    st.session_state.last_master_comment = scores.get("master_comment", "")
                    
                    # 清除校准输入框的缓存状态，确保显示新结果
                    factors = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]
                    for f in factors:
                        for key in [f"s_{f}", f"c_{f}", f"sg_{f}"]:
                            if key in st.session_state:
                                del st.session_state[key]
                    st.rerun()
    
    if st.session_state.last_scores:
        s = st.session_state.last_scores["scores"]
        mc = st.session_state.last_master_comment
        st.markdown(f'<div class="master-comment"><b>👵 宗师总评：</b><br>{mc}</div>', unsafe_allow_html=True)
        
        left_col, right_col = st.columns([35, 65]) 
        with left_col:
            st.subheader("📊 风味形态")
            st.pyplot(plot_flavor_shape(st.session_state.last_scores), use_container_width=True)
        with right_col:
            cols = st.columns(2)
            factors = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]
            for i, f in enumerate(factors):
                if f in s:
                    d = s[f]
                    with cols[i%2]:
                        st.markdown(f"""<div class="factor-card"><div class="score-header"><span>{f}</span><span>{d['score']}/9</span></div><div>{d['comment']}</div><div class="advice-tag">💡 {d.get('suggestion','')}</div></div>""", unsafe_allow_html=True)
        
        st.subheader("🛠️ 评分校准与修正")
        cal_master = st.text_area("校准总评", mc)
        cal_scores = {}
        st.write("分项调整")
        active_factors = [f for f in factors if f in s]
        grid_cols = st.columns(3) 
        for i, f in enumerate(active_factors):
            with grid_cols[i % 3]:
                with st.container(border=True):
                    t_col, s_col = st.columns([1, 1])
                    with t_col:
                        st.markdown(f"<div style='padding-top: 5px;'><b>📌 {f}</b></div>", unsafe_allow_html=True)
                    with s_col:
                        new_score = st.number_input("分数", 0, 9, int(s[f]['score']), 1, key=f"s_{f}", label_visibility="collapsed")
                    cal_scores[f] = {
                        "score": new_score,
                        "comment": st.text_area(f"评语", s[f]['comment'], key=f"c_{f}", height=80, placeholder="评语"),
                        "suggestion": st.text_area(f"建议", s[f].get('suggestion',''), key=f"sg_{f}", height=68, placeholder="建议")
                    }
        
        if st.button("💾 保存校准评分", type="primary"):
            nc = {"text": user_input, "scores": cal_scores, "tags": "交互-校准", "master_comment": cal_master, "created_at": time.strftime("%Y-%m-%d")}
            st.session_state.cases[1].append(nc)
            st.session_state.cases[0].add(embedder.encode([user_input]))
            ResourceManager.save(st.session_state.cases[0], st.session_state.cases[1], PATHS.case_index, PATHS.case_data, is_json=True)
            ResourceManager.append_to_finetune(user_input, cal_scores, st.session_state.prompt_config['system_template'], st.session_state.prompt_config['user_template'], cal_master)
            
            # 同步到GitHub
            with st.spinner("同步判例到GitHub..."):
                GithubSync.sync_cases(st.session_state.cases[1])
            
            st.success("校准已保存并同步"); st.rerun()

# --- Tab 2: 批量评分 ---
with tab2:
    f = st.file_uploader("上传文件 (.txt/.docx)")
    c1, c2, c3, c4, c5 = st.columns([1, 3, 1, 3, 1])
    r_n = c2.number_input("参考知识库条目数量", 1, 20, 3, key="rb")
    c_n = c4.number_input("参考判例库条目数量", 1, 20, 2, key="cb")
    if f and st.button("批量处理"):
        lines = [l.strip() for l in parse_file(f).split('\n') if len(l)>10]
        res, bar = [], st.progress(0)
        for i, l in enumerate(lines):
            l = llm_normalize_user_input(l, client_d)
            s, _, _ = run_scoring(l, st.session_state.kb, st.session_state.cases, st.session_state.prompt_config, embedder, client, "Qwen2.5-7B-Instruct", r_n, c_n)
            res.append({"id":i+1, "text":l, "scores":s})
            bar.progress((i+1)/len(lines))
        st.success("完成")
        st.download_button("下载Word", create_word_report(res), "report.docx")

# --- Tab 3: 模型调优 ---
with tab3:
    MANAGER_URL = "http://117.50.89.74:8001"
    c1, c2 = st.columns([6, 4])
    
    with c1:
        st.subheader("📚 知识库 (RAG)")
        st.caption("上传PDF/文档以增强模型回答的准确性。文件将同步到GitHub。")
        
        # 显示当前知识库文件
        current_kb_files = st.session_state.get('kb_files', [])
        if current_kb_files:
            st.info(f"当前知识库文件：{', '.join(current_kb_files)}")
        
        up = st.file_uploader("上传资料", accept_multiple_files=True, key="kb_uploader", 
                              type=['pdf', 'txt', 'docx'])
        
        if up and st.button("更新知识库并同步到GitHub"):
            with st.spinner("正在切片与向量化..."):
                raw = "".join([parse_file(u) for u in up])
                cks = [raw[i:i+600] for i in range(0, len(raw), 500)]
                idx = faiss.IndexFlatL2(1024)
                
                if len(cks) > 0:
                    idx.add(embedder.encode(cks))
                    st.session_state.kb = (idx, cks)
                    ResourceManager.save(idx, cks, PATHS.kb_index, PATHS.kb_chunks)
                    
                    # 保存文件列表
                    file_names = [u.name for u in up]
                    st.session_state.kb_files = file_names
                    ResourceManager.save_kb_files(file_names)
                    
                    # 同步到GitHub
                    with st.spinner("同步到GitHub..."):
                        success = GithubSync.sync_rag_folder(file_names, up, "tea_data/RAG")
                        if success:
                            st.success(f"✅ 已更新 {len(cks)} 个知识片段，并同步到GitHub")
                        else:
                            st.warning(f"⚠️ 本地更新成功，但GitHub同步失败")
                    
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("未提取到有效文本")
        
        st.divider()
        st.subheader("📕 判例库 (CASE)")
        
        # ===== 新增：展示当前判例按钮 =====
        if st.button("📋 展示当前判例", use_container_width=True):
            show_cases_dialog(embedder)
        
        # 检查是否需要打开编辑弹窗
        if st.session_state.get('editing_case_idx') is not None:
            edit_case_dialog(st.session_state.editing_case_idx, embedder)
        
        with st.expander("➕ 手动添加精细判例"):
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
                
                if st.form_submit_button("保存判例并同步"):
                    new_c = {"text": f_txt, "tags": f_tag, "scores": input_scores, "created_at": time.strftime("%Y-%m-%d")}
                    st.session_state.cases[1].append(new_c)
                    vec = embedder.encode([f_txt])
                    st.session_state.cases[0].add(vec)
                    ResourceManager.save(st.session_state.cases[0], st.session_state.cases[1], PATHS.case_index, PATHS.case_data, is_json=True)
                    
                    # 同步到GitHub
                    with st.spinner("同步到GitHub..."):
                        GithubSync.sync_cases(st.session_state.cases[1])
                    
                    st.success("已保存并同步！")
                    time.sleep(1); st.rerun()

    # --- 右侧：微调控制 ---
    with c2:
        st.subheader("🚀 模型微调 (LoRA)")
        
        server_status = "unknown"
        try:
            resp = requests.get(f"{MANAGER_URL}/status", timeout=2)
            if resp.status_code == 200:
                status_data = resp.json()
                if status_data.get("vllm_status") == "running":
                    server_status = "idle"
                else:
                    server_status = "training"
            else:
                server_status = "error"
        except:
            server_status = "offline"
        
        if server_status == "idle":
            st.success("🟢 服务器就绪 (正在进行推理服务)")
        elif server_status == "training":
            st.warning("🟠 正在微调训练中... (推理服务暂停)")
            st.markdown("⚠️ **注意：** 此时无法进行评分交互，请耐心等待训练完成。")
        elif server_status == "offline":
            st.error("🔴 无法连接到 GPU 服务器 (请联系管理员)")

        st.markdown("#### 1. 数据准备")
        
        if PATHS.training_file.exists():
            with open(PATHS.training_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            data_count = len(lines)
        else:
            data_count = 0
            
        st.info(f"当前微调数据：**{data_count} 条** | 判例库：**{len(st.session_state.cases[1])} 条**")
        
        # ===== 修改：覆盖逻辑 =====
        if st.button("🔄 将当前所有判例转为微调数据（覆盖）"):
            cnt = ResourceManager.overwrite_finetune(
                st.session_state.cases[1],
                st.session_state.prompt_config.get('system_template',''), 
                st.session_state.prompt_config.get('user_template','')
            )
            st.success(f"已覆盖写入 {cnt} 条微调数据！")
            time.sleep(1); st.rerun()

        st.markdown("#### 2. 启动训练")
        st.caption("点击下方按钮将把数据上传至 GPU 服务器并开始训练。训练期间服务将中断约 2-5 分钟。")

        btn_disabled = (server_status != "idle") or (data_count == 0)
        
        if st.button("🔥 开始微调 (Start LoRA)", type="primary", disabled=btn_disabled):
            if not PATHS.training_file.exists():
                st.error("找不到训练数据文件！")
            else:
                try:
                    with open(PATHS.training_file, "rb") as f:
                        with st.spinner("正在上传数据并启动训练任务..."):
                            files = {'file': ('tea_feedback.jsonl', f, 'application/json')}
                            r = requests.post(f"{MANAGER_URL}/upload_and_train", files=files, timeout=100)
                            
                        if r.status_code == 200:
                            st.balloons()
                            st.success(f"✅ 任务已提交！服务器响应: {r.json().get('message')}")
                            st.info("💡 你可以稍后刷新页面查看状态，训练完成后服务会自动恢复。")
                        else:
                            st.error(f"❌ 提交失败: {r.text}")
                except Exception as e:
                    st.error(f"❌ 连接错误: {e}")

# --- Tab 4: Prompt配置 ---
with tab4:
    pc = st.session_state.prompt_config
    st.markdown("系统提示词**可以修改**。完整全面的提示词会让大语言模型返回的更准确结果。")    
    sys_t = st.text_area("系统提示词", pc.get('system_template',''), height=350)
    st.markdown("用户提示词**不可修改**。其保证了发送内容与回答内容的基本结构，因此大语言模型的回答可被准确解析。")
    user_t = st.text_area("用户提示词", pc.get('user_template',''), height=250, disabled=True)
    
    if st.button("💾 保存（永久化同步）", type="primary"):
        if sys_t == pc.get('system_template'):
            st.info("内容没有变化，无需保存。")
        else:
            new_cfg = {"system_template": sys_t, "user_template": user_t}
            
            with st.spinner("正在连接 Github 仓库并写入数据..."):
                success = GithubSync.push_json(
                    file_path_in_repo="tea_data/prompts.json", 
                    data_dict=new_cfg,
                    commit_msg="Update prompts.json from App"
                )
            
            if success:
                st.success("✅ 成功写入 Github！")
                st.session_state.prompt_config = new_cfg
                with open(PATHS.prompt_config_file, 'w', encoding='utf-8') as f:
                    json.dump(new_cfg, f, ensure_ascii=False, indent=2)
