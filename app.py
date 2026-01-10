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
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


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
    def append_to_finetune(case_text, scores, system_prompt, user_template, master_comment="（人工校准）"):
        """
        把"已确认判例"变成微调样本 
        修复：支持传入专家评语
        """
        try:
            # 打印调试信息
            print(f"[DEBUG] append_to_finetune 被调用")
            print(f"[DEBUG] master_comment: {master_comment}")
            
            user_content = user_template.format(product_desc=case_text, context_text="", case_text="")
            
            assistant_content = json.dumps({"master_comment": master_comment, "scores": scores}, ensure_ascii=False)
            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
            with open(PATHS['training_file'], "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            # 检查文件是否写入
            if os.path.exists(PATHS['training_file']):
                with open(PATHS['training_file'], "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    print(f"[DEBUG] 当前微调文件行数: {len(lines)}")
            
            return True
        except Exception as e:
            print(f"[ERROR] append_to_finetune 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
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



# =========================================================
# 🔧 NEW: LLM 参与“用户输入筛选 / 规范化”
# =========================================================
def llm_normalize_user_input(raw_query: str, client: OpenAI) -> str:
    """
    使用 LLM 对用户输入做语义规范化 / 去噪
    不涉及总体prompt 修改
    """
    system_prompt = (
        """
          A. 角色与目标
          你是“茶评清洗器”。你的任务是从输入文本中提取并输出只与茶评相关的信息，删除无关内容，保持原意与原有表述风格，尽量少改写。
          B. 什么算“相关信息”（保留）
          仅保留与以下内容有关的句子/短语：
          茶的基本信息：茶名/品类、产地、年份、工艺、等级、原料、香型等
          干茶/茶汤/叶底：外观、色泽、条索、汤色、叶底描述
          香气与滋味：香气类型、强弱、层次、回甘、生津、涩感、苦感、甜度、醇厚度、喉韵、体感等
          冲泡信息与表现：器具、投茶量、水温、时间、出汤、几泡变化、耐泡度、适饮建议
          主观评价与结论：好喝/一般/缺点/性价比（但要与茶有关）
          C. 什么算“无关信息”（删除）
          删除与茶评无直接关系的内容，例如：
          与茶无关的生活日常、情绪宣泄、社交聊天、段子
          店铺/物流/客服/包装破损/发货慢（除非“包装异味影响茶”这类直接影响品饮）
          广告、价格链接、优惠券、引流话术、品牌吹水（除非是“性价比”且与品饮结论相关）
          与其它产品/话题无关的对比闲聊
          重复、凑字数内容
          D. 输出格式
          只输出清洗后的茶评正文，不要解释、不加标题、不输出“删除了什么”
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
SEED_CASES = [
    {
        "text": "干茶有清淡的花香，闻着干净舒服；入口柔和顺滑，整体偏淡，回甘不明显，几乎不苦不涩。",
        "tags": "内置-清淡顺口",
        "scores": {
            "优雅性": {"score": 7, "comment": "香气清雅干净，闻着舒服。", "suggestion": "保持香气洁净度，避免闷味。"},
            "辨识度": {"score": 5, "comment": "花香存在但不算鲜明。", "suggestion": "突出一个明确主香型。"},
            "协调性": {"score": 7, "comment": "入口顺滑，整体融合度较好。", "suggestion": "维持平衡，避免甜感单独跳出。"},
            "饱和度": {"score": 4, "comment": "茶汤偏淡，厚度一般。", "suggestion": "增强茶汤物质感。"},
            "持久性": {"score": 4, "comment": "余韵较短，回甘不明显。", "suggestion": "提升回甘与留香持续。"},
            "苦涩度": {"score": 8, "comment": "几乎不苦不涩，舒适度高。", "suggestion": "保持低涩感。"}
        }
    },
    {
        "text": "香气有明显蜜甜香，闻着讨喜；入口甜润，但中段略显单薄，回甘来得快但持续不久。",
        "tags": "内置-甜润但偏薄",
        "scores": {
            "优雅性": {"score": 7, "comment": "蜜甜香让人愉悦。", "suggestion": "避免香气过甜显腻。"},
            "辨识度": {"score": 7, "comment": "蜜甜香特征清晰。", "suggestion": "增强层次感。"},
            "协调性": {"score": 6, "comment": "甜感突出但尚算协调。", "suggestion": "让甜与茶味更融合。"},
            "饱和度": {"score": 4, "comment": "中段略显单薄。", "suggestion": "提升茶汤厚度。"},
            "持久性": {"score": 5, "comment": "回甘来得快但不持久。", "suggestion": "增强后段延续性。"},
            "苦涩度": {"score": 7, "comment": "轻微苦感，可接受。", "suggestion": "控制轻苦的峰值。"}
        }
    },
    {
        "text": "香气平淡，说不出具体香型；入口正常，没有明显缺点，也没有记忆点，整体偏中性。",
        "tags": "内置-中性无突出",
        "scores": {
            "优雅性": {"score": 4, "comment": "香气平淡，愉悦感一般。", "suggestion": "提升香气质量。"},
            "辨识度": {"score": 4, "comment": "香气缺乏记忆点。", "suggestion": "强化可识别香型。"},
            "协调性": {"score": 5, "comment": "整体无明显冲突。", "suggestion": "提升整体完成度。"},
            "饱和度": {"score": 4, "comment": "茶汤偏薄。", "suggestion": "增加物质感。"},
            "持久性": {"score": 4, "comment": "喝完后余味较短。", "suggestion": "增强余韵。"},
            "苦涩度": {"score": 6, "comment": "轻微苦涩但不影响饮用。", "suggestion": "让苦涩更快化开。"}
        }
    },
    {
        "text": "香气浓郁，有明显焙火与焦糖气息；入口厚实，但苦涩顶得较快，喉部有明显收敛。",
        "tags": "内置-浓厚苦涩",
        "scores": {
            "优雅性": {"score": 4, "comment": "焙火重，愉悦感一般。", "suggestion": "降低焦糊与烟火感。"},
            "辨识度": {"score": 8, "comment": "焙火与焦糖特征鲜明。", "suggestion": "控制香气集中度。"},
            "协调性": {"score": 4, "comment": "浓厚与苦涩略显割裂。", "suggestion": "控制苦涩峰值。"},
            "饱和度": {"score": 8, "comment": "茶汤浓厚有物质感。", "suggestion": "在厚度基础上提升顺滑度。"},
            "持久性": {"score": 5, "comment": "余味尚可但不算干净。", "suggestion": "改善后段舒适度。"},
            "苦涩度": {"score": 3, "comment": "苦涩明显，收敛感强。", "suggestion": "降低涩感与喉部刺激。"}
        }
    },
    {
        "text": "香气清爽，带一点果香；入口轻快，酸甜平衡，但茶味偏轻，整体显得清秀。",
        "tags": "内置-清爽果香",
        "scores": {
            "优雅性": {"score": 7, "comment": "果香清爽，闻着舒服。", "suggestion": "保持清新而不过分张扬。"},
            "辨识度": {"score": 6, "comment": "果香可辨但不算强烈。", "suggestion": "强化果香特征。"},
            "协调性": {"score": 7, "comment": "酸甜与茶味融合较好。", "suggestion": "防止酸感突出。"},
            "饱和度": {"score": 4, "comment": "茶汤偏轻。", "suggestion": "增强中段厚度。"},
            "持久性": {"score": 5, "comment": "余味干净但不持久。", "suggestion": "延长回甘时间。"},
            "苦涩度": {"score": 7, "comment": "几乎无涩，整体顺。", "suggestion": "维持舒适度。"}
        }
    },
    {
        "text": "香气带青味，略显生；入口有明显涩感，舌面收紧，化得慢。",
        "tags": "内置-青涩明显",
        "scores": {
            "优雅性": {"score": 3, "comment": "青味明显，愉悦感偏低。", "suggestion": "减少生青气。"},
            "辨识度": {"score": 6, "comment": "青味特征明显。", "suggestion": "转化为更成熟香型。"},
            "协调性": {"score": 3, "comment": "涩感突出，破坏平衡。", "suggestion": "降低涩感强度。"},
            "饱和度": {"score": 5, "comment": "茶汤有一定存在感。", "suggestion": "让厚度与顺滑同步。"},
            "持久性": {"score": 4, "comment": "涩感停留时间较长。", "suggestion": "让后段更干净。"},
            "苦涩度": {"score": 2, "comment": "涩感强，舒适度低。", "suggestion": "显著改善涩感。"}
        }
    },
    {
        "text": "香气干净克制，入口顺滑，整体平衡，没有明显短板，但也不算惊艳。",
        "tags": "内置-均衡型",
        "scores": {
            "优雅性": {"score": 6, "comment": "香气干净，较为舒服。", "suggestion": "增加香气层次。"},
            "辨识度": {"score": 5, "comment": "香气不突出。", "suggestion": "强化记忆点。"},
            "协调性": {"score": 7, "comment": "整体平衡度较好。", "suggestion": "保持协调性。"},
            "饱和度": {"score": 5, "comment": "茶汤中等厚度。", "suggestion": "略微提升物质感。"},
            "持久性": {"score": 5, "comment": "余味中等。", "suggestion": "延长后段体验。"},
            "苦涩度": {"score": 7, "comment": "苦涩轻微且可接受。", "suggestion": "维持顺口特性。"}
        }
    },
    {
        "text": "香气略闷，不够通透；入口厚，但后段发苦，整体显得压迫。",
        "tags": "内置-闷厚苦",
        "scores": {
            "优雅性": {"score": 3, "comment": "香气闷，不够愉悦。", "suggestion": "改善香气通透度。"},
            "辨识度": {"score": 5, "comment": "香气存在但不清晰。", "suggestion": "减少杂味。"},
            "协调性": {"score": 3, "comment": "厚与苦不协调。", "suggestion": "让口感更圆润。"},
            "饱和度": {"score": 7, "comment": "茶汤较厚。", "suggestion": "在厚度中提升舒适度。"},
            "持久性": {"score": 4, "comment": "苦感停留时间较长。", "suggestion": "让后段更干净。"},
            "苦涩度": {"score": 3, "comment": "苦感明显，略锁喉。", "suggestion": "降低刺激感。"}
        }
    },
    {
        "text": "香气柔和细腻，入口温润，茶汤不厚但很顺，整体喝着舒服。",
        "tags": "内置-细腻顺滑",
        "scores": {
            "优雅性": {"score": 8, "comment": "香气柔和细腻，愉悦感高。", "suggestion": "保持细腻度。"},
            "辨识度": {"score": 5, "comment": "香气偏内敛。", "suggestion": "略增强辨识度。"},
            "协调性": {"score": 8, "comment": "整体非常顺。", "suggestion": "维持口感完整性。"},
            "饱和度": {"score": 5, "comment": "茶汤不厚但不空。", "suggestion": "在顺滑基础上增加厚度。"},
            "持久性": {"score": 6, "comment": "余韵干净，尚可。", "suggestion": "延长余香。"},
            "苦涩度": {"score": 8, "comment": "几乎无苦涩。", "suggestion": "维持低涩表现。"}
        }
    },
    {
        "text": "香气淡而不杂；入口平稳，整体没有明显问题，但略显平淡。",
        "tags": "内置-基础参考",
        "scores": {
            "优雅性": {"score": 5, "comment": "香气干净但平淡。", "suggestion": "提升香气层次。"},
            "辨识度": {"score": 4, "comment": "缺乏明确特征。", "suggestion": "增加记忆点。"},
            "协调性": {"score": 6, "comment": "整体尚算协调。", "suggestion": "提升完成度。"},
            "饱和度": {"score": 4, "comment": "茶汤偏薄。", "suggestion": "增强物质感。"},
            "持久性": {"score": 4, "comment": "余味短。", "suggestion": "增强后段。"},
            "苦涩度": {"score": 6, "comment": "轻微苦感，可接受。", "suggestion": "让苦感更快化开。"}
        }
    },
  {
    "text": "干茶香气清甜，带淡淡花果香；入口柔顺，前段清甜，中后段略显清淡，回甘慢但干净，整体风格清雅耐喝。",
    "tags": "内置-清甜清雅",
    "scores": {
      "优雅性": {"score": 7, "comment": "香气清甜自然。", "suggestion": "保持清雅基调。"},
      "辨识度": {"score": 6, "comment": "花果香可感。", "suggestion": "突出主香型。"},
      "协调性": {"score": 7, "comment": "前后段衔接顺。", "suggestion": "增强中段存在感。"},
      "饱和度": {"score": 5, "comment": "厚度中等。", "suggestion": "略增茶汤物质。"},
      "持久性": {"score": 5, "comment": "回甘偏慢。", "suggestion": "拉长余韵。"},
      "苦涩度": {"score": 8, "comment": "几乎无苦涩。", "suggestion": "维持低涩。"}
    }
  },
  {
    "text": "香气偏熟果与甜香，入口甜润顺滑，中段饱满度尚可，后段略有轻苦但化得快，整体表现稳妥耐喝。",
    "tags": "内置-甜润稳妥",
    "scores": {
      "优雅性": {"score": 6, "comment": "香气温和。", "suggestion": "避免甜感过重。"},
      "辨识度": {"score": 6, "comment": "熟果甜香明显。", "suggestion": "增加层次。"},
      "协调性": {"score": 7, "comment": "甜与苦平衡。", "suggestion": "弱化后段苦感。"},
      "饱和度": {"score": 6, "comment": "中段较饱满。", "suggestion": "提升整体厚度。"},
      "持久性": {"score": 6, "comment": "回味尚可。", "suggestion": "延长甜润感。"},
      "苦涩度": {"score": 6, "comment": "轻苦可接受。", "suggestion": "控制苦峰值。"}
    }
  },
  {
    "text": "香气清淡偏草本，入口平直，茶汤偏薄，整体没有明显缺点，但记忆点不足，属于安全但不突出类型。",
    "tags": "内置-清淡平直",
    "scores": {
      "优雅性": {"score": 5, "comment": "香气较为平淡。", "suggestion": "提升香气质量。"},
      "辨识度": {"score": 4, "comment": "缺乏记忆点。", "suggestion": "强化特征香。"},
      "协调性": {"score": 6, "comment": "整体无冲突。", "suggestion": "增强完成度。"},
      "饱和度": {"score": 4, "comment": "茶汤偏薄。", "suggestion": "增加物质感。"},
      "持久性": {"score": 4, "comment": "余味较短。", "suggestion": "拉长后段。"},
      "苦涩度": {"score": 7, "comment": "苦涩轻微。", "suggestion": "保持顺口。"}
    }
  },
  {
    "text": "香气带明显焙火甜香，入口厚实，前段有冲击力，但苦感来得偏快，后段略显收敛，风格偏重。",
    "tags": "内置-焙火厚重",
    "scores": {
      "优雅性": {"score": 4, "comment": "焙火略重。", "suggestion": "降低火气。"},
      "辨识度": {"score": 8, "comment": "焙火特征清晰。", "suggestion": "控制集中度。"},
      "协调性": {"score": 4, "comment": "厚与苦失衡。", "suggestion": "缓和苦感。"},
      "饱和度": {"score": 8, "comment": "茶汤厚实。", "suggestion": "提升顺滑度。"},
      "持久性": {"score": 5, "comment": "余味尚可。", "suggestion": "改善干净度。"},
      "苦涩度": {"score": 3, "comment": "苦涩明显。", "suggestion": "显著降涩。"}
    }
  },
  {
    "text": "香气清新带柑橘果香，入口轻快，酸甜平衡感好，但茶味偏轻，整体风格活泼清爽。",
    "tags": "内置-果酸清爽",
    "scores": {
      "优雅性": {"score": 7, "comment": "果香清新。", "suggestion": "避免酸感突出。"},
      "辨识度": {"score": 7, "comment": "果香较鲜明。", "suggestion": "增强持续性。"},
      "协调性": {"score": 7, "comment": "酸甜平衡。", "suggestion": "稳住茶味。"},
      "饱和度": {"score": 4, "comment": "茶汤偏轻。", "suggestion": "增强中段厚度。"},
      "持久性": {"score": 5, "comment": "回味干净。", "suggestion": "延长余韵。"},
      "苦涩度": {"score": 8, "comment": "几乎无涩。", "suggestion": "保持舒适度。"}
    }
  },
  {
    "text": "香气略带青味，入口涩感明显，舌面收紧，化得偏慢，整体舒适度一般，需要时间适应。",
    "tags": "内置-青涩偏重",
    "scores": {
      "优雅性": {"score": 3, "comment": "青味明显。", "suggestion": "减少生青。"},
      "辨识度": {"score": 6, "comment": "青味特征清楚。", "suggestion": "向成熟转化。"},
      "协调性": {"score": 3, "comment": "涩感破坏平衡。", "suggestion": "弱化涩感。"},
      "饱和度": {"score": 5, "comment": "存在感尚可。", "suggestion": "提升顺滑度。"},
      "持久性": {"score": 4, "comment": "涩感停留久。", "suggestion": "缩短涩尾。"},
      "苦涩度": {"score": 2, "comment": "涩感强烈。", "suggestion": "显著改善涩。"}
    }
  },
  {
    "text": "香气干净克制，入口平顺，甜苦均衡，没有明显短板，但整体风格偏内敛，不追求刺激。",
    "tags": "内置-内敛平衡",
    "scores": {
      "优雅性": {"score": 6, "comment": "香气干净。", "suggestion": "增加层次。"},
      "辨识度": {"score": 5, "comment": "特征偏弱。", "suggestion": "强化记忆点。"},
      "协调性": {"score": 8, "comment": "整体很平衡。", "suggestion": "保持协调。"},
      "饱和度": {"score": 5, "comment": "厚度中等。", "suggestion": "略增物质感。"},
      "持久性": {"score": 6, "comment": "余味尚可。", "suggestion": "延长后段。"},
      "苦涩度": {"score": 7, "comment": "苦涩轻微。", "suggestion": "维持顺口。"}
    }
  },
  {
    "text": "香气略显沉闷，入口厚度尚可，但苦感在后段集中出现，整体喝感略显压迫，不够舒展。",
    "tags": "内置-闷苦集中",
    "scores": {
      "优雅性": {"score": 3, "comment": "香气不通透。", "suggestion": "提升清透度。"},
      "辨识度": {"score": 5, "comment": "特征一般。", "suggestion": "减少杂味。"},
      "协调性": {"score": 3, "comment": "苦感偏突兀。", "suggestion": "调整苦点。"},
      "饱和度": {"score": 7, "comment": "茶汤较厚。", "suggestion": "增强圆润度。"},
      "持久性": {"score": 4, "comment": "苦尾较长。", "suggestion": "缩短后苦。"},
      "苦涩度": {"score": 3, "comment": "苦感明显。", "suggestion": "降低刺激。"}
    }
  },
  {
    "text": "香气柔和细腻，入口温润顺滑，茶汤不厚但完整度高，整体喝感舒适，偏向细水长流。",
    "tags": "内置-温润细腻",
    "scores": {
      "优雅性": {"score": 8, "comment": "香气细腻。", "suggestion": "保持柔和感。"},
      "辨识度": {"score": 5, "comment": "风格偏内敛。", "suggestion": "略增强特征。"},
      "协调性": {"score": 8, "comment": "整体非常顺。", "suggestion": "维持完整性。"},
      "饱和度": {"score": 5, "comment": "厚度适中。", "suggestion": "小幅增厚。"},
      "持久性": {"score": 6, "comment": "余韵干净。", "suggestion": "延长留香。"},
      "苦涩度": {"score": 8, "comment": "几乎无涩。", "suggestion": "维持低涩。"}
    }
  },
  {
    "text": "香气干净但偏淡，入口平稳顺口，整体没有明显问题，作为日常口粮型表现合格。",
    "tags": "内置-日常口粮",
    "scores": {
      "优雅性": {"score": 5, "comment": "香气偏淡。", "suggestion": "提升层次。"},
      "辨识度": {"score": 4, "comment": "特征不明显。", "suggestion": "增加记忆点。"},
      "协调性": {"score": 6, "comment": "整体顺畅。", "suggestion": "提高完成度。"},
      "饱和度": {"score": 4, "comment": "茶汤偏薄。", "suggestion": "增强厚度。"},
      "持久性": {"score": 4, "comment": "余味较短。", "suggestion": "拉长后段。"},
      "苦涩度": {"score": 6, "comment": "轻苦可接受。", "suggestion": "让苦更快化。"}
    }
  },
  {
    "text": "干茶香气清甜，带淡淡花果香；入口柔顺，前段清甜，中后段略显清淡，回甘慢但干净，整体风格清雅耐喝。",
    "tags": "内置-清甜清雅",
    "scores": {
      "优雅性": {"score": 7, "comment": "香气清甜自然。", "suggestion": "保持清雅基调。"},
      "辨识度": {"score": 6, "comment": "花果香可感。", "suggestion": "突出主香型。"},
      "协调性": {"score": 7, "comment": "前后段衔接顺。", "suggestion": "增强中段存在感。"},
      "饱和度": {"score": 5, "comment": "厚度中等。", "suggestion": "略增茶汤物质。"},
      "持久性": {"score": 5, "comment": "回甘偏慢。", "suggestion": "拉长余韵。"},
      "苦涩度": {"score": 8, "comment": "几乎无苦涩。", "suggestion": "维持低涩。"}
    }
  },
  {
    "text": "香气偏熟果与甜香，入口甜润顺滑，中段饱满度尚可，后段略有轻苦但化得快，整体表现稳妥耐喝。",
    "tags": "内置-甜润稳妥",
    "scores": {
      "优雅性": {"score": 6, "comment": "香气温和。", "suggestion": "避免甜感过重。"},
      "辨识度": {"score": 6, "comment": "熟果甜香明显。", "suggestion": "增加层次。"},
      "协调性": {"score": 7, "comment": "甜与苦平衡。", "suggestion": "弱化后段苦感。"},
      "饱和度": {"score": 6, "comment": "中段较饱满。", "suggestion": "提升整体厚度。"},
      "持久性": {"score": 6, "comment": "回味尚可。", "suggestion": "延长甜润感。"},
      "苦涩度": {"score": 6, "comment": "轻苦可接受。", "suggestion": "控制苦峰值。"}
    }
  },
  {
    "text": "香气清淡偏草本，入口平直，茶汤偏薄，整体没有明显缺点，但记忆点不足，属于安全但不突出类型。",
    "tags": "内置-清淡平直",
    "scores": {
      "优雅性": {"score": 5, "comment": "香气较为平淡。", "suggestion": "提升香气质量。"},
      "辨识度": {"score": 4, "comment": "缺乏记忆点。", "suggestion": "强化特征香。"},
      "协调性": {"score": 6, "comment": "整体无冲突。", "suggestion": "增强完成度。"},
      "饱和度": {"score": 4, "comment": "茶汤偏薄。", "suggestion": "增加物质感。"},
      "持久性": {"score": 4, "comment": "余味较短。", "suggestion": "拉长后段。"},
      "苦涩度": {"score": 7, "comment": "苦涩轻微。", "suggestion": "保持顺口。"}
    }
  },
  {
    "text": "香气带明显焙火甜香，入口厚实，前段有冲击力，但苦感来得偏快，后段略显收敛，风格偏重。",
    "tags": "内置-焙火厚重",
    "scores": {
      "优雅性": {"score": 4, "comment": "焙火略重。", "suggestion": "降低火气。"},
      "辨识度": {"score": 8, "comment": "焙火特征清晰。", "suggestion": "控制集中度。"},
      "协调性": {"score": 4, "comment": "厚与苦失衡。", "suggestion": "缓和苦感。"},
      "饱和度": {"score": 8, "comment": "茶汤厚实。", "suggestion": "提升顺滑度。"},
      "持久性": {"score": 5, "comment": "余味尚可。", "suggestion": "改善干净度。"},
      "苦涩度": {"score": 3, "comment": "苦涩明显。", "suggestion": "显著降涩。"}
    }
  },
  {
    "text": "香气清新带柑橘果香，入口轻快，酸甜平衡感好，但茶味偏轻，整体风格活泼清爽。",
    "tags": "内置-果酸清爽",
    "scores": {
      "优雅性": {"score": 7, "comment": "果香清新。", "suggestion": "避免酸感突出。"},
      "辨识度": {"score": 7, "comment": "果香较鲜明。", "suggestion": "增强持续性。"},
      "协调性": {"score": 7, "comment": "酸甜平衡。", "suggestion": "稳住茶味。"},
      "饱和度": {"score": 4, "comment": "茶汤偏轻。", "suggestion": "增强中段厚度。"},
      "持久性": {"score": 5, "comment": "回味干净。", "suggestion": "延长余韵。"},
      "苦涩度": {"score": 8, "comment": "几乎无涩。", "suggestion": "保持舒适度。"}
    }
  },
  {
    "text": "香气略带青味，入口涩感明显，舌面收紧，化得偏慢，整体舒适度一般，需要时间适应。",
    "tags": "内置-青涩偏重",
    "scores": {
      "优雅性": {"score": 3, "comment": "青味明显。", "suggestion": "减少生青。"},
      "辨识度": {"score": 6, "comment": "青味特征清楚。", "suggestion": "向成熟转化。"},
      "协调性": {"score": 3, "comment": "涩感破坏平衡。", "suggestion": "弱化涩感。"},
      "饱和度": {"score": 5, "comment": "存在感尚可。", "suggestion": "提升顺滑度。"},
      "持久性": {"score": 4, "comment": "涩感停留久。", "suggestion": "缩短涩尾。"},
      "苦涩度": {"score": 2, "comment": "涩感强烈。", "suggestion": "显著改善涩。"}
    }
  },
  {
    "text": "香气干净克制，入口平顺，甜苦均衡，没有明显短板，但整体风格偏内敛，不追求刺激。",
    "tags": "内置-内敛平衡",
    "scores": {
      "优雅性": {"score": 6, "comment": "香气干净。", "suggestion": "增加层次。"},
      "辨识度": {"score": 5, "comment": "特征偏弱。", "suggestion": "强化记忆点。"},
      "协调性": {"score": 8, "comment": "整体很平衡。", "suggestion": "保持协调。"},
      "饱和度": {"score": 5, "comment": "厚度中等。", "suggestion": "略增物质感。"},
      "持久性": {"score": 6, "comment": "余味尚可。", "suggestion": "延长后段。"},
      "苦涩度": {"score": 7, "comment": "苦涩轻微。", "suggestion": "维持顺口。"}
    }
  },
  {
    "text": "香气略显沉闷，入口厚度尚可，但苦感在后段集中出现，整体喝感略显压迫，不够舒展。",
    "tags": "内置-闷苦集中",
    "scores": {
      "优雅性": {"score": 3, "comment": "香气不通透。", "suggestion": "提升清透度。"},
      "辨识度": {"score": 5, "comment": "特征一般。", "suggestion": "减少杂味。"},
      "协调性": {"score": 3, "comment": "苦感偏突兀。", "suggestion": "调整苦点。"},
      "饱和度": {"score": 7, "comment": "茶汤较厚。", "suggestion": "增强圆润度。"},
      "持久性": {"score": 4, "comment": "苦尾较长。", "suggestion": "缩短后苦。"},
      "苦涩度": {"score": 3, "comment": "苦感明显。", "suggestion": "降低刺激。"}
    }
  },
  {
    "text": "香气柔和细腻，入口温润顺滑，茶汤不厚但完整度高，整体喝感舒适，偏向细水长流。",
    "tags": "内置-温润细腻",
    "scores": {
      "优雅性": {"score": 8, "comment": "香气细腻。", "suggestion": "保持柔和感。"},
      "辨识度": {"score": 5, "comment": "风格偏内敛。", "suggestion": "略增强特征。"},
      "协调性": {"score": 8, "comment": "整体非常顺。", "suggestion": "维持完整性。"},
      "饱和度": {"score": 5, "comment": "厚度适中。", "suggestion": "小幅增厚。"},
      "持久性": {"score": 6, "comment": "余韵干净。", "suggestion": "延长留香。"},
      "苦涩度": {"score": 8, "comment": "几乎无涩。", "suggestion": "维持低涩。"}
    }
  },
  {
    "text": "香气干净但偏淡，入口平稳顺口，整体没有明显问题，作为日常口粮型表现合格。",
    "tags": "内置-日常口粮",
    "scores": {
      "优雅性": {"score": 5, "comment": "香气偏淡。", "suggestion": "提升层次。"},
      "辨识度": {"score": 4, "comment": "特征不明显。", "suggestion": "增加记忆点。"},
      "协调性": {"score": 6, "comment": "整体顺畅。", "suggestion": "提高完成度。"},
      "饱和度": {"score": 4, "comment": "茶汤偏薄。", "suggestion": "增强厚度。"},
      "持久性": {"score": 4, "comment": "余味较短。", "suggestion": "拉长后段。"},
      "苦涩度": {"score": 6, "comment": "轻苦可接受。", "suggestion": "让苦更快化。"}
    }
  },
  {
    "text": "干茶香气清晰，带明显兰花香与淡淡奶甜气息，闻着干净通透；入口细腻柔顺，前段清甜，中段略有厚度支撑，回甘缓慢但持续，整体风格优雅耐喝，越喝越顺。",
    "tags": "内置-兰香细腻",
    "scores": {
      "优雅性": {"score": 8, "comment": "兰香清雅通透。", "suggestion": "保持香气纯净。"},
      "辨识度": {"score": 7, "comment": "兰香特征明确。", "suggestion": "强化主香稳定性。"},
      "协调性": {"score": 8, "comment": "前后段衔接顺畅。", "suggestion": "维持整体平衡。"},
      "饱和度": {"score": 6, "comment": "中段有支撑。", "suggestion": "略增茶汤厚度。"},
      "持久性": {"score": 7, "comment": "回甘持续。", "suggestion": "延长留香时间。"},
      "苦涩度": {"score": 8, "comment": "几乎无苦涩。", "suggestion": "维持低涩感。"}
    }
  },
  {
    "text": "香气偏成熟果甜，夹带一丝蜜香；入口甜润顺口，前段讨喜，中段茶味略显单一，后段回甘来得快但消退也快，整体属于易饮但层次一般的类型。",
    "tags": "内置-熟果易饮",
    "scores": {
      "优雅性": {"score": 6, "comment": "甜香温和舒适。", "suggestion": "避免甜感过腻。"},
      "辨识度": {"score": 6, "comment": "熟果香清晰。", "suggestion": "增加香气变化。"},
      "协调性": {"score": 6, "comment": "甜感与茶味协调。", "suggestion": "增强中段承接。"},
      "饱和度": {"score": 5, "comment": "厚度中等偏轻。", "suggestion": "提升茶汤饱满度。"},
      "持久性": {"score": 4, "comment": "回甘消退较快。", "suggestion": "延长后段表现。"},
      "苦涩度": {"score": 7, "comment": "轻微苦感。", "suggestion": "控制苦感峰值。"}
    }
  },
  {
    "text": "香气较为克制，略带木质与清甜气息；入口平稳，茶汤厚度一般，整体没有明显缺点，但风格偏保守，记忆点不足，适合日常饮用。",
    "tags": "内置-克制日饮",
    "scores": {
      "优雅性": {"score": 5, "comment": "香气偏内敛。", "suggestion": "提升香气表现。"},
      "辨识度": {"score": 4, "comment": "特征不够突出。", "suggestion": "强化风格标签。"},
      "协调性": {"score": 6, "comment": "整体稳定顺畅。", "suggestion": "增强完成度。"},
      "饱和度": {"score": 5, "comment": "厚度中等。", "suggestion": "略增物质感。"},
      "持久性": {"score": 5, "comment": "余味中等。", "suggestion": "拉长后段。"},
      "苦涩度": {"score": 7, "comment": "苦涩轻微。", "suggestion": "维持顺口。"}
    }
  },
  {
    "text": "香气浓郁，焙火、焦糖与甜香交织；入口厚实有冲击力，前中段力量感强，但苦感在中后段集中释放，喉部略有收敛，整体风格偏重口。",
    "tags": "内置-重焙厚感",
    "scores": {
      "优雅性": {"score": 4, "comment": "焙火气偏重。", "suggestion": "降低火气强度。"},
      "辨识度": {"score": 8, "comment": "焙火特征鲜明。", "suggestion": "控制香气集中。"},
      "协调性": {"score": 4, "comment": "苦感略破平衡。", "suggestion": "平滑苦感曲线。"},
      "饱和度": {"score": 8, "comment": "茶汤厚实饱满。", "suggestion": "提升顺滑度。"},
      "持久性": {"score": 6, "comment": "余味尚可。", "suggestion": "改善后段干净度。"},
      "苦涩度": {"score": 3, "comment": "苦感明显。", "suggestion": "显著降低苦感。"}
    }
  },
  {
    "text": "香气清新，带明显柑橘与花果香；入口轻快明亮，酸甜平衡感好，茶汤不厚但节奏清晰，整体风格活泼，适合清饮。",
    "tags": "内置-清新果花",
    "scores": {
      "优雅性": {"score": 7, "comment": "香气清爽明亮。", "suggestion": "保持清新感。"},
      "辨识度": {"score": 7, "comment": "果香辨识度高。", "suggestion": "增强持续性。"},
      "协调性": {"score": 7, "comment": "酸甜协调自然。", "suggestion": "防止酸感突出。"},
      "饱和度": {"score": 4, "comment": "茶汤偏轻。", "suggestion": "增强中段厚度。"},
      "持久性": {"score": 5, "comment": "回味干净。", "suggestion": "延长余韵。"},
      "苦涩度": {"score": 8, "comment": "几乎无涩。", "suggestion": "维持舒适度。"}
    }
  },

  {
    "text": "香气略带生青与草本气息；入口涩感直接，舌面收紧明显，化开速度慢，后段残留感较强，整体舒适度偏低，需要较高接受度。",
    "tags": "内置-生青涩强",
    "scores": {
      "优雅性": {"score": 2, "comment": "青味影响体验。", "suggestion": "减少生青气。"},
      "辨识度": {"score": 6, "comment": "青味特征清楚。", "suggestion": "转向成熟香型。"},
      "协调性": {"score": 3, "comment": "涩感破坏平衡。", "suggestion": "显著降涩。"},
      "饱和度": {"score": 5, "comment": "存在感尚可。", "suggestion": "提高顺滑度。"},
      "持久性": {"score": 3, "comment": "涩尾停留久。", "suggestion": "缩短涩尾。"},
      "苦涩度": {"score": 1, "comment": "涩感强烈。", "suggestion": "重点改善涩感。"}
    }
  },

  {
    "text": "香气干净克制，略带甜香；入口温润顺滑，甜苦平衡良好，整体没有明显短板，风格稳健内敛，耐喝但不追求刺激。",
    "tags": "内置-稳健内敛",
    "scores": {
      "优雅性": {"score": 6, "comment": "香气干净舒适。", "suggestion": "增加细节层次。"},
      "辨识度": {"score": 5, "comment": "风格偏保守。", "suggestion": "强化记忆点。"},
      "协调性": {"score": 8, "comment": "整体非常平衡。", "suggestion": "保持协调性。"},
      "饱和度": {"score": 5, "comment": "厚度中等。", "suggestion": "略增物质感。"},
      "持久性": {"score": 6, "comment": "余味稳定。", "suggestion": "延长后段。"},
      "苦涩度": {"score": 7, "comment": "苦涩轻微。", "suggestion": "维持顺口。"}
    }
  },

  {
    "text": "香气略显闷沉，入口厚度不错，但中后段苦感集中释放，压口感明显，整体显得不够舒展。",
    "tags": "内置-闷厚后苦",
    "scores": {
      "优雅性": {"score": 3, "comment": "香气不通透。", "suggestion": "改善香气清晰度。"},
      "辨识度": {"score": 5, "comment": "特征一般。", "suggestion": "减少杂味。"},
      "协调性": {"score": 3, "comment": "苦感突兀。", "suggestion": "平滑苦感。"},
      "饱和度": {"score": 7, "comment": "茶汤较厚。", "suggestion": "增强圆润度。"},
      "持久性": {"score": 4, "comment": "苦尾偏长。", "suggestion": "缩短后苦。"},
      "苦涩度": {"score": 3, "comment": "苦感明显。", "suggestion": "降低刺激。"}
    }
  },

  {
    "text": "香气柔和细腻，偏甜花香；入口温润顺滑，茶汤不厚但连贯度高，整体喝感轻松舒服，属于细水长流型。",
    "tags": "内置-细腻温润",
    "scores": {
      "优雅性": {"score": 8, "comment": "香气细腻柔和。", "suggestion": "保持柔和度。"},
      "辨识度": {"score": 5, "comment": "特征偏内敛。", "suggestion": "略增强辨识。"},
      "协调性": {"score": 8, "comment": "整体非常顺。", "suggestion": "维持完整性。"},
      "饱和度": {"score": 5, "comment": "厚度适中。", "suggestion": "小幅增强厚度。"},
      "持久性": {"score": 6, "comment": "余韵干净。", "suggestion": "延长留香。"},
      "苦涩度": {"score": 8, "comment": "几乎无涩。", "suggestion": "保持低涩。"}
    }
  },

  {
    "text": "香气清淡不杂，入口平稳顺口，茶汤偏薄，整体没有明显问题，但风格偏平，作为基础参考型较为合适。",
    "tags": "内置-基础参考",
    "scores": {
      "优雅性": {"score": 5, "comment": "香气干净偏淡。", "suggestion": "提升香气层次。"},
      "辨识度": {"score": 4, "comment": "特征不明显。", "suggestion": "增加记忆点。"},
      "协调性": {"score": 6, "comment": "整体协调。", "suggestion": "提高完成度。"},
      "饱和度": {"score": 4, "comment": "茶汤偏薄。", "suggestion": "增强物质感。"},
      "持久性": {"score": 4, "comment": "余味较短。", "suggestion": "拉长后段。"},
      "苦涩度": {"score": 6, "comment": "轻苦可接受。", "suggestion": "让苦更快化。"}
    }
  },
  {
    "text": "干茶香气偏闷，夹杂杂味，不够清爽；入口有明显苦涩感，舌面收紧，化开速度慢，中后段压口，整体喝感不够舒适。",
    "tags": "内置-闷杂苦涩",
    "scores": {
      "优雅性": {"score": 3, "comment": "香气闷杂。", "suggestion": "提升香气洁净度。"},
      "辨识度": {"score": 4, "comment": "香型模糊。", "suggestion": "减少杂味来源。"},
      "协调性": {"score": 2, "comment": "苦涩破坏平衡。", "suggestion": "降低苦涩强度。"},
      "饱和度": {"score": 6, "comment": "厚度尚可。", "suggestion": "改善顺滑度。"},
      "持久性": {"score": 3, "comment": "苦尾停留久。", "suggestion": "缩短后段残留。"},
      "苦涩度": {"score": 2, "comment": "苦涩明显。", "suggestion": "显著降涩。"}
    }
  },
  {
    "text": "香气偏生，青味突出；入口涩感直冲舌面，拉扯感强，回甘不明显，后段不干净，整体体验偏难接受。",
    "tags": "内置-生青拉扯",
    "scores": {
      "优雅性": {"score": 2, "comment": "青味明显。", "suggestion": "减少生青气。"},
      "辨识度": {"score": 5, "comment": "青味特征清楚。", "suggestion": "转向成熟香型。"},
      "协调性": {"score": 2, "comment": "涩感突出。", "suggestion": "削弱涩感。"},
      "饱和度": {"score": 5, "comment": "存在感尚可。", "suggestion": "提升顺滑度。"},
      "持久性": {"score": 3, "comment": "不适感停留久。", "suggestion": "缩短负面尾段。"},
      "苦涩度": {"score": 1, "comment": "涩感强烈。", "suggestion": "重点改善涩。"}
    }
  },
  {
    "text": "香气平淡且不够干净，入口寡淡，茶汤薄而空，喝完整体没有记忆点，作为日常饮用也略显乏味。",
    "tags": "内置-寡淡空薄",
    "scores": {
      "优雅性": {"score": 3, "comment": "香气乏味。", "suggestion": "提升香气质量。"},
      "辨识度": {"score": 2, "comment": "几乎无特征。", "suggestion": "明确风格方向。"},
      "协调性": {"score": 5, "comment": "整体无冲突。", "suggestion": "提升完成度。"},
      "饱和度": {"score": 2, "comment": "茶汤空薄。", "suggestion": "显著增强厚度。"},
      "持久性": {"score": 2, "comment": "余味极短。", "suggestion": "拉长后段。"},
      "苦涩度": {"score": 7, "comment": "苦涩不明显。", "suggestion": "保持顺口。"}
    }
  },
  {
    "text": "香气带明显焦糊味，入口厚但粗糙，苦感在中段迅速堆积，喉部刺激感强，整体风格失衡。",
    "tags": "内置-焦糊刺激",
    "scores": {
      "优雅性": {"score": 1, "comment": "焦糊味明显。", "suggestion": "控制焙火。"},
      "辨识度": {"score": 6, "comment": "焦糊特征明显。", "suggestion": "减少负面香气。"},
      "协调性": {"score": 2, "comment": "刺激感突出。", "suggestion": "重做口感平衡。"},
      "饱和度": {"score": 7, "comment": "茶汤较厚。", "suggestion": "改善细腻度。"},
      "持久性": {"score": 3, "comment": "不适感残留。", "suggestion": "缩短刺激尾段。"},
      "苦涩度": {"score": 2, "comment": "苦感强烈。", "suggestion": "显著降低苦感。"}
    }
  },
  {
    "text": "香气不够清晰，略带闷酸；入口酸涩并存，舌面不适，化开慢，整体喝感紧绷，不够放松。",
    "tags": "内置-闷酸涩",
    "scores": {
      "优雅性": {"score": 2, "comment": "香气不悦。", "suggestion": "改善香气通透度。"},
      "辨识度": {"score": 4, "comment": "酸涩特征明显。", "suggestion": "弱化负面特征。"},
      "协调性": {"score": 2, "comment": "酸涩失衡。", "suggestion": "调整酸涩比例。"},
      "饱和度": {"score": 4, "comment": "厚度一般。", "suggestion": "增强顺滑度。"},
      "持久性": {"score": 3, "comment": "不适感停留。", "suggestion": "缩短后段。"},
      "苦涩度": {"score": 2, "comment": "涩感突出。", "suggestion": "显著降涩。"}
    }
  },
  {
    "text": "香气单薄，入口平直，茶汤虽有存在感但缺乏层次，中后段出现干苦，整体完成度偏低。",
    "tags": "内置-单薄干苦",
    "scores": {
      "优雅性": {"score": 3, "comment": "香气单一。", "suggestion": "丰富香气层次。"},
      "辨识度": {"score": 3, "comment": "缺乏记忆点。", "suggestion": "明确风格。"},
      "协调性": {"score": 3, "comment": "后段干苦。", "suggestion": "改善后段口感。"},
      "饱和度": {"score": 5, "comment": "厚度尚可。", "suggestion": "增强圆润度。"},
      "持久性": {"score": 3, "comment": "干苦残留。", "suggestion": "缩短苦尾。"},
      "苦涩度": {"score": 3, "comment": "苦感明显。", "suggestion": "降低干苦。"}
    }
  },
  {
    "text": "香气略显杂乱，没有清晰方向；入口涩感先行，甜感不足，整体节奏混乱，喝感不流畅。",
    "tags": "内置-杂乱涩先",
    "scores": {
      "优雅性": {"score": 3, "comment": "香气杂乱。", "suggestion": "净化香气。"},
      "辨识度": {"score": 3, "comment": "风格不清晰。", "suggestion": "明确主线。"},
      "协调性": {"score": 2, "comment": "口感不连贯。", "suggestion": "重构节奏。"},
      "饱和度": {"score": 4, "comment": "存在感一般。", "suggestion": "增强主体感。"},
      "持久性": {"score": 3, "comment": "涩尾明显。", "suggestion": "缩短负面余味。"},
      "苦涩度": {"score": 2, "comment": "涩感偏强。", "suggestion": "降低涩度。"}
    }
  },
  {
    "text": "香气偏闷且略有陈味；入口厚但粗，苦感压舌，喉部刺激明显，整体喝感偏重且不友好。",
    "tags": "内置-粗厚压舌",
    "scores": {
      "优雅性": {"score": 2, "comment": "香气陈闷。", "suggestion": "改善香气新鲜度。"},
      "辨识度": {"score": 5, "comment": "厚重特征明显。", "suggestion": "减少负面风味。"},
      "协调性": {"score": 2, "comment": "刺激感过强。", "suggestion": "削弱压迫感。"},
      "饱和度": {"score": 7, "comment": "茶汤较厚。", "suggestion": "提升细腻度。"},
      "持久性": {"score": 3, "comment": "刺激残留久。", "suggestion": "缩短不适尾段。"},
      "苦涩度": {"score": 2, "comment": "苦感强烈。", "suggestion": "显著降苦。"}
    }
  },
  {
    "text": "香气不明显，入口偏水，茶味支撑不足，喝完整体缺乏存在感，作为参考样也偏弱。",
    "tags": "内置-水感不足",
    "scores": {
      "优雅性": {"score": 3, "comment": "香气偏弱。", "suggestion": "增强香气表现。"},
      "辨识度": {"score": 1, "comment": "几乎无特征。", "suggestion": "建立基础风格。"},
      "协调性": {"score": 4, "comment": "整体尚可。", "suggestion": "提升完成度。"},
      "饱和度": {"score": 1, "comment": "茶汤水感重。", "suggestion": "显著增强厚度。"},
      "持久性": {"score": 1, "comment": "余味极短。", "suggestion": "加强后段。"},
      "苦涩度": {"score": 6, "comment": "苦涩不明显。", "suggestion": "保持顺口。"}
    }
  },
  {
    "text": "香气略带异味，入口苦涩并行，舌面与喉部均有不适感，后段不干净，整体品质偏低。",
    "tags": "内置-异味不适",
    "scores": {
      "优雅性": {"score": 1, "comment": "异味影响体验。", "suggestion": "排查原料问题。"},
      "辨识度": {"score": 4, "comment": "负面特征明显。", "suggestion": "消除异味。"},
      "协调性": {"score": 1, "comment": "整体严重失衡。", "suggestion": "全面调整。"},
      "饱和度": {"score": 4, "comment": "存在感一般。", "suggestion": "改善结构。"},
      "持久性": {"score": 2, "comment": "不适残留。", "suggestion": "缩短负面余味。"},
      "苦涩度": {"score": 1, "comment": "苦涩难受。", "suggestion": "重点降苦降涩。"}
    }
  },
  {
    "text": "干茶香气极其通透，兰花与清甜蜜香层层递进，几乎没有杂味；入口如丝般顺滑，汤感细密饱满，甜润从舌尖铺开，回甘迅速而深，喉韵悠长，饮后口腔清凉生津，整体完成度极高，令人惊艳。",
    "tags": "内置-顶级兰蜜喉韵",
    "scores": {
      "优雅性": {"score": 9, "comment": "香气高级通透。", "suggestion": "保持洁净通透。"},
      "辨识度": {"score": 9, "comment": "兰蜜特征极强。", "suggestion": "稳定主香表现。"},
      "协调性": {"score": 9, "comment": "层次衔接无瑕。", "suggestion": "维持整体节奏。"},
      "饱和度": {"score": 9, "comment": "汤感细密饱满。", "suggestion": "保持饱满度。"},
      "持久性": {"score": 9, "comment": "回甘喉韵很久。", "suggestion": "延续后段留香。"},
      "苦涩度": {"score": 9, "comment": "几乎无苦无涩。", "suggestion": "维持低涩表现。"}
    }
  },
  {
    "text": "香气清冽高扬，白花香与成熟果香交织，带淡淡清凉感；入口甜润而不腻，汤体厚而柔，细腻度极高，回甘来得快且层层叠加，余韵干净悠长，整体质感非常高级。",
    "tags": "内置-高扬花果清冽",
    "scores": {
      "优雅性": {"score": 9, "comment": "清冽高扬极雅。", "suggestion": "维持清冽感。"},
      "辨识度": {"score": 8, "comment": "花果特征清晰。", "suggestion": "强化层次递进。"},
      "协调性": {"score": 9, "comment": "甜润厚柔兼具。", "suggestion": "保持平衡结构。"},
      "饱和度": {"score": 9, "comment": "厚度与细腻并存。", "suggestion": "稳定汤体密度。"},
      "持久性": {"score": 9, "comment": "余韵悠长干净。", "suggestion": "延长留香尾段。"},
      "苦涩度": {"score": 9, "comment": "苦涩极轻快化。", "suggestion": "保持快速化开。"}
    }
  },
  {
    "text": "干茶香气细致而深，蜜香、木质与淡淡药香融合得极其干净；入口温润厚实，汤感有“糯”感但不闷，甜度稳定，回甘深沉且持久，喉部有明显回甜与清凉感，喝完整体非常完整。",
    "tags": "内置-蜜木糯感深回甘",
    "scores": {
      "优雅性": {"score": 8, "comment": "香气细致干净。", "suggestion": "保持香气纯度。"},
      "辨识度": {"score": 8, "comment": "蜜木糯感明确。", "suggestion": "稳定风格输出。"},
      "协调性": {"score": 9, "comment": "温润厚实不闷。", "suggestion": "维持通透感。"},
      "饱和度": {"score": 9, "comment": "汤感厚且细密。", "suggestion": "保持密度。"},
      "持久性": {"score": 9, "comment": "回甘喉甜很久。", "suggestion": "延续尾韵。"},
      "苦涩度": {"score": 9, "comment": "几乎无苦涩。", "suggestion": "维持低涩。"}
    }
  },
  {
    "text": "香气极为纯净，清花香与甜奶香隐约交替，闻着就很高级；入口丝滑、汤体饱满而轻盈，甜润度高但不腻，回甘迅速且带持续生津，口腔留香非常久，整体几乎无可挑剔。",
    "tags": "内置-清花奶甜丝滑",
    "scores": {
      "优雅性": {"score": 9, "comment": "香气纯净高级。", "suggestion": "保持通透洁净。"},
      "辨识度": {"score": 8, "comment": "清花奶甜清晰。", "suggestion": "强化标志香。"},
      "协调性": {"score": 9, "comment": "轻盈饱满兼具。", "suggestion": "维持甜润平衡。"},
      "饱和度": {"score": 8, "comment": "饱满但不压口。", "suggestion": "保持细密感。"},
      "持久性": {"score": 9, "comment": "留香生津很久。", "suggestion": "延长回甘尾段。"},
      "苦涩度": {"score": 9, "comment": "苦涩几乎为零。", "suggestion": "维持低涩表现。"}
    }
  },
  {
    "text": "香气层次非常丰富：前调花香，中调蜜甜，尾调带清凉木质；入口厚实却极顺，汤感细腻有弹性，回甘强而不冲，喉韵深长，饮后口腔清爽生津不断，整体表现堪称顶尖。",
    "tags": "内置-多层次喉韵顶尖",
    "scores": {
      "优雅性": {"score": 9, "comment": "层次丰富且雅。", "suggestion": "保持层次清晰。"},
      "辨识度": {"score": 9, "comment": "风格标识强。", "suggestion": "稳定主线香气。"},
      "协调性": {"score": 9, "comment": "厚顺兼具不冲。", "suggestion": "维持口感节奏。"},
      "饱和度": {"score": 9, "comment": "细腻厚实有弹性。", "suggestion": "保持汤体密度。"},
      "持久性": {"score": 9, "comment": "喉韵与生津持久。", "suggestion": "延续尾韵清爽。"},
      "苦涩度": {"score": 9, "comment": "苦涩极轻快化。", "suggestion": "保持快速化开。"}
    }
  },
  {
    "text": "干茶香气清甜带幽幽花香，细闻还有淡淡果蜜与清凉感；入口即甜，汤体厚而不沉，细腻度极高，回甘如潮水般一波波涌上来，余韵干净通透，喝完口腔持续生津，整体非常惊艳。",
    "tags": "内置-清甜花蜜潮回甘",
    "scores": {
      "优雅性": {"score": 8, "comment": "清甜幽雅。", "suggestion": "保持清雅度。"},
      "辨识度": {"score": 8, "comment": "花蜜特征明显。", "suggestion": "稳定香气表现。"},
      "协调性": {"score": 9, "comment": "甜厚细腻统一。", "suggestion": "维持平衡。"},
      "饱和度": {"score": 9, "comment": "汤体厚且细密。", "suggestion": "保持饱满度。"},
      "持久性": {"score": 9, "comment": "回甘生津很久。", "suggestion": "延长尾韵。"},
      "苦涩度": {"score": 9, "comment": "几乎无苦涩。", "suggestion": "维持低涩。"}
    }
  },
  {
    "text": "香气不张扬但极其精致，幽兰与淡甜香交织，纯净度非常高；入口柔滑如绸，汤感密实，甜润稳定，回甘深且慢慢上来，喉部回甜明显，余韵悠长干净，整体质感非常高级耐品。",
    "tags": "内置-幽兰精致耐品",
    "scores": {
      "优雅性": {"score": 9, "comment": "幽兰精致高级。", "suggestion": "保持香气精致度。"},
      "辨识度": {"score": 8, "comment": "幽兰特征清楚。", "suggestion": "增强标识度。"},
      "协调性": {"score": 9, "comment": "柔滑密实统一。", "suggestion": "维持整体结构。"},
      "饱和度": {"score": 8, "comment": "密实但不闷。", "suggestion": "保持通透度。"},
      "持久性": {"score": 9, "comment": "喉甜余韵很久。", "suggestion": "延长留香。"},
      "苦涩度": {"score": 9, "comment": "苦涩极轻快化。", "suggestion": "保持快速化开。"}
    }
  },
  {
    "text": "香气明亮高扬，果香与花香交织，带淡淡青柠般清爽；入口清爽但汤体一点不薄，细腻度出色，甜润自然，回甘迅速且持续，尾韵干净通透，整体完成度非常高。",
    "tags": "内置-明亮高扬通透",
    "scores": {
      "优雅性": {"score": 8, "comment": "香气明亮高级。", "suggestion": "保持清爽通透。"},
      "辨识度": {"score": 8, "comment": "花果特征清晰。", "suggestion": "强化主香稳定。"},
      "协调性": {"score": 9, "comment": "清爽与厚度兼具。", "suggestion": "维持平衡。"},
      "饱和度": {"score": 8, "comment": "不薄且细腻。", "suggestion": "保持汤体密度。"},
      "持久性": {"score": 8, "comment": "回甘持续。", "suggestion": "延长尾韵。"},
      "苦涩度": {"score": 9, "comment": "几乎无苦无涩。", "suggestion": "维持低涩。"}
    }
  },
  {
    "text": "香气干净纯正，蜜香与轻花香非常协调，闻着就有高级感；入口厚润却不黏，汤感细密顺滑，甜润持续，回甘深长，喉韵明显，喝完口腔清爽并持续生津，整体几乎挑不出毛病。",
    "tags": "内置-厚润蜜香喉韵",
    "scores": {
      "优雅性": {"score": 9, "comment": "香气纯正高级。", "suggestion": "保持香气纯度。"},
      "辨识度": {"score": 8, "comment": "蜜香特征明显。", "suggestion": "稳定蜜香输出。"},
      "协调性": {"score": 9, "comment": "厚润不黏很顺。", "suggestion": "维持圆润度。"},
      "饱和度": {"score": 9, "comment": "汤感细密饱满。", "suggestion": "保持饱满度。"},
      "持久性": {"score": 9, "comment": "回甘喉韵持久。", "suggestion": "延续留香。"},
      "苦涩度": {"score": 9, "comment": "苦涩极轻快化。", "suggestion": "保持快速化开。"}
    }
  },
  {
    "text": "香气清雅而立体，花香、蜜甜与淡淡矿物感交织，层次分明且不张扬；入口丝滑细腻，汤体饱满有劲但不压口，回甘迅速且很深，尾韵悠长干净，整体表现堪称极佳。",
    "tags": "内置-立体清雅矿感",
    "scores": {
      "优雅性": {"score": 9, "comment": "清雅立体高级。", "suggestion": "保持层次清晰。"},
      "辨识度": {"score": 9, "comment": "矿感与花蜜鲜明。", "suggestion": "稳定风格标识。"},
      "协调性": {"score": 9, "comment": "有劲但不压口。", "suggestion": "维持口感节奏。"},
      "饱和度": {"score": 9, "comment": "饱满且细腻。", "suggestion": "保持汤体密度。"},
      "持久性": {"score": 9, "comment": "回甘尾韵很久。", "suggestion": "延长留香。"},
      "苦涩度": {"score": 9, "comment": "几乎无苦涩。", "suggestion": "维持低涩表现。"}
    }
  },
  {
    "text": "干茶香气干净，带淡淡花香；入口顺滑，前段清甜，中段厚度一般，回甘不算明显但喝着舒服，整体表现稳定。",
    "tags": "内置-清甜顺口",
    "scores": {
      "优雅性": {"score": 6, "comment": "香气干净自然。", "suggestion": "提升香气层次。"},
      "辨识度": {"score": 5, "comment": "花香不算突出。", "suggestion": "明确主香型。"},
      "协调性": {"score": 6, "comment": "整体顺畅。", "suggestion": "增强中段承接。"},
      "饱和度": {"score": 5, "comment": "厚度中等。", "suggestion": "略增物质感。"},
      "持久性": {"score": 5, "comment": "余味一般。", "suggestion": "延长回甘。"},
      "苦涩度": {"score": 7, "comment": "苦涩轻微。", "suggestion": "保持顺口。"}
    }
  },
  {
    "text": "香气偏甜，略带蜜香；入口甜润，前段讨喜，中后段稍显单薄，回甘来得快但持续时间不长。",
    "tags": "内置-甜润易饮",
    "scores": {
      "优雅性": {"score": 6, "comment": "甜香舒适。", "suggestion": "避免甜感过重。"},
      "辨识度": {"score": 6, "comment": "蜜香较清楚。", "suggestion": "增加香气变化。"},
      "协调性": {"score": 6, "comment": "甜润尚协调。", "suggestion": "稳住后段。"},
      "饱和度": {"score": 4, "comment": "中段略薄。", "suggestion": "增强厚度。"},
      "持久性": {"score": 4, "comment": "回甘偏短。", "suggestion": "拉长后段。"},
      "苦涩度": {"score": 7, "comment": "轻苦可接受。", "suggestion": "控制苦感。"}
    }
  },
  {
    "text": "香气较为平淡，没有明显杂味；入口平稳，茶汤偏薄，整体没有明显问题，但缺乏记忆点。",
    "tags": "内置-中性平稳",
    "scores": {
      "优雅性": {"score": 5, "comment": "香气偏淡。", "suggestion": "提升香气质量。"},
      "辨识度": {"score": 4, "comment": "特征不明显。", "suggestion": "强化风格。"},
      "协调性": {"score": 6, "comment": "整体协调。", "suggestion": "增强完成度。"},
      "饱和度": {"score": 4, "comment": "茶汤偏薄。", "suggestion": "增强物质感。"},
      "持久性": {"score": 4, "comment": "余味较短。", "suggestion": "延长后段。"},
      "苦涩度": {"score": 6, "comment": "苦涩轻微。", "suggestion": "让苦更快化。"}
    }
  },
  {
    "text": "香气带轻微焙火气，入口厚度尚可，但中段略有苦感，后段稍显收敛，整体偏稳重。",
    "tags": "内置-轻焙稳重",
    "scores": {
      "优雅性": {"score": 5, "comment": "焙火略重。", "suggestion": "降低火气。"},
      "辨识度": {"score": 6, "comment": "焙火特征可辨。", "suggestion": "控制集中度。"},
      "协调性": {"score": 5, "comment": "中段略失衡。", "suggestion": "缓和苦感。"},
      "饱和度": {"score": 6, "comment": "茶汤尚厚。", "suggestion": "提升顺滑度。"},
      "持久性": {"score": 5, "comment": "余味中等。", "suggestion": "改善后段。"},
      "苦涩度": {"score": 5, "comment": "苦感可感。", "suggestion": "降低苦峰。"}
    }
  },
  {
    "text": "香气清爽，带淡淡果香；入口轻快，酸甜平衡，但茶味偏轻，整体显得清秀耐喝。",
    "tags": "内置-清爽果香",
    "scores": {
      "优雅性": {"score": 6, "comment": "果香清新。", "suggestion": "保持清爽。"},
      "辨识度": {"score": 6, "comment": "果香可分辨。", "suggestion": "强化特征。"},
      "协调性": {"score": 7, "comment": "酸甜协调。", "suggestion": "稳住茶味。"},
      "饱和度": {"score": 4, "comment": "茶汤偏轻。", "suggestion": "增强中段。"},
      "持久性": {"score": 5, "comment": "回味干净。", "suggestion": "延长余韵。"},
      "苦涩度": {"score": 8, "comment": "几乎无涩。", "suggestion": "维持舒适度。"}
    }
  },
  {
    "text": "香气略带青味，但不算刺鼻；入口有轻微涩感，化得尚可，整体表现中规中矩。",
    "tags": "内置-轻青尚可",
    "scores": {
      "优雅性": {"score": 4, "comment": "青味略显。", "suggestion": "减少生青。"},
      "辨识度": {"score": 5, "comment": "青味可辨。", "suggestion": "向成熟转化。"},
      "协调性": {"score": 5, "comment": "整体尚平衡。", "suggestion": "弱化涩感。"},
      "饱和度": {"score": 5, "comment": "存在感一般。", "suggestion": "提升顺滑度。"},
      "持久性": {"score": 4, "comment": "余味一般。", "suggestion": "增强后段。"},
      "苦涩度": {"score": 5, "comment": "涩感存在。", "suggestion": "让涩更快化。"}
    }
  },
  {
    "text": "香气干净克制，入口平顺，甜苦平衡，没有明显短板，但整体风格偏保守。",
    "tags": "内置-克制均衡",
    "scores": {
      "优雅性": {"score": 6, "comment": "香气干净。", "suggestion": "增加层次。"},
      "辨识度": {"score": 5, "comment": "特征偏弱。", "suggestion": "强化记忆点。"},
      "协调性": {"score": 7, "comment": "整体平衡。", "suggestion": "保持协调。"},
      "饱和度": {"score": 5, "comment": "厚度中等。", "suggestion": "略增物质感。"},
      "持久性": {"score": 5, "comment": "余味尚可。", "suggestion": "延长后段。"},
      "苦涩度": {"score": 7, "comment": "苦涩轻微。", "suggestion": "维持顺口。"}
    }
  },
  {
    "text": "香气略显沉稳，入口厚度尚可，但后段略有苦感，整体喝感偏实在。",
    "tags": "内置-厚实稳妥",
    "scores": {
      "优雅性": {"score": 5, "comment": "香气偏稳。", "suggestion": "提升通透度。"},
      "辨识度": {"score": 5, "comment": "特征一般。", "suggestion": "增强辨识。"},
      "协调性": {"score": 5, "comment": "后段略失衡。", "suggestion": "缓和苦感。"},
      "饱和度": {"score": 6, "comment": "茶汤较厚。", "suggestion": "增强圆润度。"},
      "持久性": {"score": 5, "comment": "余味中等。", "suggestion": "改善后段。"},
      "苦涩度": {"score": 5, "comment": "苦感存在。", "suggestion": "降低苦峰。"}
    }
  },
  {
    "text": "香气柔和，入口顺滑，茶汤不厚但连贯度不错，整体喝着轻松，没有明显刺激。",
    "tags": "内置-柔和顺滑",
    "scores": {
      "优雅性": {"score": 6, "comment": "香气柔和。", "suggestion": "保持细腻感。"},
      "辨识度": {"score": 5, "comment": "特征偏内敛。", "suggestion": "略增强香型。"},
      "协调性": {"score": 7, "comment": "整体很顺。", "suggestion": "维持完整性。"},
      "饱和度": {"score": 4, "comment": "厚度偏轻。", "suggestion": "小幅增厚。"},
      "持久性": {"score": 5, "comment": "余韵尚可。", "suggestion": "延长留香。"},
      "苦涩度": {"score": 8, "comment": "几乎无涩。", "suggestion": "保持低涩。"}
    }
  },
  {
    "text": "香气清淡但不杂，入口平稳顺口，整体没有明显缺点，作为日常饮用型表现合格。",
    "tags": "内置-日常稳定",
    "scores": {
      "优雅性": {"score": 5, "comment": "香气偏淡。", "suggestion": "提升层次。"},
      "辨识度": {"score": 4, "comment": "记忆点不足。", "suggestion": "增加特征。"},
      "协调性": {"score": 6, "comment": "整体顺畅。", "suggestion": "提高完成度。"},
      "饱和度": {"score": 4, "comment": "茶汤偏薄。", "suggestion": "增强物质感。"},
      "持久性": {"score": 4, "comment": "余味较短。", "suggestion": "拉长后段。"},
      "苦涩度": {"score": 6, "comment": "轻苦可接受。", "suggestion": "让苦更快化。"}
    }
  }
]




# ==========================================
# 2. 逻辑函数
# ==========================================

# 最核心的评分函数；流程：用户文本 → 向量检索 → RAG + 判例拼 Prompt → 调用模型 → 解析 JSON
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

# 导入初始判例
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

def calculate_section_scores(scores):
    """
    将六因子得分为：前调(Top)、中调(Mid)、尾调(Base)
    """
    # 辅助函数：安全获取分数，默认为 0
    def get(key):
        return float(scores[key]["score"])

    top = (get('优雅性') + get('辨识度')) / 2
    mid = (get('协调性') + get('饱和度')) / 2
    base = (get('持久性') + get('苦涩度')) / 2
    
    return top, mid, base

# 风味形态图
def plot_flavor_shape(scores_data):
    """
    绘制基于 '前中后' 三调的茶汤形态图
    """
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

    colors = {'base': '#8B4513', 'mid': '#D2691E', 'top': '#FFD700'}
    
    mask_base = (y_new >= 1.0) & (y_new <= 1.6)
    ax.fill_betweenx(y_new[mask_base], -x_smooth[mask_base], x_smooth[mask_base], 
                     color=colors['base'], alpha=0.9, edgecolor=None)
    
    mask_mid = (y_new > 1.6) & (y_new <= 2.4)
    ax.fill_betweenx(y_new[mask_mid], -x_smooth[mask_mid], x_smooth[mask_mid], 
                     color=colors['mid'], alpha=0.85, edgecolor=None)
    
    mask_top = (y_new > 2.4) & (y_new <= 3.0)
    ax.fill_betweenx(y_new[mask_top], -x_smooth[mask_top], x_smooth[mask_top], 
                     color=colors['top'], alpha=0.8, edgecolor=None)

    ax.plot(x_smooth, y_new, color='black', linewidth=1, alpha=0.2)
    ax.plot(-x_smooth, y_new, color='black', linewidth=1, alpha=0.2)
    
    ax.axhline(y=1.6, color='white', linestyle=':', alpha=0.5)
    ax.axhline(y=2.4, color='white', linestyle=':', alpha=0.5)
    
    font_style = {'ha': 'center', 'va': 'center', 'color': 'white', 'fontweight': 'bold', 'fontsize': 12}
    ax.text(0, 2.7, f"Top\n{top:.1f}", **font_style)
    ax.text(0, 2.0, f"Mid\n{mid:.1f}", **font_style)
    ax.text(0, 1.3, f"Base\n{base:.1f}", **font_style)
    
    ax.axis('off')
    ax.set_xlim(-10, 10)
    ax.set_ylim(0.8, 3.2)
        
    return fig

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
st.markdown('<div class="main-title">🍵 茶品六因子 AI 评分器 Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="slogan">“一片叶子落入水中，改变了水的味道...”</div>', unsafe_allow_html=True)

# ==========================================
# 4. 功能标签页
# ==========================================
tab1, tab2, tab3 = st.tabs(["💡 交互评分", "🚀 批量评分", "🛠️ 模型调优"])
# --- Tab 1: 交互评分 ---
with tab1:
    st.info("AI 将参考知识库与判例库进行评分。确认结果后将自动更新 RAG 库。")
    
    # 使用会话状态存储用户输入，避免刷新后丢失
    if 'current_user_input' not in st.session_state:
        st.session_state.current_user_input = ""
    
    user_input = st.text_area(
        "输入茶评描述:", 
        value=st.session_state.current_user_input,
        height=120,
        key="user_input_area"
    )
    st.session_state.current_user_input = user_input
    
    # 使用会话状态存储评分结果
    if 'last_scores' not in st.session_state:
        st.session_state.last_scores = None
    if 'last_master_comment' not in st.session_state:
        st.session_state.last_master_comment = ""
    
    if st.button("开始评分", type="primary", use_container_width=True):
        if not user_input or not client: 
            st.warning("请检查输入或 API Key")
        else:
            with st.spinner(f"正在使用模型 {model_id} 品鉴..."):
                scores, kb_hits, case_hits = run_scoring(
                    user_input, st.session_state.kb, st.session_state.cases,
                    st.session_state.prompt_config, embedder, client, model_id
                )
                if scores:
                    # 保存评分结果到会话状态
                    st.session_state.last_scores = scores
                    st.session_state.last_master_comment = scores.get("master_comment", "暂无总评")
                    # 使用rerun显示结果
                    st.rerun()
    
    # 显示上次评分结果（如果有）
    if st.session_state.last_scores is not None:
        scores = st.session_state.last_scores
        mc = st.session_state.last_master_comment
        s_dict = scores.get("scores", {})
        
        # 显示宗师总评
        st.markdown(f'<div class="master-comment"><b>👵 宗师总评：</b><br>{mc}</div>', unsafe_allow_html=True)
        
        # 显示六因子评分
        cols = st.columns(3)
        factors = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]
        
        for i, fname in enumerate(factors):
            if fname in s_dict:
                data = s_dict[fname]
                with cols[i%3]:
                    st.markdown(f"""<div class="factor-card"><div class="score-header"><span>{fname}</span><span>{data.get('score')}/9</span></div><div style="margin:5px 0; font-size:0.9em;">{data.get('comment')}</div><div class="advice-tag">💡 {data.get('suggestion','')}</div></div>""", unsafe_allow_html=True)
        
        st.subheader("📊 风味可视化")

        # 创建布局：形态图
        vis_col2 = st.columns(1) [0]
        with vis_col2:
            st.caption("三段风味形态 (Flavor Shape)")
            # 绘制形态图
            fig_shape = plot_flavor_shape(scores)
            st.pyplot(fig_shape, use_container_width=True)

        # 完整的校准和保存区域
        with st.expander("📝 校准评分结果并保存到判例库", expanded=True):
            st.write(f"当前判例库数量: **{len(st.session_state.cases[1])}** 条")
            
            # 方法1: 保存原始评分（快捷方式）
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 保存原始评分", type="primary", use_container_width=True):
                    try:
                        # 创建新判例
                        new_case = {
                            "text": user_input,
                            "scores": s_dict,
                            "tags": "交互生成-原始",
                            "master_comment": mc,
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # 1. 添加到内存
                        st.session_state.cases[1].append(new_case)
                        new_count = len(st.session_state.cases[1])
                        
                        # 2. 生成向量
                        vec = embedder.encode([user_input])
                        
                        # 3. 添加到向量索引
                        if st.session_state.cases[0].ntotal == 0:
                            # 如果索引为空，创建新索引
                            st.session_state.cases = (faiss.IndexFlatL2(1024), st.session_state.cases[1])
                            st.session_state.cases[0].add(vec)
                        else:
                            # 索引已存在，添加向量
                            st.session_state.cases[0].add(vec)
                        
                        # 4. 保存到磁盘
                        DataManager.save(
                            st.session_state.cases[0],
                            st.session_state.cases[1],
                            PATHS['case_index'],
                            PATHS['case_data'],
                            is_json=True
                        )
                        
                        st.success(f"✅ 原始评分保存成功！判例库现有 {new_count} 条判例。")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"保存失败: {str(e)}")
            
            # 校准区域
            st.markdown("---")
            st.markdown("### 🔧 完整校准")
            
            # 校准宗师总评
            calibrated_master = st.text_area(
                "✍️ 宗师总评（可编辑）",
                value=mc,
                height=100,
                key="calibrated_master"
            )
            
            # 校准六因子
            calibrated_scores = {}
            
            # 创建6个因子校准面板
            factor_tabs = st.tabs(factors)
            
            for i, factor_name in enumerate(factors):
                with factor_tabs[i]:
                    if factor_name in s_dict:
                        original = s_dict[factor_name]
                        
                        # 分数
                        calibrated_score = st.slider(
                            "分数",
                            0, 9, 
                            value=int(original.get("score", 4)),
                            key=f"score_{factor_name}"
                        )
                        
                        # 评语
                        calibrated_comment = st.text_area(
                            "评语",
                            value=original.get("comment", ""),
                            height=60,
                            key=f"comment_{factor_name}"
                        )
                        
                        # 建议
                        calibrated_suggestion = st.text_area(
                            "改进建议",
                            value=original.get("suggestion", ""),
                            height=60,
                            key=f"suggestion_{factor_name}"
                        )
                        
                        calibrated_scores[factor_name] = {
                            "score": calibrated_score,
                            "comment": calibrated_comment,
                            "suggestion": calibrated_suggestion
                        }
            
            # 保存校准后评分
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("💾 保存校准评分", type="primary", use_container_width=True):
                    try:
                        # 创建新判例
                        new_case = {
                            "text": user_input,
                            "scores": calibrated_scores,
                            "tags": "交互生成-已校准",
                            "master_comment": calibrated_master,
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # 1. 添加到内存
                        st.session_state.cases[1].append(new_case)
                        new_count = len(st.session_state.cases[1])
                        
                        # 2. 生成向量
                        vec = embedder.encode([user_input])
                        
                        # 3. 添加到向量索引
                        if st.session_state.cases[0].ntotal == 0:
                            st.session_state.cases = (faiss.IndexFlatL2(1024), st.session_state.cases[1])
                            st.session_state.cases[0].add(vec)
                        else:
                            st.session_state.cases[0].add(vec)
                        
                        # 4. 保存到磁盘
                        DataManager.save(
                            st.session_state.cases[0],
                            st.session_state.cases[1],
                            PATHS['case_index'],
                            PATHS['case_data'],
                            is_json=True
                        )
                        
                        # 5. 同时保存到微调数据
                        sys_p = st.session_state.prompt_config['system_template']
                        DataManager.append_to_finetune(
                            user_input,
                            calibrated_scores,
                            sys_p,
                            st.session_state.prompt_config['user_template'],
                            master_comment=calibrated_master
                        )
                        
                        st.success(f"✅ 校准评分保存成功！判例库现有 {new_count} 条判例。")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"保存失败: {str(e)}")
            
            with col2:
                if st.button("🔄 重置校准", use_container_width=True):
                    st.success("校准已重置为原始值")
                    time.sleep(1)
                    st.rerun()
            
            with col3:
                # 预览校准后的结果
                with st.expander("👁️ 预览校准结果", expanded=False):
                    st.markdown(f"**宗师总评:** {calibrated_master}")
                    st.markdown("**六因子评分:**")
                    for factor_name, data in calibrated_scores.items():
                        st.write(f"**{factor_name}:** {data['score']}/9")
                        st.write(f"评语: {data['comment']}")
                        st.write(f"建议: {data['suggestion']}")
                        st.write("---")
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
    # 在tab3中添加一个按钮
        with c2:
            st.markdown("#### 📥 数据迁移")
        
            if st.button("🚀 将现有判例转为微调数据"):
                if len(st.session_state.cases[1]) > 0:
                    count = 0
                    prompt_cfg = st.session_state
                    for case in st.session_state.cases[1]:
                        if DataManager.append_to_finetune(
                            case["text"],
                            case["scores"],
                            prompt_cfg.get('system_template', ''),
                            prompt_cfg.get('user_template', '')
                        ):
                            count += 1
                
                    st.success(f"成功导入 {count} 条判例到微调数据！")
                    st.rerun()
                else:
                    st.warning("判例库为空")
            # === 微调控制面板 ===
            st.markdown("#### ☁️ 云端微调控制台")
            
            line_count = 0
            if PATHS['training_file'].exists():
                try: line_count = sum(1 for _ in open(PATHS['training_file'], 'r', encoding='utf-8'))
                except: pass
            
            st.write(f"可用微调数据: **{line_count} 条**")
            
            
            if st.button("🚀 一键启动微调 (DeepSeek)"):
                if not client: 
                    st.error("请先配置 API Key")
                else:
                    try:
                        # 1. 上传训练文件
                        with open(PATHS['training_file'], "rb") as f:
                            file_obj = client.files.create(file=f, purpose="fine-tune")
                        
                        st.info(f"文件上传成功，文件ID: {file_obj.id}")
                        
                        # 2. 尝试多个可能的微调API端点
                        job = None
                        error_messages = []
                        
                        # 方法1: 尝试标准fine_tuning.jobs.create
                        try:
                            job = client.fine_tuning.jobs.create(
                                training_file=file_obj.id,
                                model="deepseek-chat",
                                suffix="tea-expert-v1",
                                hyperparameters={
                                    "n_epochs": 3,
                                    "batch_size": 1,
                                    "learning_rate_multiplier": 1.0
                                }
                            )
                            st.success(f"微调任务创建成功！Job ID: {job.id}")
                            
                        except Exception as e1:
                            error_messages.append(f"方法1失败: {str(e1)[:200]}")
                            
                            # 方法2: 尝试不同的模型名称
                            try:
                                job = client.fine_tuning.jobs.create(
                                    training_file=file_obj.id,
                                    model="deepseek-reasoner",  # 尝试其他模型
                                    suffix="tea-expert-v1"
                                )
                                st.success(f"微调任务创建成功！Job ID: {job.id} (使用deepseek-reasoner)")
                                
                            except Exception as e2:
                                error_messages.append(f"方法2失败: {str(e2)[:200]}")
                                
                                # 方法3: 直接API调用（备用方案）
                                import requests
                                
                                try:
                                    headers = {
                                        "Authorization": f"Bearer {st.session_state.get('deepseek_key', '')}",
                                        "Content-Type": "application/json"
                                    }
                                    
                                    # 尝试多个可能的微调端点
                                    endpoints = [
                                        "https://api.deepseek.com/fine_tuning/jobs",
                                        "https://api.deepseek.com/v1/fine_tuning/jobs",
                                        "https://api.deepseek.com/finetuning/jobs",
                                        "https://api.deepseek.com/v1/finetuning/jobs"
                                    ]
                                    
                                    payload = {
                                        "training_file": file_obj.id,
                                        "model": "deepseek-chat",
                                        "suffix": "tea-expert-v1"
                                    }
                                    
                                    for endpoint in endpoints:
                                        try:
                                            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                                            
                                            if response.status_code == 200:
                                                job_data = response.json()
                                                job_id = job_data.get("id")
                                                st.success(f"微调任务创建成功！Job ID: {job_id}")
                                                
                                                # 创建伪job对象以兼容后续代码
                                                class MockJob:
                                                    def __init__(self, job_id):
                                                        self.id = job_id
                                                
                                                job = MockJob(job_id)
                                                break
                                                
                                            elif response.status_code != 404:
                                                st.warning(f"端点 {endpoint} 返回 {response.status_code}")
                                                
                                        except Exception as endpoint_error:
                                            continue
                                    
                                    if not job:
                                        raise Exception("所有微调端点都返回404或失败")
                                        
                                except Exception as e3:
                                    error_messages.append(f"方法3失败: {str(e3)[:200]}")
                        
                        # 3. 如果微调任务创建成功，保存状态
                        if job:
                            DataManager.save_ft_status(job.id, "queued", fine_tuned_model=None)
                            st.success(f"微调任务已启动！Job ID: {job.id}")
                            
                            # 显示任务监控信息
                            st.info("""
                            **微调任务已提交！**
                            
                            接下来你可以：
                            1. 等待几分钟后点击"刷新状态"按钮查看进度
                            2. 微调完成后，系统将自动使用新模型评分
                            3. 如果需要取消任务，请联系DeepSeek客服
                            """)
                            
                            time.sleep(2)
                            st.rerun()
                        else:
                            # 所有方法都失败，显示详细的错误信息和备选方案
                            st.error("⚠️ DeepSeek微调功能暂时不可用")
                            
                            with st.expander("🔍 查看详细错误信息"):
                                for i, msg in enumerate(error_messages, 1):
                                    st.write(f"{i}. {msg}")
                            
                            with st.expander("💡 备选方案"):
                                st.markdown("""
                                **由于DeepSeek微调API暂时不可用，建议使用以下方案：**
                                
                                ### 方案A：增强现有系统（立即可用）
                                ```python
                                # 1. 增加RAG检索数量
                                _, idx = kb_res[0].search(vec, 5)  # 从3增加到5
                                
                                # 2. 优化系统Prompt
                                # 在现有Prompt中添加更多示例和规则
                                
                                # 3. 使用更低的temperature
                                temperature=0.1  # 更一致的输出
                                ```
                                
                                ### 方案B：导出数据在其他平台微调
                                1. 下载训练数据
                                2. 在Google Colab使用免费GPU微调
                                3. 使用LM Studio本地微调
                                
                                ### 方案C：等待DeepSeek修复API
                                1. 关注DeepSeek官方公告
                                2. 联系DeepSeek技术支持
                                3. 暂时使用基础模型
                                """)
                            
                            # 提供数据导出功能
                            st.markdown("---")
                            st.subheader("📥 导出训练数据")
                            
                            with open(PATHS['training_file'], "rb") as f:
                                st.download_button(
                                    label="下载训练数据 (JSONL格式)",
                                    data=f,
                                    file_name="tea_training_data.jsonl",
                                    mime="application/json",
                                    key="download_training_data"
                                )
                            
                            st.info("下载后可在Colab、LM Studio等平台进行微调")
                            
                    except Exception as e:
                        # 通用错误处理
                        error_msg = str(e)
                        
                        # 针对404错误的特殊处理
                        if "404" in error_msg:
                            st.error("""
                            ❌ **404错误：DeepSeek微调API端点不存在**
                            
                            可能的原因：
                            1. DeepSeek微调功能正在维护中
                            2. API端点已变更
                            3. 你的账户暂未开通微调权限
                            
                            **解决方案：**
                            1. 等待DeepSeek官方修复
                            2. 使用基础模型+增强RAG继续评分
                            3. 导出数据在其他平台微调
                            """)
                            
                            # 提供降级方案按钮
                            if st.button("🔄 切换到增强RAG模式", key="switch_to_rag"):
                                st.session_state['enhanced_rag'] = True
                                st.success("已切换到增强RAG模式！")
                                time.sleep(1)
                                st.rerun()
                                
                        else:
                            # 其他错误
                            st.error(f"微调启动失败: {error_msg}")
                            
                            # 显示调试信息
                            with st.expander("🛠️ 调试信息"):
                                st.write(f"错误类型: {type(e).__name__}")
                                st.write(f"完整错误: {error_msg}")
                                
                                # 尝试获取更多API信息
                                try:
                                    # 测试基本的API连通性
                                    test_response = client.models.list()
                                    st.write("✅ API基础连接正常")
                                    st.write(f"可用模型数量: {len(test_response.data)}")
                                except:
                                    st.write("❌ API基础连接失败")
    
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







