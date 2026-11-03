# -*- coding: utf-8 -*-
"""
茅台 2024 年报 · 智能问答（Streamlit 前端）
==========================================
运行：双击 run_ui.bat，或在本目录执行
      E:\\Users\\lenovo\\miniconda3\\envs\\langchain1.2\\python.exe -m streamlit run app.py
"""
from __future__ import annotations

import itertools
import uuid

import streamlit as st

st.set_page_config(
    page_title="茅台 2024 年报 · 智能问答",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from rag_engine import RagEngine  # noqa: E402  （必须在 set_page_config 之后导入）

# ------------------------------------------------------------------ 样式
st.markdown(
    """
    <style>
      .block-container {max-width: 1080px; padding-top: 2.2rem;}
      .wb-title {font-size: 1.55rem; font-weight: 700; margin-bottom: .1rem;}
      .wb-sub {color: #7a8087; font-size: .86rem; margin-bottom: 1.1rem;}
      .src-chip {
        display: inline-block; padding: .12rem .5rem; margin: .12rem .28rem .12rem 0;
        border: 1px solid rgba(128,128,128,.35); border-radius: .55rem;
        font-size: .78rem; color: #6b7280;
      }
      [data-testid="stSidebar"] .stSlider {padding-top: .2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ 引擎
@st.cache_resource(show_spinner="正在加载向量库与精排模型（首次约 60 秒，之后秒开）…")
def load_engine() -> RagEngine:
    return RagEngine()


# ------------------------------------------------------------------ 引擎加载
try:
    engine = load_engine()
    st.session_state["engine"] = engine
except Exception as exc:  # 缺库、缺 key、编号错位都在这里兜住
    st.error(f"引擎初始化失败：{exc}")
    st.info(
        "常见原因：① 向量库 chroma_db_v2 不存在 → 先在 notebook 的「3.2 建库」跑一次；"
        "② .env 里缺 DASHSCOPE_API_KEY / DEEPSEEK_API_KEY；"
        "③ 余额不足（阿里云 Arrearage / DeepSeek Insufficient Balance）。"
    )
    st.stop()

# ------------------------------------------------------------------ 会话 ID
# 多轮记忆的隔离键：换一个 id 就是一段新对话（checkpointer 按 thread_id 存历史）
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex[:8]

# ------------------------------------------------------------------ 侧边栏
with st.sidebar:
    st.markdown("### ⚙️ 检索参数")
    use_rewrite = st.toggle("查询改写", value=True, help="把口语问题改写成年报术语，能显著提升召回")
    use_memory = st.toggle(
        "多轮记忆",
        value=False,
        help=(
            "开启后由 LangGraph checkpointer 保存对话历史，追问（「那它的增长率呢」）"
            "能正确承接上文。关闭则是单轮独立检索——评测用的就是这一模式。"
        ),
    )
    fetch_k = st.slider("精排保留子块数 fetch_k", 3, 15, 5, help="越大召回越全，但噪声也越多")
    n_pages = st.slider("回填父块数上限 n_pages", 2, 8, 6, help="决定送进模型的上下文长度")
    vec_w = st.slider("向量检索权重", 0.0, 1.0, 0.6, 0.1, help="其余权重给 BM25 关键词检索")
    show_context = st.toggle("显示上下文原文", value=False)

    st.divider()
    st.markdown("### 📚 知识库")
    if "engine" in st.session_state:
        e = st.session_state["engine"]
        st.caption(
            f"年报 {e.pdf_path.name}｜父块 {len(e.parents_store)}｜子块 {len(e.children)}"
            f"｜向量库 {len(e.db_pids)} 个父块"
        )
    else:
        st.caption("加载中…")

    st.divider()
    if st.button("🧹 清空对话", width="stretch"):
        old = st.session_state.get("thread_id", "")
        if "engine" in st.session_state and old:
            st.session_state["engine"].reset_memory(old)  # 释放旧会话的短期记忆
        st.session_state.messages = []
        st.session_state.thread_id = uuid.uuid4().hex[:8]
        st.rerun()
    if use_memory:
        st.caption(f"🧠 多轮记忆已开启 · 会话 `{st.session_state.thread_id}`")
    else:
        st.caption("💬 单轮模式：每问独立检索，不携带历史")
    st.caption("答案均由年报原文生成，末尾标注依据页码；找不到时明确回复「资料中未提及」。")

# ------------------------------------------------------------------ 标题
st.markdown('<div class="wb-title">📊 茅台 2024 年报 · 智能问答</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wb-sub">基于 143 页年度报告构建的中文 RAG 系统 · '
    "父子块索引 + BM25/向量混合召回 + 交叉编码器精排</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ 会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_details(info: dict, show_ctx: bool) -> None:
    """把一次检索的中间结果摊开展示，方便核对答案依据。"""
    pages = [str(p["page"]) for p in info["picked"]]
    chips = "".join(f'<span class="src-chip">第 {p} 页</span>' for p in pages)
    st.markdown(chips, unsafe_allow_html=True)
    st.caption(
        f"召回 {info['raw_n']} 条子块 → 命中 {len(info['picked'])} 个父块"
        f"｜上下文 {len(info['context'])} 字｜检索耗时 {info['elapsed']:.1f}s"
    )
    with st.expander("检索详情", expanded=False):
        if info["rewritten"] != info["question"]:
            st.markdown(f"**改写后的问题**：`{info['rewritten']}`")
        tab1, tab2, tab3 = st.tabs(["命中父块", "精排明细", "上下文原文"])
        with tab1:
            st.dataframe(
                [
                    {"页码": p["page"], "父块编号": p["pid"], "精排分数": round(p["score"], 4)}
                    for p in info["picked"]
                ],
                width="stretch",
                hide_index=True,
            )
        with tab2:
            st.dataframe(
                [
                    {"页码": r["page"], "精排分数": round(r["score"], 4), "片段": r["snippet"]}
                    for r in info["ranked"]
                ],
                width="stretch",
                hide_index=True,
            )
        with tab3:
            if show_ctx:
                st.text_area("送入模型的上下文", info["context"], height=280, label_visibility="collapsed")
            else:
                st.caption("左侧开启「显示上下文原文」后可查看全文。")

# ------------------------------------------------------------------ 历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("info"):
            render_details(msg["info"], show_context)

# ------------------------------------------------------------------ 示例问题
if not st.session_state.messages:
    st.markdown("##### 试试这些问题")
    samples = [
        "2024年营业收入是多少？",
        "茅台去年赚了多少钱？",
        "公司的核心竞争力是什么？",
        "每10股派发现金红利多少元？",
        "董事长的手机号是多少？",  # 故意问不到，用来验证拒答
    ]
    cols = st.columns(len(samples))
    for col, q in zip(cols, samples):
        if col.button(q, width="stretch"):
            st.session_state.pending = q
            st.rerun()

# ------------------------------------------------------------------ 提问
prompt = st.chat_input("输入你的问题，例如：2024年加权平均净资产收益率是多少？")
if prompt is None:
    prompt = st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if use_memory:
                # 多轮：先跑 condense + retrieve（这步慢），再流式吐生成结果
                with st.spinner("理解上下文并检索中…"):
                    gen = engine.chat_stream(
                        prompt,
                        thread_id=st.session_state.thread_id,
                        fetch_k=fetch_k,
                        n_pages=n_pages,
                        weights=(round(1 - vec_w, 2), vec_w),
                    )
                    try:
                        first = next(gen)  # 触发 condense + retrieve
                    except StopIteration:
                        first = ""
                answer = (
                    st.write_stream(itertools.chain([first], gen)) if first else ""
                )
                info = engine.last_info
            else:
                with st.spinner("检索中…"):
                    info = engine.retrieve(
                        prompt,
                        fetch_k=fetch_k,
                        n_pages=n_pages,
                        use_rewrite=use_rewrite,
                        weights=(round(1 - vec_w, 2), vec_w),
                    )
                answer = st.write_stream(engine.stream_answer(prompt, info))
        except Exception as exc:
            answer = f"生成失败：{exc}"
            info = None
            st.error(answer)
            if "Insufficient Balance" in str(exc) or "402" in str(exc):
                st.info("DeepSeek 余额不足，请到 platform.deepseek.com 充值后重试。")
            elif "Arrearage" in str(exc):
                st.info("阿里云百炼欠费，向量化调用被拒绝，请充值后重试。")

        if info:
            render_details(info, show_context)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer or "", "info": info}
    )
