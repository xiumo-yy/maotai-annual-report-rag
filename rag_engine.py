# -*- coding: utf-8 -*-
"""
RAG 引擎（前端专用）
====================
把 test.ipynb 里已经验证过的流水线封装成一个类，前端 app.py 直接调用，不用重复实现：

    PDF 加载 -> 页眉清洗 -> 父子块切分 -> 千问向量库(chroma_db_v2)
    -> BM25 + 向量混合召回 -> bge-reranker 精排 -> 父块回填 -> DeepSeek 生成

设计要点
--------
1. 向量库直接读磁盘上已有的 chroma_db_v2，不重新 embedding（省钱、秒级启动）；
2. 父子块用 pickle 缓存，第二次启动跳过 PDF 解析；缓存带签名，PDF 或切分参数变了自动重建；
3. 启动时校验「库里的 parent_id」与「内存里的父块编号」完全一致，不一致直接报错，
   避免出现「检索对了、回填错段落」这种不报错的静默错误；
4. 单轮用 answer()/stream_answer()，多轮用 chat()/chat_stream()（LangGraph + checkpointer）。
   多轮的真难点不是存历史，而是先把追问改写成独立问句再检索，见 _build_memory()。
"""
from __future__ import annotations

import os
import pickle
import re
import time
from pathlib import Path
from typing import Annotated, Callable, Iterator, TypedDict

from dotenv import load_dotenv

# 必须在模块级导入：StateGraph 用 get_type_hints 解析注解，
# 而本模块启用了 `from __future__ import annotations`，注解会被字符串化后在模块全局作用域求值。
from langgraph.graph.message import add_messages

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

# ---------------------------------------------------------------- 配置区
REPORT_YEAR = 2024
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
EMBED_MODEL = "text-embedding-v4"
EMBED_DIM = 1024
COLLECTION = "maotai_v4_v2"
DB_DIR = BASE_DIR / "chroma_db_v2"
INDEX_CACHE = BASE_DIR / "store" / "index_cache.pkl"

# 切分参数（必须与 notebook 完全一致，否则 parent_id 会错位）
PARENT_SIZE, PARENT_OVERLAP = 1500, 0
CHILD_SIZE, CHILD_OVERLAP = 300, 50
SEPS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

PDF_CANDIDATES = [
    BASE_DIR / "贵州茅台2024年年度报告.pdf",
    Path.home() / "Desktop" / "贵州茅台2024年年度报告.pdf",
]

# 页眉：公司名 + 年度报告 + 页码（与 notebook 一致）
HDR = re.compile(
    r"^\s*贵州茅台酒股份有限公司\s*\d*\s*年年度报告\s*$\s*\n?\s*\d+\s*/\s*\d+\s*$\s*",
    re.MULTILINE,
)

Log = Callable[[str], None]


def _noop(msg: str) -> None:
    pass


def strip_header(text: str) -> str:
    return HDR.sub("", text).strip()


def find_pdf() -> Path:
    for p in PDF_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "找不到年报 PDF，请把「贵州茅台2024年年度报告.pdf」放到项目目录下。"
    )


def find_reranker() -> str:
    """优先用 modelscope 本地缓存，没有再联网下载。"""
    env = os.getenv("RERANKER_DIR")
    if env and Path(env).exists():
        return env
    cache = Path.home() / ".cache" / "modelscope" / "models" / "BAAI--bge-reranker-base" / "snapshots" / "master"
    if (cache / "config.json").exists():
        return str(cache)
    from modelscope import snapshot_download

    return snapshot_download("BAAI/bge-reranker-base")


def build_index(pdf: Path, log: Log = _noop) -> tuple[dict[str, str], list]:
    """页眉清洗 + 父子块切分，结果缓存到 store/index_cache.pkl。"""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    sig = {
        "name": pdf.name,
        "size": pdf.stat().st_size,
        "mtime": int(pdf.stat().st_mtime),
        "parent": (PARENT_SIZE, PARENT_OVERLAP),
        "child": (CHILD_SIZE, CHILD_OVERLAP),
        "v": 2,
    }
    if INDEX_CACHE.exists():
        try:
            with open(INDEX_CACHE, "rb") as f:
                cached = pickle.load(f)
            if cached.get("sig") == sig:
                log(f"父子块命中缓存：父块 {len(cached['parents'])}，子块 {len(cached['children'])}")
                return cached["parents"], [
                    Document(page_content=d["page_content"], metadata=d["metadata"])
                    for d in cached["children"]
                ]
            log("PDF 或切分参数已变化，重建父子块…")
        except Exception as e:  # 缓存损坏就重建，不影响使用
            log(f"缓存读取失败（{e}），重建…")

    log("正在解析 PDF 并按页清洗页眉…")
    docs = PyPDFLoader(str(pdf)).load()
    docs_clean = [
        Document(page_content=strip_header(d.page_content), metadata=d.metadata) for d in docs
    ]

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_SIZE, chunk_overlap=PARENT_OVERLAP, separators=SEPS
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_SIZE, chunk_overlap=CHILD_OVERLAP, separators=SEPS
    )

    parents = parent_splitter.split_documents(docs_clean)
    parents_store: dict[str, str] = {}
    children: list = []
    for p_idx, p_doc in enumerate(parents):
        pid = f"p{p_idx}"
        parents_store[pid] = p_doc.page_content
        for c_doc in child_splitter.split_documents([p_doc]):
            children.append(
                Document(
                    page_content=c_doc.page_content,
                    metadata={
                        **c_doc.metadata,
                        "parent_id": pid,
                        "chunk_id": f"{pid}_c{len(children)}",
                    },
                )
            )

    log(f"切分完成：父块 {len(parents_store)}，子块 {len(children)}")
    INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_CACHE, "wb") as f:
        pickle.dump(
            {
                "sig": sig,
                "parents": parents_store,
                "children": [
                    {"page_content": c.page_content, "metadata": c.metadata} for c in children
                ],
            },
            f,
        )
    return parents_store, children


class RagEngine:
    """一次初始化，反复提问。前端用 @st.cache_resource 缓存实例。"""

    def __init__(self, log: Log = _noop):
        from langchain_deepseek import ChatDeepSeek
        from langchain_openai import OpenAIEmbeddings

        t0 = time.time()
        # 1) 模型
        self.llm = ChatDeepSeek(
            model=LLM_MODEL, temperature=0, api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        self.embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL,
            dimensions=EMBED_DIM,
            chunk_size=10,
            check_embedding_ctx_length=False,  # DashScope 只收字符串，不加会报错
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
        )

        # 2) 父子块
        self.pdf_path = find_pdf()
        self.parents_store, self.children = build_index(self.pdf_path, log)

        # 3) 向量库（只读，不重建）
        if not DB_DIR.exists():
            raise FileNotFoundError(
                f"找不到向量库 {DB_DIR.name}，请先在 notebook 的「3.2 建库」那一格跑一次。"
            )
        from langchain_chroma import Chroma

        self.vs = Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=self.embeddings,
            collection_name=COLLECTION,
        )
        n_vec = self.vs._collection.count()
        log(f"向量库加载完成：{n_vec} 条")

        # 4) 一致性校验：库里的 parent_id 必须全在父块字典里
        metas = self.vs._collection.get(include=["metadatas"])["metadatas"] or []
        db_pids = {m.get("parent_id") for m in metas}
        missing = db_pids - set(self.parents_store)
        if missing:
            raise RuntimeError(
                f"向量库与内存父块编号不一致（库里多出 {len(missing)} 个 parent_id，"
                f"例如 {sorted(missing)[:3]}）。请重新执行 notebook 的 2.2 切分与 3.2 建库，"
                "确保两者来自同一次运行。"
            )
        if n_vec != len(self.children):
            log(f"提示：库内 {n_vec} 条，内存 {len(self.children)} 条，检索仍可用")
        self.db_pids = db_pids

        # 5) 混合召回
        import jieba
        from langchain_classic.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever

        def zh_tokenize(text: str) -> list[str]:
            return [w for w in jieba.cut_for_search(text) if w.strip()]

        self.bm25 = BM25Retriever.from_documents(
            self.children, k=10, preprocess_func=zh_tokenize
        )
        self.vec = self.vs.as_retriever(search_kwargs={"k": 10})
        self._Ensemble = EnsembleRetriever

        # 6) 精排
        from sentence_transformers import CrossEncoder

        self.reranker = CrossEncoder(find_reranker(), max_length=512)
        log("精排模型就绪")

        # 7) 两条链（单轮）
        self._build_chains()
        # 8) 多轮记忆（LangGraph + checkpointer），不影响上面的单轮链路
        self._build_memory()
        log(f"引擎初始化完成，耗时 {time.time() - t0:.1f}s")

    # ------------------------------------------------------------ 链路构建
    def _build_chains(self):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        rewrite_prompt = ChatPromptTemplate.from_template(
            "你是企业年报知识库的检索查询改写助手。\n"
            "知识库内容为贵州茅台2024年年度报告。\n\n"
            "【时间归一】\n"
            "用户说“去年”“今年”“本年度”“前年”等相对时间时，一律按 {RY} 年计算，"
            "不要按当前日期推算。\n\n"
            "【改写要求】\n"
            "1. 补全实体：“茅台”补成“贵州茅台”，“年报/报告/公司”理解为“{RY}年年度报告”。\n"
            "2. 术语对齐：把口语替换成年报中的标准表述，例如\n"
            "   赚了多少钱/净利润 → 归属于上市公司股东的净利润\n"
            "   营收/收入 → 营业收入\n"
            "   每股赚多少 → 基本每股收益\n"
            "   分红/派息 → 现金红利（每10股）\n"
            "   负债水平 → 资产负债率\n"
            "3. 保留疑问焦点：写清问的是“是多少”“有哪些”“是否”“占比多少”中的哪一种，"
            "不要把问句缩成名词堆砌。\n"
            "4. 数字、年份、专有名词原样保留，不增不删不换算。\n"
            "5. 控制在30字以内，不要堆砌同义词。\n\n"
            "【约束】\n"
            "- 只输出改写结果一行，不要解释，不要加引号或“改写后：”之类的前缀\n"
            "- 若问题涉及年报通常不披露的内容（如个人联系方式、未披露的细分数据），"
            "原样保留问题，不要强行改写成看似合理的财务指标\n"
            "- 不要臆造年报中不存在的指标名称\n\n"
            "问题：{q}\n"
            "改写结果："
        )

        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是贵州茅台2024年年度报告的财报分析助手。你唯一的资料来源是下面提供的年报片段，"
                    "除此之外没有任何其他信息来源，不要进行与片段无关的联想生成。\n\n"
                    "【回答规则】\n"
                    "1. 事实与数字只能来自资料原文，并照抄原文的数值、单位与口径。\n"
                    "   不要换算（如把元改成亿元）、不要四舍五入、不要补全省略的位数；\n"
                    "   也不要自行做加减乘除或推算增减额——凡原文未直接给出的派生数值，一律不要计算。\n"
                    "2. 资料中没有明确写出的内容，一律回答「资料中未提及」。不要用行业常识、"
                    "往年数据、公开新闻或逻辑推理来填补。宁可少答，不可猜测。\n"
                    "3. 资料只覆盖部分内容时，明确说明哪部分有、哪部分没有，"
                    "例如「资料提及了A，但未提及B」。\n"
                    "4. 回答要完整成句，让不看检索过程的人也能读懂。先点明问的是哪个指标或事项，"
                    "再给数值与同比变化。\n"
                    "   示例：用户问「去年赚了多少钱」，应回答「您问的应是归属于上市公司股东的净利润，"
                    "2024年为86,228,146,421.62元，同比增长15.38%」，而不是只回一个数字。\n"
                    "5. 同一指标在资料中出现多个版本时（合并报表/母公司报表、本期/上期、"
                    "含税/不含税、营业收入/营业总收入、净利润/归母净利润），必须逐一区分，"
                    "只采用与问题相符的那一个，并在数值后注明口径，如「（合并报表口径）」。"
                    "严禁把不同口径的数值混用、相加或相互比较。问题未指明口径时，"
                    "默认采用合并报表本期数据。\n"
                    "6. 不要整段复述资料原文，但也不要只甩一串数字。回答控制在350字以内，"
                    "问什么答什么，不罗列资料里其他无关数据，不做趋势预测或投资建议。\n"
                    "7. 回答末尾单独一行注明依据页码，格式「依据：第X页」；"
                    "同一信息多页出现时只引最直接的那一页。资料未提及时写「依据：无」。\n"
                    "8. 资料确实没有对应内容时，除回答「资料中未提及」外，"
                    "补一句说明查的是哪项信息，例如「资料中未提及研发人员数量的披露」。\n\n"
                    "【资料】\n{context}",
                ),
                ("human", "{question}"),
            ]
        )

        self.rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
        self.rag_chain = rag_prompt | self.llm | StrOutputParser()

    # ------------------------------------------------------------ 多轮记忆
    def _build_memory(self):
        """短期记忆：LangGraph StateGraph + checkpointer。

        关键认知：checkpointer 只负责「历史存在哪」，不会让检索自动用上历史。
        真正让多轮可用的是 condense 节点——先用历史把追问改写成独立问句，再去检索。
        """
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.graph.message import add_messages

        # ---------- 节点1：指代消解
        condense_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是企业年报知识库的检索查询改写助手。知识库内容为贵州茅台{RY}年年度报告。\n\n"
                    "【任务】把用户的最新提问改写成一个不依赖上下文、可直接用于检索的独立问句。\n\n"
                    "【对话历史】\n{history}\n\n"
                    "【改写要求】\n"
                    "1. 指代消解：若提问含「它/这个/那个/上面/该指标/那…呢/继续」等指代，"
                    "必须结合对话历史补全成完整问句。例如上文问净利润、用户接着问「那增长率呢」，"
                    "应改写为「贵州茅台{RY}年归属于上市公司股东的净利润同比增长率是多少」。\n"
                    "2. 补全实体：「茅台」补成「贵州茅台」。\n"
                    "3. 术语对齐：赚了多少钱→归属于上市公司股东的净利润；营收→营业收入；"
                    "每股赚多少→基本每股收益；分红→现金红利（每10股）；负债水平→资产负债率。\n"
                    "4. 保留疑问焦点，写清「是多少/有哪些/是否/占比多少」。\n"
                    "5. 数字、年份、专有名词原样保留，不增不删不换算。\n"
                    "6. 若提问是关于对话本身（如「我上一个问题是什么」「刚才问了什么」）"
                    "或与年报完全无关，原样返回，不要改写。\n"
                    "7. 控制在 30 字以内，不要堆砌同义词。\n\n"
                    "只输出一行改写结果，不要解释，不要加引号或「改写后：」之类的前缀。",
                ),
                ("human", "{question}"),
            ]
        )

        # ---------- 节点3：生成（带对话历史）
        mem_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是贵州茅台2024年年度报告的财报分析助手。事实与数字只能来自【资料】，"
                    "资料之外的任何内容都不得作为答案依据。\n\n"
                    "【回答规则】\n"
                    "0. 若用户问的是对话本身（如「我上一个问题是什么」「刚才问了什么」），"
                    "直接依据【对话历史】如实回答，不需要年报资料，末尾写「依据：无」。\n"
                    "0.5 【检索问句】是系统已根据上文把追问补全后的独立问句。用户原话常是"
                    "「那它的增长率呢」「这个呢」这类省略句，此时一律以【检索问句】为回答目标，"
                    "直接把该指标答出来；严禁回答「您的问题未指明具体指标」「请明确所指的指标」"
                    "这类要用户再澄清的话——指代消解是系统的职责，不该推回给用户。\n"
                    "1. 事实与数字只能来自资料原文，并照抄原文的数值、单位与口径。"
                    "不要换算（如把元改成亿元）、不要四舍五入、不要补全省略的位数；"
                    "也不要自行做加减乘除或推算增减额。\n"
                    "   反例（禁止）：把 86,228,146,421.62 元写成「862.28亿元」。\n"
                    "   正例（要求）：2024年为86,228,146,421.62元，同比增长15.38%。\n"
                    "2. 资料中没有明确写出的内容，一律回答「资料中未提及」。不要用行业常识、"
                    "往年数据、公开新闻或逻辑推理来填补。宁可少答，不可猜测。\n"
                    "3. 资料只覆盖部分内容时，明确说明哪部分有、哪部分没有。\n"
                    "4. 回答要完整成句，先点明问的是哪个指标或事项，再给数值与同比变化。\n"
                    "5. 同一指标出现多个版本时（合并报表/母公司报表、本期/上期、含税/不含税、"
                    "营业收入/营业总收入、净利润/归母净利润），必须逐一区分，"
                    "只采用与问题相符的那一个，禁止混用、相加或相互比较。\n"
                    "6. 不做趋势预测或投资建议，回答控制在 350 字以内。\n"
                    "7. 回答末尾另起一行注明依据，格式「依据：第X页」；资料未提及时写「依据：无」。\n\n"
                    "【对话历史】\n{history}\n\n【检索问句】\n{standalone}\n\n【资料】\n{context}",
                ),
                ("human", "{question}"),
            ]
        )

        self.condense_chain = condense_prompt | self.llm | StrOutputParser()
        self.mem_chain = mem_prompt | self.llm | StrOutputParser()

        class MemState(TypedDict, total=False):
            messages: Annotated[list, add_messages]  # checkpointer 负责跨轮累积
            params: dict                             # 本轮检索参数
            standalone: str                          # 消解指代后的检索句
            context: str
            info: dict

        def fmt_history(msgs: list, limit: int = 6) -> str:
            if not msgs:
                return "（无，这是对话的第一轮）"
            lines = []
            for m in msgs[-limit:]:
                role = "用户" if isinstance(m, HumanMessage) else "助手"
                text = str(m.content).replace("\n", " ")[:200]
                lines.append(f"{role}：{text}")
            return "\n".join(lines)

        def node_condense(state: MemState):
            msgs = state["messages"]
            question = msgs[-1].content
            # 第一轮也要走改写：condense 除了消解指代，还负责术语对齐
            # （「赚了多少钱」→「归属于上市公司股东的净利润」），跳过会直接检索跑偏。
            try:
                q = self.condense_chain.invoke(
                    {
                        "history": fmt_history(msgs[:-1]),
                        "question": question,
                        "RY": REPORT_YEAR,
                    }
                ).strip()
            except Exception:
                q = question  # 改写失败就退回原问题，不让整条链路断掉
            return {"standalone": q or question}

        def node_retrieve(state: MemState):
            params = dict(state.get("params") or {})
            params["use_rewrite"] = False  # condense 已做术语对齐，避免二次改写
            info = self.retrieve(state["standalone"], **params)
            return {"context": info["context"], "info": info}

        def node_generate(state: MemState):
            msgs = state["messages"]
            standalone = state.get("standalone") or msgs[-1].content
            ans = self.mem_chain.invoke(
                {
                    "history": fmt_history(msgs[:-1]),
                    "context": state.get("context", ""),
                    "question": msgs[-1].content,
                    "standalone": standalone,
                }
            )
            return {"messages": [AIMessage(content=ans)]}

        g = StateGraph(MemState)
        g.add_node("condense", node_condense)
        g.add_node("retrieve", node_retrieve)
        g.add_node("generate", node_generate)
        g.add_edge(START, "condense")
        g.add_edge("condense", "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)

        # InMemorySaver 随进程存活；要跨进程持久化需换 SqliteSaver / PostgresSaver
        self.memory_graph = g.compile(checkpointer=InMemorySaver())
        self.last_info: dict | None = None

    # ------------------------------------------------------------ 多轮问答
    @staticmethod
    def _thread_cfg(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def chat(self, question: str, thread_id: str = "default", **kw) -> tuple[str, dict]:
        """多轮版（非流式）：历史由 checkpointer 保存，检索前先做指代消解。"""
        from langchain_core.messages import HumanMessage

        out = self.memory_graph.invoke(
            {"messages": [HumanMessage(content=question)], "params": kw},
            self._thread_cfg(thread_id),
        )
        info = dict(out.get("info") or {})
        info["question"] = question
        info["rewritten"] = out.get("standalone", question)
        return out["messages"][-1].content, info

    def chat_stream(self, question: str, thread_id: str = "default", **kw):
        """多轮版（流式）：只吐出 generate 节点的 token，结束后从 last_info 取检索信息。"""
        from langchain_core.messages import AIMessageChunk, HumanMessage

        cfg = self._thread_cfg(thread_id)
        self.last_info = None
        inputs = {"messages": [HumanMessage(content=question)], "params": kw}
        streamed = False
        for mode, payload in self.memory_graph.stream(
            inputs, cfg, stream_mode=["messages", "values"]
        ):
            if mode != "messages":
                continue
            chunk, meta = payload
            if (meta or {}).get("langgraph_node") != "generate":
                continue
            # 只收流式增量（AIMessageChunk）。节点结束时返回的完整 AIMessage
            # 也会从 messages 模式冒出一次，不过滤会把答案重复输出两遍。
            if not isinstance(chunk, AIMessageChunk):
                continue
            text = chunk.content
            if text:
                streamed = True
                yield text

        state = self.memory_graph.get_state(cfg).values or {}
        info = dict(state.get("info") or {})
        info["question"] = question
        info["rewritten"] = state.get("standalone", question)
        self.last_info = info

        # 兜底：模型没有流式返回时（例如被代理吞掉），直接吐完整答案，避免空白
        if not streamed:
            msgs = state.get("messages") or []
            if msgs and getattr(msgs[-1], "content", ""):
                yield msgs[-1].content

    def reset_memory(self, thread_id: str) -> None:
        """清空某条会话的短期记忆。"""
        try:
            self.memory_graph.checkpointer.delete_thread(thread_id)
        except Exception:
            pass  # checkpointer 不提供删除时忽略，前端换 thread_id 也能达到同样效果

    # ------------------------------------------------------------ 检索
    def rewrite(self, question: str) -> str:
        try:
            return self.rewrite_chain.invoke({"q": question, "RY": REPORT_YEAR}).strip()
        except Exception:
            return question  # 改写失败就退回原问题，不让整条链路断掉

    def rerank(self, question: str, docs: list, top_n: int = 5):
        if not docs:
            return []
        scores = self.reranker.predict([(question, d.page_content) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return [(docs[i], float(scores[i])) for i in order[:top_n]]

    def retrieve(
        self,
        question: str,
        fetch_k: int = 5,
        n_pages: int = 6,
        use_rewrite: bool = True,
        weights: tuple[float, float] = (0.4, 0.6),
    ) -> dict:
        """跑一遍检索，返回上下文与全部中间结果，供界面展示。"""
        t0 = time.time()
        q = self.rewrite(question) if use_rewrite else question

        ensemble = self._Ensemble(
            retrievers=[self.bm25, self.vec], weights=list(weights)
        )
        raw = ensemble.invoke(q)
        ranked = self.rerank(question, raw, top_n=max(fetch_k, 1))

        seen, picked = set(), []
        for d, score in ranked:
            pid = d.metadata["parent_id"]
            if pid in seen:
                continue
            seen.add(pid)
            picked.append({"pid": pid, "page": d.metadata.get("page_label"), "score": score})
            if len(picked) >= n_pages:
                break

        context = "\n\n".join(
            f"[来源: 第{item['page']}页]\n{self.parents_store[item['pid']]}" for item in picked
        )

        return {
            "question": question,
            "rewritten": q,
            "raw_n": len(raw),
            "ranked": [
                {
                    "page": d.metadata.get("page_label"),
                    "score": s,
                    "snippet": d.page_content[:80].replace("\n", " "),
                }
                for d, s in ranked
            ],
            "picked": picked,
            "context": context,
            "elapsed": time.time() - t0,
        }

    # ------------------------------------------------------------ 生成
    def stream_answer(self, question: str, info: dict) -> Iterator[str]:
        for chunk in self.rag_chain.stream({"context": info["context"], "question": question}):
            yield chunk

    def answer(self, question: str, **kw) -> tuple[str, dict]:
        info = self.retrieve(question, **kw)
        return self.rag_chain.invoke(
            {"context": info["context"], "question": question}
        ), info
