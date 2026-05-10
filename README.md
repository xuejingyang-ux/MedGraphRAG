# MedGraphRAG（`app.py` 主线版）

本仓库当前版本已按 `app.py` 主应用进行清理：仅保留与 Flask 问答系统直接相关的 Python 脚本，删除了旧版 Streamlit/NER/微调等无关 Python 文件。

---

## 核心功能

- 医疗知识问答：`/api/ask`
- 图谱可视化与检索：`/api/graph/full`、`/api/graph/search`
- JSONL 入图：`/api/import/jsonl`
- 问答历史管理：`/api/history`
- 系统状态巡检：`/api/status`

---

## 当前保留的 Python 文件

```text
app.py
llm_client.py
knowledge_graph.py
vector_retriever.py
src/__init__.py
src/kg_builder.py
```

---

## 环境准备

建议 Python 3.10。

```bash
git clone https://github.com/xuejingyang-ux/MedGraphRAG.git
cd MedGraphRAG

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

如果缺少运行依赖，请补充安装：

```bash
pip install flask neo4j
```

---

## `.env` 配置

在项目根目录创建 `.env`：

```env
# LLM
LLM_PROVIDER=zhipu
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=Pro/zai-org/GLM-4.7

# Neo4j
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# 可选：向量检索编码器配置
EMBEDDING_USE_HF=0
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_LOCAL_ONLY=1
```

---

## 启动

```bash
python app.py
```

访问：

- [http://127.0.0.1:5001](http://127.0.0.1:5001)

---

## 数据与图谱说明

### 1) Neo4j 模式（推荐）

配置好 `NEO4J_*` 后，应用会直接读写 Neo4j。

### 2) 本地回退模式

若 Neo4j 不可用，系统会回退到本地数据构建图谱（优先读取 `data/medical_new_2.json`，其次 `data/medical.json`），核心问答链路仍可运行。

---

## JSONL 入图接口

请求：

```bash
curl -X POST http://127.0.0.1:5001/api/import/jsonl ^
  -H "Content-Type: application/json" ^
  -d "{\"path\":\"data/medical_corpus.jsonl\",\"reset\":false,\"max_records\":0}"
```

单条 JSONL 建议字段示例：

```json
{"title":"肺炎概述","source":"manual","text":"肺炎是...","disease":"肺炎","category":"呼吸系统"}
```

---

## 常用 API

- `GET /api/status`
- `POST /api/ask`
- `GET /api/graph/full?limit=300`
- `GET /api/graph/search?q=关键词`
- `POST /api/import/jsonl`
- `GET /api/history?limit=50`
- `DELETE /api/history`

问答示例：

```bash
curl -X POST http://127.0.0.1:5001/api/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"肺炎的常见症状有哪些？\"}"
```

---

