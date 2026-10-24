#!/usr/bin/env python3
"""FastAPI web layer for Qgent: paper library, upload pipeline, paper chat.

Three things live here and nothing else does:

  * the library — `/api/papers` reads the `papers` table, `/api/pdf/{id}`
    streams the original PDF that paper was parsed from;
  * the ingest — `/api/upload` writes the PDF into ./pdfs and hands the rest
    to a background task that shells out to the two pipeline stages in order,
    `parse_pdf.py` then `extract_pipeline.py <parsed tree> --write-db`.  The
    stages run one ingest at a time, queued behind whatever is already
    running;
  * the chat — `/api/chat` runs one question through chat_agent's ReAct loop,
    either against one paper or against the whole library.

The agent itself stays framework-free: this module imports `chat_agent` and
builds the same `ChatClient` / `Embedder` / `ToolBox` / `Agent` graph its CLI
builds, rather than teaching it about HTTP.  Paper isolation is therefore
unchanged — every tool call is still filtered by `paper_id` in Python.

A request with no `paper_id` is the multi-paper mode, and it is the same graph
with `paper_id=None`: chat_agent drops the filter, makes `papers` readable and
describes the catalogue in the prompt instead of one paper.  The reply carries
the papers it cited, resolved against the database, so the browser can offer
them as a picker without parsing prose.  The two modes never share a session:
`None` is its own cache key, with its own history and its own connection.

One session per scope is cached, because `Agent.history` is what makes a
follow-up question ("and on the test set?") resolvable at all, and because a
session holds a psycopg2 connection that must not be shared across threads.
FastAPI runs these sync endpoints in a threadpool, so each session carries its
own lock: two questions about one paper are serialised, two papers run
concurrently.

The pipeline stages are run as subprocesses with this interpreter, not
imported: they are standalone CLIs with their own argument parsers, and a
failure inside MinerU or the extraction model must not take the web server
down with it.  Their output is kept in a small ring buffer per task so the
browser can show what the task is actually doing.

Usage:
    python main.py                 # http://127.0.0.1:8000
    uvicorn main:app --reload
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import chat_agent as ca

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "pdfs"
PARSED_DIR = ROOT / "parsed"
INDEX_HTML = ROOT / "index.html"

TREE_NAME = "content_tree.json"

# chat_agent already loaded the .env beside it at import time.
DSN = os.getenv("DATABASE_URL", ca.DEFAULT_DSN)

# MinerU on a long paper is minutes, not seconds; the extraction model calls
# out once per section.  PARSE_TIMEOUT covers one parse_pdf run, which skips
# the PDFs that already have a content tree.  EXTRACT_TIMEOUT covers one
# *paper*, because the ingest imports a single tree rather than scanning the
# directory: the largest paper here (665 blocks) took about 40 minutes.
PARSE_TIMEOUT = 3600
EXTRACT_TIMEOUT = 5400

# Log lines kept per task, for the progress panel in the sidebar.
LOG_TAIL = 40

# Sessions hold a PostgreSQL connection each.  Papers are opened by clicking,
# so an unbounded cache would exhaust max_connections on a long session; the
# least recently used one is dropped instead.
MAX_SESSIONS = 8

app = FastAPI(title="Qgent", docs_url="/docs")


# --------------------------------------------------------------------------- #
# Background tasks
# --------------------------------------------------------------------------- #

# task_id -> {status, stage, filename, message, log, error, paper_id}
TASKS: dict[str, dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()

# One ingest at a time.  parse_pdf.py walks the whole ./pdfs directory and
# extract_pipeline.py holds a transaction per paper, so two uploads started
# together would re-extract the same paper, race each other's inserts and
# multiply the model spend.  A second upload therefore waits here while its
# card shows the same `queued` stage the front end already paints.
INGEST_LOCK = threading.Lock()


def task_update(task_id: str, **fields: Any) -> None:
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if task is not None:
            task.update(fields)


def task_log(task_id: str, lines: list[str]) -> None:
    """Append to the task's log and mirror the last line into `message`."""
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if task is None:
            return
        task["log"].extend(lines)
        del task["log"][:-LOG_TAIL]
        tail = [line.strip() for line in lines if line.strip()]
        if tail:
            task["message"] = tail[-1]


def run_stage(task_id: str, stage: str, script: str, argv: list[str], timeout: int) -> int:
    """Run one pipeline stage, streaming its output into the task log.

    Streamed rather than captured whole: a MinerU parse is minutes long, and
    the sidebar card shows the last line as the task's progress.  stderr is
    folded into stdout so the two stay in the order the stage wrote them.
    """
    task_update(task_id, stage=stage, message=f"{script} 启动中…")
    cmd = [sys.executable, str(ROOT / script), *argv]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise RuntimeError(f"无法启动 {script}: {exc}") from None

    deadline = time.monotonic() + timeout
    try:
        for line in proc.stdout:
            task_log(task_id, [line])
            if time.monotonic() > deadline:
                proc.kill()
                raise RuntimeError(f"{script} 超时（>{timeout}s）")
        proc.wait()
    finally:
        if proc.poll() is None:  # the timeout path killed it; a crash may not have
            proc.kill()
            proc.wait()
    return proc.returncode


def tree_title(stem: str) -> str | None:
    path = PARSED_DIR / stem / TREE_NAME
    try:
        with path.open(encoding="utf-8") as fh:
            title = json.load(fh).get("title")
    except (OSError, json.JSONDecodeError):
        return None
    return title if isinstance(title, str) and title.strip() else None


def paper_id_for_title(title: str) -> int | None:
    """The paper row this content-tree title produced, if any.

    Both keys are checked, for the same reason `load_known_titles` in
    extract_pipeline.py does: `papers.title` holds the title the model read
    out of the paper, which it may have re-cased or normalised, while
    `papers.content_tree ->> 'title'` holds the tree's own title -- the one
    `tree_title` just returned.  Matching only the former reports an import
    as failed for every paper the model renamed on the way in (for example
    "CONTRASTIVE LEARNING WITH HARD NEGATIVE SAMPLES", stored as
    "Contrastive Learning with Hard Negative Samples").
    """
    conn = ca.connect_readonly(DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM papers
                WHERE title = %s OR content_tree ->> 'title' = %s
                ORDER BY id LIMIT 1
                """,
                (title, title),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def run_ingest(task_id: str, stem: str) -> None:
    """parse_pdf.py, then extract_pipeline.py, for one freshly uploaded PDF.

    Serialised behind INGEST_LOCK: while another ingest is running this one
    waits, and its card keeps showing the `queued` stage.  Without that lock
    two uploads share ./pdfs and ./parsed, so each would re-extract the other's
    paper and the two runs would race the same `papers` row.
    """
    if not INGEST_LOCK.acquire(blocking=False):
        task_update(task_id, stage="queued", message="排队中：等待上一篇入库完成…")
        INGEST_LOCK.acquire()
    try:
        run_ingest_locked(task_id, stem)
    finally:
        INGEST_LOCK.release()


def run_ingest_locked(task_id: str, stem: str) -> None:
    """The body of run_ingest; the caller holds INGEST_LOCK."""
    try:
        run_stage(
            task_id,
            "parsing",
            "parse_pdf.py",
            ["--pdf-dir", str(PDF_DIR), "--out-dir", str(PARSED_DIR)],
            PARSE_TIMEOUT,
        )
        # parse_pdf walks the whole directory and reports a non-zero exit if
        # any *other* PDF in it failed, so its exit code cannot decide this
        # upload's fate.  The artefact can: a content tree for this stem.
        if not (PARSED_DIR / stem / TREE_NAME).is_file():
            raise RuntimeError("解析失败：未生成 content_tree.json，请查看日志")
        title = tree_title(stem)
        task_update(task_id, stage="extracting", title=title)

        # One tree, not --scan: this upload imports this paper and nothing
        # else, so a tree another upload left behind in ./parsed is never
        # picked up here and cannot eat this paper's time budget.
        #
        # A single-tree run does not dedup on its own -- it would extract for
        # forty minutes and only then be refused at insert by store_paper --
        # so the check `--scan` used to do for free happens here instead.
        existing = paper_id_for_title(title) if title else None
        if existing is not None:
            invalidate_pdf_index()
            task_update(
                task_id,
                status="done",
                stage="done",
                paper_id=existing,
                message="论文已存在，跳过入库",
            )
            return

        run_stage(
            task_id,
            "extracting",
            "extract_pipeline.py",
            [
                str(PARSED_DIR / stem / TREE_NAME),
                "--write-db",
                "--print-rows",
                "0",
            ],
            EXTRACT_TIMEOUT,
        )

        # "No error" is not the same as "imported": the stage prints its own
        # result, but the row is what counts.
        paper_id = paper_id_for_title(title) if title else None
        if paper_id is None:
            raise RuntimeError(
                f"入库失败：数据库中找不到《{title}》，请查看日志"
                if title
                else "入库失败：content tree 中没有标题，无法确认入库结果"
            )

        invalidate_pdf_index()
        # A multi-paper session's prompt lists the catalogue, so a new paper
        # makes it stale.  Dropping it here means the next question rebuilds it
        # and can cite the paper that just arrived.
        drop_session(None)
        task_update(
            task_id,
            status="done",
            stage="done",
            paper_id=paper_id,
            message="已完成解析与入库",
        )
    except Exception as exc:  # surfaced in the sidebar card, not raised at the client
        task_update(task_id, status="failed", stage="failed", error=str(exc), message=str(exc))


# --------------------------------------------------------------------------- #
# PDF lookup: papers.id -> the PDF in ./pdfs it came from
# --------------------------------------------------------------------------- #

# The database records a paper by title, not by filename; the filename only
# survives in ./parsed/<stem>/ and ./pdfs/<stem>.pdf.  The title inside the
# content tree is what links them, so an index is built once and cached.
_index_lock = threading.Lock()
_pdf_index: dict[str, str] | None = None

NON_WORD_RE = re.compile(r"[^0-9a-z一-鿿]+")


def normalise_title(value: str) -> str:
    return NON_WORD_RE.sub("", value.lower())


def invalidate_pdf_index() -> None:
    global _pdf_index
    with _index_lock:
        _pdf_index = None


def pdf_stem_index() -> dict[str, str]:
    """normalised title (and stem) -> stem, for every parsed paper with a PDF."""
    global _pdf_index
    with _index_lock:
        if _pdf_index is None:
            index: dict[str, str] = {}
            for tree_path in sorted(PARSED_DIR.glob(f"*/{TREE_NAME}")):
                stem = tree_path.parent.name
                if not (PDF_DIR / f"{stem}.pdf").is_file():
                    continue
                try:
                    with tree_path.open(encoding="utf-8") as fh:
                        title = json.load(fh).get("title")
                except (OSError, json.JSONDecodeError):
                    continue
                for key in (title, stem):
                    if isinstance(key, str) and key.strip():
                        index.setdefault(normalise_title(key), stem)
            _pdf_index = index
        return _pdf_index


def resolve_pdf(paper_id: int) -> Path:
    conn = ca.connect_readonly(DSN)
    try:
        with conn.cursor() as cur:
            # content_tree->>'title' is the tree's own title, which is exactly
            # the key the index was built from; papers.title is the model's
            # version of it and is only a fallback.
            cur.execute(
                "SELECT title, content_tree->>'title' FROM papers WHERE id = %s",
                (paper_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"papers.id={paper_id} 不存在")

    index = pdf_stem_index()
    for candidate in (row[1], row[0]):
        if isinstance(candidate, str) and candidate.strip():
            stem = index.get(normalise_title(candidate))
            if stem:
                return PDF_DIR / f"{stem}.pdf"

    raise HTTPException(
        status_code=404,
        detail=f"在 ./pdfs 中找不到《{row[0]}》对应的 PDF 文件（parsed/ 目录下缺少同名解析结果）",
    )


# --------------------------------------------------------------------------- #
# Chat sessions
# --------------------------------------------------------------------------- #


class ChatSession:
    """One scope's agent: its own read-only connection, client and history.

    `paper_id=None` is the multi-paper scope.  It is not a different graph --
    the same client, embedder and connection are built either way -- only a
    different prompt, a different ToolBox scope and a larger step budget.
    """

    def __init__(self, paper_id: int | None) -> None:
        llm_api_key = (
            os.getenv("LLM_API_KEY", "").strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()
        )
        if not llm_api_key:
            raise HTTPException(status_code=503, detail="服务端未配置 LLM_API_KEY（见 .env）")
        embed_api_key = (
            os.getenv("EMBEDDING_API_KEY", "").strip()
            or os.getenv("DASHSCOPE_API_KEY", "").strip()
            or llm_api_key
        )

        self.paper_id = paper_id
        self.lock = threading.Lock()
        self.conn = ca.connect_readonly(DSN)
        try:
            if paper_id is None:
                # The catalogue is read once, here, and baked into the prompt:
                # the model cannot cite a paper_id it was never told exists.
                library, counts, chunk_types = ca.library_context(self.conn)
                self.title = "多篇全局（跨库检索）"
                system_prompt = ca.build_multi_system_prompt(library, counts, chunk_types)
                max_steps = ca.DEFAULT_MULTI_MAX_STEPS
            else:
                title, counts, chunk_types = ca.paper_context(self.conn, paper_id)
                self.title = title
                system_prompt = ca.build_system_prompt(paper_id, title, counts, chunk_types)
                max_steps = ca.DEFAULT_MAX_STEPS

            self.agent = ca.Agent(
                client=ca.ChatClient(
                    base_url=os.getenv("LLM_BASE_URL", ca.DEFAULT_BASE_URL),
                    api_key=llm_api_key,
                    model=os.getenv("LLM_MODEL", ca.DEFAULT_LLM_MODEL),
                    timeout=ca.DEFAULT_TIMEOUT,
                    max_retries=ca.DEFAULT_MAX_RETRIES,
                    max_tokens=ca.DEFAULT_MAX_TOKENS,
                    temperature=ca.DEFAULT_TEMPERATURE,
                    # Hybrid-reasoning models do not serve non-streaming tool
                    # calls with thinking on.
                    enable_thinking=False,
                ),
                toolbox=ca.ToolBox(
                    conn=self.conn,
                    embedder=ca.Embedder(
                        model=os.getenv("EMBEDDING_MODEL", ca.DEFAULT_EMBEDDING_MODEL),
                        api_key=embed_api_key,
                        max_chars=ca.DEFAULT_EMBED_MAX_CHARS,
                        dimension=ca.DEFAULT_DIMENSION,
                    ),
                    paper_id=paper_id,
                    top_k=ca.DEFAULT_TOP_K,
                    max_rows=ca.DEFAULT_SQL_ROWS,
                    max_cell_chars=ca.DEFAULT_CELL_CHARS,
                ),
                system_prompt=system_prompt,
                max_steps=max_steps,
                history_rounds=ca.DEFAULT_HISTORY_ROUNDS,
                verify_evidence=True,
            )
        except Exception:
            self.conn.close()
            raise

    def ask(self, message: str) -> dict[str, Any]:
        # psycopg2 connections are not thread-safe, and the agent's history is
        # ordered state: one question at a time per scope.
        with self.lock:
            reply = self.agent.ask(message)
            # Which papers the answer actually cited, already checked against
            # the database by the agent.  In a single-paper session this is
            # empty: the reader is already looking at that paper.
            citations = self.agent.last_citations
        return {"reply": reply, "citations": citations}

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


_sessions: dict[int | None, ChatSession] = {}
_sessions_lock = threading.Lock()


def get_session(paper_id: int | None) -> ChatSession:
    with _sessions_lock:
        session = _sessions.pop(paper_id, None)
        if session is None:
            session = ChatSession(paper_id)  # may raise HTTPException
        while len(_sessions) >= MAX_SESSIONS:
            oldest_id = next(iter(_sessions))
            _sessions.pop(oldest_id).close()
        _sessions[paper_id] = session  # re-inserted last: most recently used
        return session


def drop_session(paper_id: int | None) -> None:
    """Forget one session, closing its connection.

    The session's own lock is taken first: a question may be running on it, and
    closing a psycopg2 connection underneath a live query is not something the
    caller can recover from.  Waiting is cheap for the one caller there is --
    the upload task, which has just spent minutes inside the pipeline.
    """
    with _sessions_lock:
        session = _sessions.pop(paper_id, None)
    if session is None:
        return
    with session.lock:
        session.close()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/")
def index() -> FileResponse:
    if not INDEX_HTML.is_file():
        raise HTTPException(status_code=404, detail="index.html 不存在")
    # No caching: this is a dev tool and the page is edited in place.
    return FileResponse(INDEX_HTML, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/api/papers")
def list_papers() -> list[dict[str, Any]]:
    """Every paper in the library, newest first."""
    conn = ca.connect_readonly(DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, domain, created_at FROM papers ORDER BY created_at DESC, id DESC"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {
            "id": row[0],
            "title": row[1],
            "domain": row[2] or "",
            "created_at": row[3].isoformat() if row[3] else None,
        }
        for row in rows
    ]


@app.get("/api/pdf/{paper_id}")
def get_pdf(paper_id: int) -> FileResponse:
    """The original PDF, streamed for PDF.js.

    FileResponse streams from disk in chunks and answers Range requests with
    206, so PDF.js fetches only the byte ranges it needs and the file is never
    held in memory.  The hour of caching is what makes paging back and forth
    cheap: a re-visited page is served from the browser cache instead of being
    re-read over the wire.  Safe to cache because an upload never overwrites a
    PDF another paper was parsed from (see `upload`).
    """
    path = resolve_pdf(paper_id)
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/api/upload")
async def upload(background: BackgroundTasks, file: UploadFile = File(...)) -> dict[str, Any]:
    """Store the PDF and start parsing + extraction in the background."""
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只接受 .pdf 文件")

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    dest = PDF_DIR / f"{stem}.pdf"
    # Never overwrite a PDF another paper was parsed from: its content tree
    # would no longer describe the file on disk.
    suffix = 2
    while dest.exists():
        dest = PDF_DIR / f"{stem} ({suffix}).pdf"
        suffix += 1

    with dest.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)

    task_id = uuid.uuid4().hex
    with TASKS_LOCK:
        TASKS[task_id] = {
            "task_id": task_id,
            "status": "processing",
            "stage": "queued",
            "filename": dest.name,
            "message": "已保存，等待后台处理…",
            "log": [],
            "error": None,
            "paper_id": None,
            "title": None,
        }
    background.add_task(run_ingest, task_id, dest.stem)
    return {"task_id": task_id, "filename": dest.name}


@app.get("/api/status/{task_id}")
def status(task_id: str) -> dict[str, Any]:
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"未知任务 {task_id}")
        return dict(task)


class ChatRequest(BaseModel):
    # Absent or null is the multi-paper mode: the question is answered against
    # the whole library, and the reply names the papers it cited.
    paper_id: int | None = None
    message: str


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """One question, answered in the scope `paper_id` names.

    Returns the reply and, in multi-paper mode, the papers it cited as
    `[{"paper_id": 4, "title": "..."}]` in the order they were mentioned --
    resolved against the database, so the browser can render a picker from it
    without parsing the answer itself.
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    try:
        session = get_session(request.paper_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None

    try:
        return session.ask(message)
    except ca.ChatError as exc:
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from None
    except Exception as exc:
        # A dead connection is not reusable; drop the session so the next
        # question rebuilds it instead of failing forever.
        with _sessions_lock:
            if _sessions.get(request.paper_id) is session:
                _sessions.pop(request.paper_id, None)
        session.close()
        raise HTTPException(status_code=500, detail=f"问答失败：{exc}") from None


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
