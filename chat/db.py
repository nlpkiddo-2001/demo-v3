"""Tiny persistent ticket store (SQLite) for the support-agent demo.

This is the "dummy DB" the customer-support agent reads and writes. It persists
to a real file on disk, so a ticket raised in one chat session can be looked up
in a completely new session (new tab, new conversation, or after a restart).

Two tables:
  tickets         — one row per support ticket (status, stage, next action, …)
  ticket_updates  — an audit trail of status/stage changes per ticket

Everything here is synchronous sqlite3; the async tool handlers call these via
asyncio.to_thread so they never block the event loop.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "tickets.db"
_LOCK = threading.Lock()
_conn: sqlite3.Connection | None = None

# Lifecycle a ticket moves through. Each status has a default "next action"
# describing what the support team will do next — shown to the customer.
STATUS_FLOW = {
    "Open":                {"stage": "Received",            "next_action": "A support engineer will triage your ticket within 24 hours."},
    "In Progress":         {"stage": "Under investigation", "next_action": "An engineer is actively working on a fix and will update you soon."},
    "Waiting on Customer": {"stage": "Awaiting your reply", "next_action": "We're waiting for the information we requested from you to continue."},
    "Resolved":            {"stage": "Fix delivered",       "next_action": "Please confirm the issue is resolved so we can close the ticket."},
    "Closed":              {"stage": "Closed",              "next_action": "This ticket is closed. Reply or raise a new ticket if you need more help."},
}
VALID_STATUSES = tuple(STATUS_FLOW.keys())


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ref           TEXT UNIQUE,
            customer_name TEXT,
            customer_email TEXT,
            subject       TEXT,
            description   TEXT,
            category      TEXT,
            priority      TEXT,
            status        TEXT,
            stage         TEXT,
            next_action   TEXT,
            created_at    TEXT,
            updated_at    TEXT
        );
        CREATE TABLE IF NOT EXISTS ticket_updates (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_ref TEXT,
            ts         TEXT,
            status     TEXT,
            note       TEXT
        );
        """
    )
    conn.commit()


def _ref_for(ticket_id: int) -> str:
    return f"TKT-{1000 + ticket_id}"


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


# ── writes ────────────────────────────────────────────────────────────────
def create_ticket(customer_name: str, subject: str, description: str = "",
                  customer_email: str = "", priority: str = "Normal",
                  category: str = "General") -> dict:
    flow = STATUS_FLOW["Open"]
    with _LOCK:
        conn = _get_conn()
        cur = conn.execute(
            "INSERT INTO tickets (customer_name, customer_email, subject, description, "
            "category, priority, status, stage, next_action, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (customer_name, customer_email, subject, description, category, priority,
             "Open", flow["stage"], flow["next_action"], _now(), _now()),
        )
        tid = cur.lastrowid
        ref = _ref_for(tid)
        conn.execute("UPDATE tickets SET ref=? WHERE id=?", (ref, tid))
        conn.execute(
            "INSERT INTO ticket_updates (ticket_ref, ts, status, note) VALUES (?,?,?,?)",
            (ref, _now(), "Open", "Ticket created."),
        )
        conn.commit()
    return get_ticket(ref)


def update_ticket(ref: str, status: str | None = None, stage: str | None = None,
                  next_action: str | None = None, note: str | None = None) -> dict:
    existing = get_ticket(ref)
    if not existing:
        return {"error": f"No ticket found with reference {ref}."}
    # If a known status is given, default the stage/next_action from the flow
    # unless the caller overrode them explicitly.
    if status and status in STATUS_FLOW:
        stage = stage or STATUS_FLOW[status]["stage"]
        next_action = next_action or STATUS_FLOW[status]["next_action"]
    sets, vals = [], []
    for col, val in (("status", status), ("stage", stage), ("next_action", next_action)):
        if val is not None:
            sets.append(f"{col}=?"); vals.append(val)
    with _LOCK:
        conn = _get_conn()
        if sets:
            sets.append("updated_at=?"); vals.append(_now()); vals.append(ref)
            conn.execute(f"UPDATE tickets SET {', '.join(sets)} WHERE ref=?", vals)
        conn.execute(
            "INSERT INTO ticket_updates (ticket_ref, ts, status, note) VALUES (?,?,?,?)",
            (ref, _now(), status or existing["status"], note or "Ticket updated."),
        )
        conn.commit()
    return get_ticket(ref)


# ── reads ─────────────────────────────────────────────────────────────────
def get_ticket(ref: str) -> dict | None:
    with _LOCK:
        conn = _get_conn()
        row = conn.execute("SELECT * FROM tickets WHERE ref=?", (ref.strip(),)).fetchone()
        if not row:
            return None
        t = _row_to_dict(row)
        updates = conn.execute(
            "SELECT ts, status, note FROM ticket_updates WHERE ticket_ref=? ORDER BY id",
            (ref,),
        ).fetchall()
    t["history"] = [_row_to_dict(u) for u in updates]
    return t


def find_tickets(query: str = "", limit: int = 10) -> list[dict]:
    """Find tickets by ticket ref, customer email, or customer name (fuzzy)."""
    q = (query or "").strip()
    with _LOCK:
        conn = _get_conn()
        if not q:
            rows = conn.execute(
                "SELECT * FROM tickets ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            like = f"%{q}%"
            rows = conn.execute(
                "SELECT * FROM tickets WHERE ref=? OR customer_email LIKE ? "
                "OR customer_name LIKE ? OR subject LIKE ? ORDER BY id DESC LIMIT ?",
                (q, like, like, like, limit),
            ).fetchall()
    return [_row_to_dict(r) for r in rows]


def stats() -> dict:
    with _LOCK:
        conn = _get_conn()
        n = conn.execute("SELECT COUNT(*) AS c FROM tickets").fetchone()["c"]
    return {"tickets": n, "path": str(DB_PATH)}


def seed_if_empty() -> None:
    """Insert a couple of demo tickets so 'look up an existing ticket' works out
    of the box. Only runs when the DB is empty."""
    if stats()["tickets"] > 0:
        return
    t1 = create_ticket(
        customer_name="Ananya Rao", customer_email="ananya.rao@example.com",
        subject="Unable to reset my password",
        description="Password reset email never arrives.",
        priority="High", category="Account",
    )
    update_ticket(t1["ref"], status="In Progress",
                  note="Escalated to the identity team; reproduced the issue.")
    t2 = create_ticket(
        customer_name="Rahul Menon", customer_email="rahul.menon@example.com",
        subject="Refund not received for order ORD-2231",
        description="Refund approved 6 days ago but not credited.",
        priority="Normal", category="Billing",
    )
    update_ticket(t2["ref"], status="Waiting on Customer",
                  note="Requested the customer's bank reference number.")
