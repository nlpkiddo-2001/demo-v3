"""Tool definitions and dispatch for the chat agents.

Each tool has:
  1. an OpenAI-style JSON schema (so the LLM knows how to call it), and
  2. an async handler that actually runs it.

CRM tools are LIVE — they hit the real Zoho CRM account via chat/crm.py.
Support and Travel tools are SIMULATED — they return realistic mock data so the
demo is self-contained and reliable (no external dependencies).

To add a tool: write a `_schema` entry and a handler, then register both in
TOOLS at the bottom.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Awaitable, Callable

from . import db
from .crm import ZohoError, crm

log = logging.getLogger("chat.tools")


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Build one OpenAI function-tool schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ── CRM tools (LIVE Zoho) ────────────────────────────────────────────────
_MODULE_ENUM = {"type": "string", "enum": ["Leads", "Contacts"], "description": "CRM module."}

CRM_SCHEMAS = [
    _fn(
        "crm_search",
        "Search the CRM for leads or contacts by a keyword (name, email, company, or phone). "
        "Use this to find a record's id before updating it.",
        {"module": _MODULE_ENUM, "query": {"type": "string", "description": "Search keyword."}},
        ["module", "query"],
    ),
    _fn(
        "crm_list_recent",
        "List the most recently modified leads or contacts. Useful to show the user what exists.",
        {"module": _MODULE_ENUM, "limit": {"type": "integer", "description": "Max records (default 5)."}},
        ["module"],
    ),
    _fn(
        "crm_get",
        "Fetch full details of one lead or contact by its record id.",
        {"module": _MODULE_ENUM, "record_id": {"type": "string", "description": "The Zoho record id."}},
        ["module", "record_id"],
    ),
    _fn(
        "crm_create_lead",
        "Create a NEW lead in the CRM. Last_Name and Company are required by Zoho.",
        {
            "Last_Name": {"type": "string"},
            "First_Name": {"type": "string"},
            "Company": {"type": "string"},
            "Email": {"type": "string"},
            "Phone": {"type": "string"},
            "Lead_Status": {"type": "string", "description": "e.g. 'Not Contacted', 'Contacted', 'Qualified'."},
            "Lead_Source": {"type": "string"},
            "Designation": {"type": "string"},
            "Description": {"type": "string"},
        },
        ["Last_Name"],
    ),
    _fn(
        "crm_update_record",
        "Update fields on an EXISTING lead or contact. First find the record_id with crm_search. "
        "Pass only the fields you want to change inside 'fields'.",
        {
            "module": _MODULE_ENUM,
            "record_id": {"type": "string", "description": "The Zoho record id to update."},
            "fields": {
                "type": "object",
                "description": "Field name → new value, e.g. {\"Lead_Status\": \"Qualified\", \"Phone\": \"+91...\"}.",
            },
        },
        ["module", "record_id", "fields"],
    ),
]


async def _crm_search(module: str, query: str, **_) -> Any:
    return await crm.search(module, query)


async def _crm_list_recent(module: str, limit: int = 5, **_) -> Any:
    return await crm.list_recent(module, limit=limit)


async def _crm_get(module: str, record_id: str, **_) -> Any:
    rec = await crm.get(module, record_id)
    return rec or {"error": "not_found", "record_id": record_id}


async def _crm_create_lead(**fields) -> Any:
    return await crm.create_lead(fields)


async def _crm_update_record(module: str, record_id: str, fields: dict, **_) -> Any:
    return await crm.update_record(module, record_id, fields or {})


# ── Support tools (LIVE, backed by the SQLite ticket DB) ──────────────────
_KB = {
    "refund": "Refunds are processed within 5–7 business days to the original payment method.",
    "shipping": "Standard shipping takes 3–5 business days; express is 1–2 days.",
    "password": "Reset your password via 'Forgot password' on the login screen; the link is valid 30 minutes.",
    "cancel": "You can cancel an order within 1 hour of placing it from Orders → Cancel.",
    "hours": "Support is available Monday–Saturday, 9am–9pm IST.",
}

SUPPORT_SCHEMAS = [
    _fn(
        "ticket_create",
        "Create a NEW support ticket in the database for a customer. Call this once you "
        "have the customer's name and a clear description of their issue. Ask for their "
        "email too so they can be found again in future sessions.",
        {
            "customer_name": {"type": "string", "description": "The customer's full name."},
            "subject": {"type": "string", "description": "A short one-line summary of the issue."},
            "description": {"type": "string", "description": "The full details of the problem."},
            "customer_email": {"type": "string", "description": "Customer email (used to look them up later)."},
            "priority": {"type": "string", "enum": ["Low", "Normal", "High", "Urgent"]},
            "category": {"type": "string", "description": "e.g. Account, Billing, Technical, General."},
        },
        ["customer_name", "subject", "description"],
    ),
    _fn(
        "ticket_lookup",
        "Find a customer's existing tickets in the database. Use this when a returning "
        "customer asks about a ticket. Search by their email, their name, or a ticket "
        "reference like 'TKT-1001'. Returns matching tickets with status, stage and next action.",
        {"query": {"type": "string", "description": "Customer email, name, or ticket reference."}},
        ["query"],
    ),
    _fn(
        "ticket_get",
        "Get the full details and update history of one ticket by its reference (e.g. 'TKT-1001').",
        {"ticket_id": {"type": "string", "description": "The ticket reference, e.g. TKT-1001."}},
        ["ticket_id"],
    ),
    _fn(
        "ticket_update",
        "Update a ticket's status/stage or add a note (e.g. mark 'In Progress' or 'Resolved'). "
        "Statuses: Open, In Progress, Waiting on Customer, Resolved, Closed.",
        {
            "ticket_id": {"type": "string"},
            "status": {"type": "string", "enum": list(db.VALID_STATUSES)},
            "note": {"type": "string", "description": "A short note about this update."},
            "next_action": {"type": "string", "description": "Override what happens next (optional)."},
        },
        ["ticket_id"],
    ),
    _fn(
        "support_search_kb",
        "Search the help center for a quick answer to a general policy question "
        "(refunds, shipping, password, cancellation, hours).",
        {"query": {"type": "string"}},
        ["query"],
    ),
]


def _slim_ticket(t: dict | None) -> dict | None:
    if not t:
        return None
    keep = ("ref", "customer_name", "customer_email", "subject", "priority",
            "status", "stage", "next_action", "category", "created_at", "updated_at")
    out = {k: t.get(k) for k in keep if t.get(k) not in (None, "")}
    if t.get("history"):
        out["history"] = t["history"]
    return out


async def _ticket_create(customer_name: str, subject: str, description: str = "",
                         customer_email: str = "", priority: str = "Normal",
                         category: str = "General", **_) -> Any:
    t = await asyncio.to_thread(db.create_ticket, customer_name, subject, description,
                                customer_email, priority, category)
    return {"success": True, "ticket": _slim_ticket(t),
            "message": f"Ticket {t['ref']} created for {customer_name}."}


async def _ticket_lookup(query: str, **_) -> Any:
    rows = await asyncio.to_thread(db.find_tickets, query)
    if not rows:
        return {"found": 0, "tickets": [],
                "message": "No tickets found for that. You may not have raised one yet."}
    return {"found": len(rows), "tickets": [_slim_ticket(r) for r in rows]}


async def _ticket_get(ticket_id: str, **_) -> Any:
    t = await asyncio.to_thread(db.get_ticket, ticket_id)
    if not t:
        return {"error": f"No ticket found with reference {ticket_id}."}
    return {"ticket": _slim_ticket(t)}


async def _ticket_update(ticket_id: str, status: str | None = None,
                         note: str | None = None, next_action: str | None = None, **_) -> Any:
    t = await asyncio.to_thread(db.update_ticket, ticket_id, status, None, next_action, note)
    if isinstance(t, dict) and t.get("error"):
        return t
    return {"success": True, "ticket": _slim_ticket(t)}


async def _support_search_kb(query: str, **_) -> Any:
    q = (query or "").lower()
    hits = [{"topic": k, "answer": v} for k, v in _KB.items() if k in q]
    if not hits:
        hits = [{"topic": k, "answer": v} for k, v in _KB.items()
                if any(w in v.lower() for w in q.split())][:2]
    return {"results": hits or [{"topic": "general",
            "answer": "No article found — offer to raise a ticket for a human agent."}]}


# ── Travel tools (SIMULATED) ─────────────────────────────────────────────
TRAVEL_SCHEMAS = [
    _fn(
        "travel_search_flights",
        "Search for flights between two cities on a date.",
        {
            "origin": {"type": "string", "description": "Origin city or airport."},
            "destination": {"type": "string"},
            "date": {"type": "string", "description": "Departure date, YYYY-MM-DD."},
        },
        ["origin", "destination", "date"],
    ),
    _fn(
        "travel_search_hotels",
        "Search for hotels in a city for a date range.",
        {
            "city": {"type": "string"},
            "checkin": {"type": "string", "description": "YYYY-MM-DD"},
            "checkout": {"type": "string", "description": "YYYY-MM-DD"},
        },
        ["city"],
    ),
    _fn(
        "travel_book",
        "Book a flight or hotel option by the id returned from a search.",
        {"item_id": {"type": "string"}, "traveller_name": {"type": "string"}},
        ["item_id"],
    ),
]

_AIRLINES = ["IndiGo", "Air India", "Vistara", "Akasa Air"]
_HOTELS = ["The Grand Regency", "Seaside Suites", "Urban Nest", "Palm Court Residency"]


async def _travel_search_flights(origin: str, destination: str, date: str, **_) -> Any:
    seed = int(hashlib.md5(f"{origin}{destination}{date}".encode()).hexdigest(), 16)
    flights = []
    for i in range(3):
        s = seed >> (i * 8)
        flights.append({
            "id": f"FL-{(s % 9000) + 1000}",
            "airline": _AIRLINES[s % len(_AIRLINES)],
            "depart": f"{6 + (s % 14):02d}:{(s % 6) * 10:02d}",
            "duration_min": 90 + (s % 180),
            "stops": s % 2,
            "price": f"₹{(s % 8000) + 3499}",
        })
    return {"origin": origin, "destination": destination, "date": date, "flights": flights}


async def _travel_search_hotels(city: str, checkin: str = "", checkout: str = "", **_) -> Any:
    seed = int(hashlib.md5(f"{city}{checkin}".encode()).hexdigest(), 16)
    hotels = []
    for i in range(3):
        s = seed >> (i * 8)
        hotels.append({
            "id": f"HT-{(s % 9000) + 1000}",
            "name": _HOTELS[s % len(_HOTELS)],
            "rating": round(3.6 + (s % 14) / 10, 1),
            "price_per_night": f"₹{(s % 12000) + 2499}",
            "area": ["City Centre", "Airport", "Beachfront", "Old Town"][s % 4],
        })
    return {"city": city, "checkin": checkin, "checkout": checkout, "hotels": hotels}


async def _travel_book(item_id: str, traveller_name: str = "Guest", **_) -> Any:
    ref = hashlib.md5((item_id + traveller_name).encode()).hexdigest()[:8].upper()
    return {"success": True, "booking_ref": ref, "item_id": item_id, "traveller": traveller_name,
            "status": "CONFIRMED"}


# ── Shared: ask the user to TYPE a value (voice → text handoff) ───────────
# Speech recognition mangles emails / phone numbers / exact name spellings, so
# the agent calls this to pop an on-screen text box instead. In the voice path
# (chat/voice_agent.py) this also emits a "request_text" event that pauses the
# mic; here we just return an instruction the model can act on.
REQUEST_TEXT_SCHEMA = _fn(
    "request_typed_input",
    "Ask the user to TYPE a value into an on-screen text box instead of saying it aloud. "
    "ALWAYS use this for emails, phone numbers, and the exact spelling of names — spoken "
    "input mis-hears these. After calling it, tell the user to type it and wait for their reply.",
    {
        "field": {"type": "string", "description": "What to collect, e.g. 'email', 'phone number', 'full name'."},
        "prompt": {"type": "string", "description": "Short instruction shown above the box."},
    },
    ["field"],
)


async def _request_typed_input(field: str = "information", prompt: str = "", **_) -> Any:
    return {"status": "text_box_shown", "field": field,
            "message": f"A text box for '{field}' was shown. Ask the user to type it; "
                       "their typed value arrives as their next message."}


# ── Shared: end the voice call when the conversation is over ───────────────
END_CALL_SCHEMA = _fn(
    "end_call",
    "End the voice call. Call this ONLY when the conversation has clearly concluded — "
    "the user said goodbye / 'that's all' / 'thanks, nothing else' / 'bye', or their "
    "request is fully resolved and they have nothing more. Give a short, warm spoken "
    "farewell in the same turn; the call hangs up right after you finish speaking.",
    {"farewell": {"type": "string", "description": "Optional one-line goodbye to say."}},
    [],
)


async def _end_call(farewell: str = "", **_) -> Any:
    return {"status": "ending_call",
            "message": "The call will hang up after your spoken farewell. Say a brief, "
                       "warm goodbye now (one short sentence)."}


# ── registry ──────────────────────────────────────────────────────────────
Handler = Callable[..., Awaitable[Any]]

TOOLS: dict[str, tuple[dict, Handler]] = {}


def _register(schemas: list[dict], handlers: dict[str, Handler]) -> None:
    for schema in schemas:
        name = schema["function"]["name"]
        TOOLS[name] = (schema, handlers[name])


_register(CRM_SCHEMAS, {
    "crm_search": _crm_search,
    "crm_list_recent": _crm_list_recent,
    "crm_get": _crm_get,
    "crm_create_lead": _crm_create_lead,
    "crm_update_record": _crm_update_record,
})
_register(SUPPORT_SCHEMAS, {
    "ticket_create": _ticket_create,
    "ticket_lookup": _ticket_lookup,
    "ticket_get": _ticket_get,
    "ticket_update": _ticket_update,
    "support_search_kb": _support_search_kb,
})
_register(TRAVEL_SCHEMAS, {
    "travel_search_flights": _travel_search_flights,
    "travel_search_hotels": _travel_search_hotels,
    "travel_book": _travel_book,
})
_register([REQUEST_TEXT_SCHEMA], {"request_typed_input": _request_typed_input})
_register([END_CALL_SCHEMA], {"end_call": _end_call})


def schemas_for(tool_names: list[str]) -> list[dict]:
    """Return the JSON schemas for a set of tool names (for the LLM request)."""
    return [TOOLS[n][0] for n in tool_names if n in TOOLS]


async def run_tool(name: str, args: dict) -> Any:
    """Execute a tool by name. Returns a JSON-serializable result (never raises)."""
    if name not in TOOLS:
        return {"error": f"unknown tool '{name}'"}
    _, handler = TOOLS[name]
    try:
        return await handler(**(args or {}))
    except ZohoError as e:
        log.warning("CRM tool %s failed: %s", name, e)
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:  # noqa: BLE001 — tools must never crash the chat loop
        log.exception("Tool %s crashed", name)
        return {"error": f"{type(e).__name__}: {e}"}
