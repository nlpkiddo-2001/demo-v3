"""Live Zoho CRM client for the chat demo — Leads & Contacts.

What does this file do?
------------------------
This is a small, self-contained async client for the Zoho CRM v8 REST API.
It handles OAuth (refresh-token flow) and the handful of record operations the
sales agent needs: search, read, create a lead, update a record, and list the
most recently modified records.

It is intentionally standalone (does not import the older ``llm_agents``
graph_executor) so the demo has one clean, dependable code path.

Auth
----
Zoho access tokens expire every hour. We exchange the long-lived
``refresh_token`` for a fresh access token and cache it until ~2 minutes before
expiry, so callers never deal with expired tokens.

Data center
-----------
Everything is region-specific. For India:
  accounts host : accounts.zoho.in       (token endpoint)
  API base      : https://www.zohoapis.in/crm/v8
Both come from settings (ZOHO_ACCOUNTS_HOST / ZOHO_API_BASE_URL in .env).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from server.config import settings

log = logging.getLogger("chat.crm")

# Fields we read back for each module so the agent has something useful to show.
_MODULE_FIELDS = {
    "Leads": (
        "id,First_Name,Last_Name,Full_Name,Email,Phone,Mobile,Company,"
        "Lead_Status,Lead_Source,Designation,City,Description,Modified_Time"
    ),
    "Contacts": (
        "id,First_Name,Last_Name,Full_Name,Email,Phone,Mobile,Account_Name,"
        "Title,Mailing_City,Description,Modified_Time"
    ),
}

SUPPORTED_MODULES = tuple(_MODULE_FIELDS.keys())


class ZohoError(RuntimeError):
    """Raised when a Zoho API call fails in a way the agent should hear about."""


class ZohoCRM:
    """Async Zoho CRM client with cached refresh-token auth.

    Usage:
        crm = ZohoCRM()
        if crm.configured:
            leads = await crm.search("Leads", "acme")
    """

    def __init__(self) -> None:
        self.accounts_host = (settings.zoho_accounts_host or "accounts.zoho.in").strip()
        self.api_base = (settings.zoho_api_base_url or "https://www.zohoapis.in/crm/v8").rstrip("/")
        self._client_id = settings.zoho_client_id.strip()
        self._client_secret = settings.zoho_client_secret.strip()
        self._refresh_token = settings.zoho_refresh_token.strip()
        # Optional short-lived token for quick testing (skips refresh entirely).
        self._access_token: str = settings.zoho_access_token.strip()
        self._expiry: float = time.time() + 3300 if self._access_token else 0.0
        self._http: Optional[httpx.AsyncClient] = None

    # ── configuration / lifecycle ────────────────────────────────────────
    @property
    def configured(self) -> bool:
        """True if we have enough credentials to talk to Zoho."""
        has_refresh = bool(self._client_id and self._client_secret and self._refresh_token)
        return has_refresh or bool(self._access_token)

    def status(self) -> dict:
        """Human-readable config status for the /api/health endpoint."""
        return {
            "configured": self.configured,
            "auth_mode": "refresh_token" if self._refresh_token else ("access_token" if self._access_token else "none"),
            "api_base": self.api_base,
            "accounts_host": self.accounts_host,
        }

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ── auth ─────────────────────────────────────────────────────────────
    async def _token(self) -> str:
        """Return a valid access token, refreshing via the refresh token if needed."""
        if self._access_token and time.time() < (self._expiry - 120):
            return self._access_token
        if not self._refresh_token:
            if self._access_token:
                # A manually-pasted token that may be expired — try it anyway.
                return self._access_token
            raise ZohoError(
                "Zoho is not configured. Set ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET and "
                "ZOHO_REFRESH_TOKEN (or ZOHO_ACCESS_TOKEN) in .env."
            )
        http = await self._get_http()
        url = f"https://{self.accounts_host}/oauth/v2/token"
        params = {
            "refresh_token": self._refresh_token,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "grant_type": "refresh_token",
        }
        resp = await http.post(url, params=params)
        data = resp.json() if resp.content else {}
        if resp.status_code != 200 or "access_token" not in data:
            raise ZohoError(f"Zoho token refresh failed: {data or resp.status_code}")
        self._access_token = data["access_token"]
        self._expiry = time.time() + int(data.get("expires_in", 3600))
        log.info("Refreshed Zoho access token (expires in %ss)", data.get("expires_in"))
        return self._access_token

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        http = await self._get_http()
        token = await self._token()
        headers = {"Authorization": f"Zoho-oauthtoken {token}", "Content-Type": "application/json"}
        url = f"{self.api_base}/{path.lstrip('/')}"
        resp = await http.request(method, url, headers=headers, **kwargs)

        if resp.status_code == 204:  # No Content (e.g. empty search)
            return {}
        if resp.status_code == 401:
            # Token might have just expired — force one refresh and retry once.
            self._expiry = 0.0
            token = await self._token()
            headers["Authorization"] = f"Zoho-oauthtoken {token}"
            resp = await http.request(method, url, headers=headers, **kwargs)
            if resp.status_code == 204:
                return {}
        if resp.status_code >= 400:
            detail = resp.text[:500]
            raise ZohoError(f"Zoho {method} {path} → {resp.status_code}: {detail}")
        return resp.json() if resp.content else {}

    # ── operations ───────────────────────────────────────────────────────
    @staticmethod
    def _norm_module(module: str) -> str:
        m = (module or "Leads").strip().capitalize()
        if m not in SUPPORTED_MODULES:
            # Accept singular / lowercase variants gracefully.
            aliases = {"Lead": "Leads", "Contact": "Contacts"}
            m = aliases.get(m, m)
        if m not in SUPPORTED_MODULES:
            raise ZohoError(f"Unsupported module '{module}'. Use one of {SUPPORTED_MODULES}.")
        return m

    async def search(self, module: str, query: str, limit: int = 10) -> list[dict]:
        """Free-text search a module by a word (name, email, company, phone…).

        Uses Zoho's ``/search?word=`` which matches across searchable fields.
        """
        module = self._norm_module(module)
        params = {"word": query, "per_page": min(limit, 50)}
        try:
            data = await self._request("GET", f"/{module}/search", params=params)
        except ZohoError as e:
            # A malformed query returns 400 — surface it as "no results" for the agent.
            if "→ 400" in str(e) or "→ 204" in str(e):
                return []
            raise
        return [self._slim(module, r) for r in (data.get("data") or [])][:limit]

    async def get(self, module: str, record_id: str) -> Optional[dict]:
        module = self._norm_module(module)
        data = await self._request(
            "GET", f"/{module}/{record_id}", params={"fields": _MODULE_FIELDS[module]}
        )
        recs = data.get("data") or []
        return self._slim(module, recs[0]) if recs else None

    async def list_recent(self, module: str, limit: int = 10) -> list[dict]:
        module = self._norm_module(module)
        params = {
            "fields": _MODULE_FIELDS[module],
            "per_page": min(limit, 50),
            "sort_by": "Modified_Time",
            "sort_order": "desc",
        }
        data = await self._request("GET", f"/{module}", params=params)
        return [self._slim(module, r) for r in (data.get("data") or [])]

    async def create_lead(self, fields: dict) -> dict:
        """Create a Lead. ``Last_Name`` and ``Company`` are required by Zoho."""
        payload = {k: v for k, v in fields.items() if v not in (None, "")}
        if not payload.get("Last_Name"):
            raise ZohoError("Creating a lead requires at least 'Last_Name'.")
        payload.setdefault("Company", payload.get("Company") or "Unknown")
        data = await self._request("POST", "/Leads", json={"data": [payload]})
        return self._creation_result("Leads", data)

    async def update_record(self, module: str, record_id: str, fields: dict) -> dict:
        """Update fields on an existing Lead or Contact."""
        module = self._norm_module(module)
        payload = {k: v for k, v in fields.items() if v is not None}
        if not payload:
            raise ZohoError("No fields provided to update.")
        data = await self._request(
            "PUT", f"/{module}/{record_id}", json={"data": [payload]}
        )
        return self._creation_result(module, data, record_id=record_id)

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _slim(module: str, rec: dict) -> dict:
        """Trim a raw Zoho record to the fields we care about (flattening lookups)."""
        wanted = _MODULE_FIELDS[module].split(",")
        out: dict[str, Any] = {}
        for f in wanted:
            v = rec.get(f)
            if isinstance(v, dict):  # lookup fields like Account_Name → {"name":..,"id":..}
                v = v.get("name")
            if v not in (None, ""):
                out[f] = v
        out["id"] = rec.get("id")
        out["_module"] = module
        return out

    @staticmethod
    def _creation_result(module: str, data: dict, record_id: str | None = None) -> dict:
        rows = data.get("data") or []
        first = rows[0] if rows else {}
        code = first.get("code")
        if code not in (None, "SUCCESS"):
            raise ZohoError(f"Zoho rejected the write: {first.get('message')} ({first})")
        details = first.get("details") or {}
        return {
            "success": True,
            "module": module,
            "id": details.get("id") or record_id,
            "message": first.get("message", "Record written."),
        }


# A single shared instance for the app.
crm = ZohoCRM()
