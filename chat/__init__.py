"""Text chatbot demo package.

A lightweight, OpenWebUI-style multi-agent chat app that reuses the project's
LLM (GLM-4.7-Flash) and the Zoho CRM integration. It is completely separate
from the realtime voice pipeline in ``server/``.

Modules:
  crm.py      — live Zoho CRM Leads/Contacts client (refresh-token auth)
  tools.py    — tool schemas + dispatch (CRM live; support/travel simulated)
  agents.py   — the three demo agents and their system prompts
  runtime.py  — the tool-calling agent loop (native OpenAI function calling)
  app.py      — FastAPI server (serves the UI + streaming chat API)
"""
