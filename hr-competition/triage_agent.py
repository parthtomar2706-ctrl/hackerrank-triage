"""
================================================================================
  HACKATHON TRIAGE AGENT — HackerRank / Claude / Visa Support Ticket Triager
================================================================================

HOW TO RUN (beginner-friendly):
---------------------------------
1. Make sure you have Python 3.8+ installed. Check with:
       python --version

2. Install the required libraries (only needed once):
       pip install pandas anthropic

3. Place this script in the same folder as your 'support_issues.csv' file.
   Your CSV must have at least these columns:
       ticket_id, company, issue_text
   (Any additional columns will be preserved in the output.)

4. Set your Anthropic API key as an environment variable:
       # macOS / Linux:
       export ANTHROPIC_API_KEY="sk-ant-..."

       # Windows (Command Prompt):
       set ANTHROPIC_API_KEY=sk-ant-...

       # Windows (PowerShell):
       $env:ANTHROPIC_API_KEY="sk-ant-..."

5. Run the script:
       python triage_agent.py

6. Results will be written to 'output_results.csv' in the same folder,
   and a summary will be printed in the terminal.

OUTPUT SCHEMA (output_results.csv):
-------------------------------------
  ticket_id      — original ticket identifier
  company        — inferred or original company name
  issue_text     — original issue description
  status         — 'replied' or 'escalated'
  product_area   — logical grouping (e.g., "Account Security")
  request_type   — one of: product_issue | feature_request | bug | invalid
  justification  — short sentence explaining the triage decision

================================================================================
"""

import os
import json
import re
import time
import pandas as pd
import anthropic

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Supported companies and their keyword signals for inference
COMPANY_KEYWORDS: dict[str, list[str]] = {
    "HackerRank": [
        "coding", "challenge", "contest", "hackerrank", "assessment",
        "leaderboard", "submission", "compiler", "test case", "hire",
        "interview", "developer skill", "code editor", "certification",
    ],
    "Claude": [
        "claude", "anthropic", "llm", "language model", "prompt", "chat",
        "api key", "token", "context window", "hallucination", "ai assistant",
        "model", "inference", "generation", "fine-tun",
    ],
    "Visa": [
        "visa", "transaction", "payment", "card", "charge", "fraud",
        "chargeback", "merchant", "decline", "authorization", "billing",
        "credit", "debit", "purchase", "refund", "statement", "atm",
    ],
}

# Escalation triggers — any issue touching these themes gets escalated
ESCALATION_KEYWORDS: list[str] = [
    "fraud", "unauthorized", "billing", "charge", "chargeback",
    "account hacked", "security breach", "data breach", "stolen",
    "cannot login", "locked out", "critical bug", "production down",
    "outage", "data loss", "legal", "lawsuit", "regulatory",
]

# Claude model to use for LLM-powered triage
MODEL = "claude-sonnet-4-20250514"

# Retry settings for API calls
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


# ──────────────────────────────────────────────────────────────────────────────
# TRIAGE AGENT CLASS
# ──────────────────────────────────────────────────────────────────────────────

class TriageAgent:
    """
    An AI-powered triage agent that classifies support tickets for
    HackerRank, Claude, and Visa using the Anthropic Claude API.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialise the agent and Anthropic client.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        """
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.stats = {"processed": 0, "escalated": 0, "invalid": 0, "errors": 0}

    # ── Company Inference ──────────────────────────────────────────────────────

    def infer_company(self, issue_text: str) -> str:
        """
        Infer the company from keyword signals in the issue text.

        Scoring approach: count matching keywords per company and pick the
        highest scorer. Falls back to 'Unknown' if nothing matches.

        Args:
            issue_text: Raw issue description string.

        Returns:
            One of 'HackerRank', 'Claude', 'Visa', or 'Unknown'.
        """
        text_lower = issue_text.lower()
        scores: dict[str, int] = {company: 0 for company in COMPANY_KEYWORDS}

        for company, keywords in COMPANY_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    scores[company] += 1

        best_company = max(scores, key=lambda c: scores[c])
        return best_company if scores[best_company] > 0 else "Unknown"

    # ── Quick Pre-filter (no LLM needed) ──────────────────────────────────────

    def _should_escalate_heuristic(self, issue_text: str) -> bool:
        """Return True if issue_text contains any escalation trigger keywords."""
        text_lower = issue_text.lower()
        return any(kw in text_lower for kw in ESCALATION_KEYWORDS)

    # ── LLM Triage Call ───────────────────────────────────────────────────────

    def _build_prompt(self, ticket_id: str, company: str, issue_text: str) -> str:
        """Construct the triage prompt for the LLM."""
        return f"""You are a senior support triage specialist for three products:
- **HackerRank** — a developer skills and coding challenge platform
- **Claude** — Anthropic's AI assistant and API platform
- **Visa** — a global payment network and card issuer

Your job is to classify a single support ticket and return ONLY a JSON object.

---

TICKET:
  ID      : {ticket_id}
  Company : {company}
  Issue   : {issue_text}

---

RULES:
1. status:
   - "escalated" → high-risk issues: fraud, billing disputes, account lockouts,
     production outages, data breaches, critical bugs, or legal/regulatory matters.
   - "replied"   → everything else that has a clear, safe answer.

2. product_area: A concise 2–4 word label grouping the issue logically.
   Examples: "Account Security", "API Support", "Payment Processing",
   "Code Submission", "Billing & Refunds", "Model Performance", "Contest Access".

3. request_type: EXACTLY one of:
   - product_issue   → existing feature not working as expected
   - feature_request → user wants a new capability
   - bug             → confirmed or suspected software defect
   - invalid         → unrelated to HackerRank, Claude, or Visa ecosystems

4. justification: One sentence (max 20 words) explaining your triage decision.

5. If the issue is unrelated to all three companies, set:
   request_type = "invalid", status = "replied",
   justification = "Out of scope."

---

Respond with ONLY this JSON (no markdown, no extra text):
{{
  "status": "<replied|escalated>",
  "product_area": "<label>",
  "request_type": "<product_issue|feature_request|bug|invalid>",
  "justification": "<one sentence>"
}}"""

    def _call_llm(self, ticket_id: str, company: str, issue_text: str) -> dict:
        """
        Call the Claude API to triage one ticket, with retry logic.

        Returns a dict with keys: status, product_area, request_type, justification.
        """
        prompt = self._build_prompt(ticket_id, company, issue_text)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                message = self.client.messages.create(
                    model=MODEL,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = message.content[0].text.strip()

                # Strip accidental markdown fences
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw).strip()

                result = json.loads(raw)

                # Validate request_type
                valid_types = {"product_issue", "feature_request", "bug", "invalid"}
                if result.get("request_type") not in valid_types:
                    result["request_type"] = "product_issue"

                # Validate status
                if result.get("status") not in {"replied", "escalated"}:
                    result["status"] = "replied"

                # Override with heuristic escalation if LLM missed it
                if self._should_escalate_heuristic(issue_text):
                    result["status"] = "escalated"

                return result

            except json.JSONDecodeError as e:
                print(f"  [WARN] Ticket {ticket_id}: JSON parse error on attempt {attempt}: {e}")
            except anthropic.APIError as e:
                print(f"  [WARN] Ticket {ticket_id}: API error on attempt {attempt}: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

        # Fallback if all retries failed
        self.stats["errors"] += 1
        return {
            "status": "escalated",
            "product_area": "Triage Error",
            "request_type": "product_issue",
            "justification": "Triage failed after retries; escalated for manual review.",
        }

    # ── Main Processing Pipeline ───────────────────────────────────────────────

    def process_ticket(self, row: pd.Series) -> dict:
        """
        Process a single ticket row and return the enriched triage result.

        Args:
            row: A pandas Series representing one CSV row.

        Returns:
            Dict with all original fields plus triage fields.
        """
        ticket_id = str(row.get("ticket_id", "UNKNOWN"))
        issue_text = str(row.get("issue_text", "")).strip()
        company = str(row.get("company", "")).strip()

        # ── Step 1: Company Inference ──────────────────────────────────────────
        if not company or company.lower() in ("none", "nan", ""):
            company = self.infer_company(issue_text)
            print(f"  [INFO] Ticket {ticket_id}: Company inferred → {company}")

        # ── Step 2: LLM Triage ────────────────────────────────────────────────
        triage = self._call_llm(ticket_id, company, issue_text)

        # ── Step 3: Update stats ───────────────────────────────────────────────
        self.stats["processed"] += 1
        if triage["status"] == "escalated":
            self.stats["escalated"] += 1
        if triage["request_type"] == "invalid":
            self.stats["invalid"] += 1

        return {
            **row.to_dict(),          # preserve all original columns
            "company": company,       # may have been inferred
            "status": triage["status"],
            "product_area": triage["product_area"],
            "request_type": triage["request_type"],
            "justification": triage["justification"],
        }

    def process_csv(self, input_path: str, output_path: str) -> None:
        """
        Read input CSV, triage all tickets, and write output CSV.

        Args:
            input_path:  Path to 'support_issues.csv'.
            output_path: Path to write 'output_results.csv'.
        """
        # ── Load CSV ───────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("  TRIAGE AGENT — Starting")
        print(f"{'='*60}")
        print(f"  Input  : {input_path}")
        print(f"  Output : {output_path}")
        print(f"  Model  : {MODEL}\n")

        try:
            df = pd.read_csv(input_path)
        except FileNotFoundError:
            print(f"[ERROR] File not found: {input_path}")
            print("  Make sure 'support_issues.csv' is in the same directory.")
            return
        except Exception as e:
            print(f"[ERROR] Could not read CSV: {e}")
            return

        total = len(df)
        print(f"  Loaded {total} ticket(s) from CSV.\n")

        # ── Process each row ───────────────────────────────────────────────────
        results = []
        for idx, row in df.iterrows():
            ticket_id = row.get("ticket_id", idx + 1)
            print(f"  Processing ticket {ticket_id} ({idx + 1}/{total})…")
            result = self.process_ticket(row)
            results.append(result)

        # ── Write output ───────────────────────────────────────────────────────
        output_df = pd.DataFrame(results)

        # Enforce column order — triage columns come after originals
        triage_cols = ["status", "product_area", "request_type", "justification"]
        other_cols = [c for c in output_df.columns if c not in triage_cols]
        output_df = output_df[other_cols + triage_cols]

        output_df.to_csv(output_path, index=False)

        # ── Terminal Summary ───────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("  TRIAGE COMPLETE — Summary")
        print(f"{'='*60}")
        print(f"  Tickets processed : {self.stats['processed']}")
        print(f"  Escalated         : {self.stats['escalated']}")
        print(f"  Invalid / OOS     : {self.stats['invalid']}")
        print(f"  API errors        : {self.stats['errors']}")
        print(f"  Output written to : {output_path}")
        print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_CSV = "support_issues.csv"
    OUTPUT_CSV = "output_results.csv"

    agent = TriageAgent()
    agent.process_csv(INPUT_CSV, OUTPUT_CSV)