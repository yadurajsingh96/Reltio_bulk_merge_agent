#!/usr/bin/env python3
"""
Mock Reltio API Server for Integration Testing

Simulates the Reltio API endpoints used by the Bulk Merge Agent:
- POST /oauth/token         -> Returns mock OAuth token
- POST /entities/_search    -> Returns mock search results
- GET  /entities/{id}       -> Returns mock entity
- GET  /entities/{id}/_matches -> Returns mock matches
- POST /entities/{id}/_sameAs  -> Simulates merge

Usage:
    python mock_reltio_server.py               # Start on port 8888
    python mock_reltio_server.py --port 9999   # Custom port

Then set environment:
    export RELTIO_ENVIRONMENT=localhost:8888
    export RELTIO_CLIENT_ID=test
    export RELTIO_CLIENT_SECRET=test
    export RELTIO_TENANT_ID=test-tenant
"""

import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import uuid


# Mock data store
MOCK_ENTITIES = {
    "abc123": {
        "uri": "entities/abc123",
        "type": "configuration/entityTypes/HCP",
        "label": "John Smith",
        "attributes": {
            "NPI": [{"value": "1234567890", "ov": True}],
            "FirstName": [{"value": "John", "ov": True}],
            "LastName": [{"value": "Smith", "ov": True}],
            "Specialty": [{"value": "Cardiology", "ov": True}],
            "City": [{"value": "Boston", "ov": True}],
            "State": [{"value": "MA", "ov": True}],
        },
    },
    "def456": {
        "uri": "entities/def456",
        "type": "configuration/entityTypes/HCP",
        "label": "Jane Doe",
        "attributes": {
            "NPI": [{"value": "1234567891", "ov": True}],
            "FirstName": [{"value": "Jane", "ov": True}],
            "LastName": [{"value": "Doe", "ov": True}],
            "Specialty": [{"value": "Oncology", "ov": True}],
            "City": [{"value": "New York", "ov": True}],
            "State": [{"value": "NY", "ov": True}],
        },
    },
    "ghi789": {
        "uri": "entities/ghi789",
        "type": "configuration/entityTypes/HCP",
        "label": "Robert Johnson",
        "attributes": {
            "NPI": [{"value": "1234567892", "ov": True}],
            "FirstName": [{"value": "Robert", "ov": True}],
            "LastName": [{"value": "Johnson", "ov": True}],
            "Specialty": [{"value": "Neurology", "ov": True}],
            "City": [{"value": "Chicago", "ov": True}],
            "State": [{"value": "IL", "ov": True}],
        },
    },
}

# Merge log
MERGE_LOG = []


class MockReltioHandler(BaseHTTPRequestHandler):
    """Handles mock Reltio API requests."""

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length else ""

        # OAuth token
        if "/oauth/token" in path:
            self._respond(200, {
                "access_token": f"mock-token-{uuid.uuid4().hex[:8]}",
                "token_type": "bearer",
                "expires_in": 3600,
            })
            return

        # Entity search
        if "/_search" in path:
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                payload = {}

            filter_expr = payload.get("filter", "")
            results = []

            for eid, entity in MOCK_ENTITIES.items():
                # Simple keyword matching against filter
                flat_str = json.dumps(entity).lower()
                if any(kw.lower() in flat_str for kw in filter_expr.replace("'", "").split()):
                    results.append(entity)

            if not results:
                results = list(MOCK_ENTITIES.values())[:int(payload.get("max", 5))]

            self._respond(200, results)
            return

        # Merge (sameAs)
        if "/_sameAs" in path:
            params = parse_qs(parsed.query)
            loser_uri = params.get("uri", ["unknown"])[0]
            winner_id = path.split("/entities/")[1].split("/")[0] if "/entities/" in path else "unknown"

            merge_record = {
                "winner": f"entities/{winner_id}",
                "loser": loser_uri,
                "status": "merged",
                "timestamp": "2026-01-01T00:00:00Z",
            }
            MERGE_LOG.append(merge_record)

            self._respond(200, {
                "uri": f"entities/{winner_id}",
                "status": "merged",
                "mergedEntities": [loser_uri],
            })
            return

        self._respond(404, {"error": f"Unknown endpoint: {path}"})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Get entity by ID
        if "/entities/" in path and "/_" not in path:
            entity_id = path.split("/entities/")[-1].strip("/")
            entity = MOCK_ENTITIES.get(entity_id)

            if entity:
                self._respond(200, entity)
            else:
                self._respond(404, {"error": f"Entity not found: {entity_id}"})
            return

        # Get matches
        if "/_matches" in path:
            entity_id = path.split("/entities/")[1].split("/")[0]
            entity = MOCK_ENTITIES.get(entity_id)

            if entity:
                matches = [
                    {**e, "matchScore": 85.0}
                    for eid, e in MOCK_ENTITIES.items()
                    if eid != entity_id
                ][:3]
                self._respond(200, matches)
            else:
                self._respond(200, [])
            return

        # Merge log (extra endpoint for testing)
        if "/merge-log" in path:
            self._respond(200, MERGE_LOG)
            return

        # Health
        if "/health" in path or path == "/":
            self._respond(200, {"status": "ok", "mock": True, "entities": len(MOCK_ENTITIES)})
            return

        self._respond(404, {"error": f"Unknown endpoint: {path}"})

    def _respond(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        print(f"  [MOCK] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="Mock Reltio API Server")
    parser.add_argument("--port", type=int, default=8888, help="Port to listen on")
    args = parser.parse_args()

    server = HTTPServer(("", args.port), MockReltioHandler)
    print(f"\n  Mock Reltio API Server running on http://localhost:{args.port}")
    print(f"  Serving {len(MOCK_ENTITIES)} mock entities")
    print(f"  Press Ctrl+C to stop\n")
    print(f"  To use with Bulk Merge Agent:")
    print(f"    export RELTIO_ENVIRONMENT=localhost:{args.port}")
    print(f"    export RELTIO_CLIENT_ID=test")
    print(f"    export RELTIO_CLIENT_SECRET=test")
    print(f"    export RELTIO_TENANT_ID=test-tenant\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Mock server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
