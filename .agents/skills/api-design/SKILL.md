---
name: api-design
description: "Industry-standard RESTful API design guidelines: resource naming, HTTP semantics, structured error taxonomy (Stripe-style and RFC 7807), cursor pagination, idempotency keys, and rate-limiting resilience."
license: MIT
metadata:
  origin: "Best-of-market REST standards (Stripe, Zalando, Google API Design Guide, RFC 7807 / RFC 9457, RFC 9110)"
---

# Enterprise RESTful API Design Guidelines

This skill guides coding agents to design consistent, resilient, and developer-friendly REST APIs. It establishes strict architectural conventions across resource modeling, status codes, response envelopes, error handling, idempotency, and versioning.

---

## 1. Resource Modeling & URL Hierarchy

### 🔗 URL Conventions
- **Nouns over verbs**: Resources represent entities, not remote procedure calls.
- **Plural, lowercase, kebab-case**: `/api/v1/team-members`, `/api/v1/payment-intents`.
- **Hierarchical nesting for ownership**:
  - `GET /api/v1/organizations/:orgId/projects`
  - `GET /api/v1/projects/:projectId/tasks/:taskId`
  - *Rule*: Limit nesting to 2 levels maximum (`/parents/:id/children/:id`). For deeper relationships, use top-level resources with query filters: `/api/v1/comments?task_id=:taskId`.
- **Non-CRUD Actions**: When an operation does not map cleanly to standard CRUD, use explicit sub-resource action verbs sparingly:
  - `POST /api/v1/orders/:id/cancel`
  - `POST /api/v1/auth/refresh-token`
  - `POST /api/v1/reports/export`

### 🚫 URL Anti-Patterns
| Anti-Pattern | Bad Example | Correct Design |
| :--- | :--- | :--- |
| Verbs in URLs | `GET /api/v1/getUsers` | `GET /api/v1/users` |
| Method in URLs | `POST /api/v1/users/delete` | `DELETE /api/v1/users/:id` |
| Singular resource | `/api/v1/user/:id` | `/api/v1/users/:id` |
| Mixed casing | `/api/v1/teamMembers` or `team_members` | `/api/v1/team-members` |
| Query params as path | `/api/v1/orders/status/active` | `/api/v1/orders?status=active` |

---

## 2. HTTP Method Semantics & Status Codes

### 📋 Method Contract Matrix
| Method | Idempotent | Safe | Typical Response Code | Semantics |
| :--- | :---: | :---: | :---: | :--- |
| **GET** | Yes | Yes | `200 OK` | Retrieve resource representation without side effects. |
| **POST** | No | No | `201 Created` (with `Location`) | Create a new subordinate resource or trigger a process. |
| **PUT** | Yes | No | `200 OK` / `204 No Content` | Complete replacement of the identified resource. |
| **PATCH** | No (RFC 9110) | No | `200 OK` / `204 No Content` | Partial modification of specific resource attributes (can be made idempotent with JSON Patch or conditional headers). |
| **DELETE** | Yes | No | `204 No Content` | Remove resource; subsequent calls return `404` or `204`. |

### 🎯 Status Code Invariants
- **200 OK**: Request succeeded with response body.
- **201 Created**: Resource created; MUST include `Location: /api/v1/resources/:id` header.
- **204 No Content**: Action succeeded with zero response body (common for `DELETE`).
- **400 Bad Request**: Malformed JSON syntax, invalid query parameters.
- **401 Unauthorized**: Missing, expired, or invalid authentication credentials.
- **403 Forbidden**: Authenticated caller lacks permissions/scopes for this resource.
- **404 Not Found**: Resource does not exist (or hidden for security).
- **409 Conflict**: Duplicate key, concurrent state modification, or invariant violation.
- **422 Unprocessable Entity**: Semantically invalid payload (schema valid, but business validation failed).
- **429 Too Many Requests**: Rate limit exceeded; MUST include `Retry-After` header.
- **500 Internal Server Error**: Unexpected unhandled failure; NEVER expose stack traces or DB credentials.
- **502 Bad Gateway / 503 Service Unavailable**: Downstream dependency failure or maintenance mode.

> [!CAUTION]
> **Zero Tolerance for "200 OK Errors"**: NEVER return HTTP 200 with `{ "status": "error", "message": "..." }`. The HTTP status code is the protocol-level source of truth.

---

## 3. Standard Response Envelopes & Error Contracts

### 📦 Success Responses
```typescript
// Single Resource Response:
interface ItemResponse<T> {
  data: T;
}

// Collection Response:
interface CollectionResponse<T> {
  data: T[];
  meta: {
    total?: number;       // Omitted for infinite scroll cursor queries
    has_more: boolean;
    next_cursor?: string;
  };
}
```

### 🚨 Standardized Error Envelopes

Choose one consistent error representation across the API:

#### Option A: Stripe-Style Error Envelope (Recommended for Developer-Facing Web APIs)
```json
{
  "error": {
    "code": "validation_failed",
    "message": "The request payload failed field validation",
    "details": [
      {
        "field": "email",
        "code": "invalid_format",
        "message": "Must be a valid RFC 5322 email address"
      },
      {
        "field": "amount_cents",
        "code": "out_of_bounds",
        "message": "Amount must be greater than 0"
      }
    ],
    "request_id": "req_01h8xyz789"
  }
}
```

#### Option B: RFC 7807 Problem Details (`application/problem+json`)
IETF standard format specifying machine-readable problem details for HTTP APIs:
```json
{
  "type": "https://api.example.com/errors/validation-failed",
  "title": "Validation Failed",
  "status": 422,
  "detail": "The request payload failed validation on 2 fields",
  "instance": "/api/v1/payment-intents",
  "invalid_params": [
    {
      "name": "email",
      "reason": "Must be a valid RFC 5322 email address"
    },
    {
      "name": "amount_cents",
      "reason": "Amount must be greater than 0"
    }
  ]
}
```

---

## 4. Pagination Strategies

### 1. Cursor-Based Pagination (Recommended for High Volume / Real-Time Data)
- **Request**: `GET /api/v1/events?limit=25&starting_after=evt_123` (or `cursor=eyJpZCI6MTIzfQ`)
- **Response**:
  ```json
  {
    "data": [ ... ],
    "meta": {
      "has_more": true,
      "next_cursor": "evt_148"
    }
  }
  ```
- **Advantages**: Stable under concurrent inserts/deletes, consistent performance (O(1) indexed lookup), no duplicate records.

### 2. Offset-Based Pagination (Acceptable for Internal Admin Tables)
- **Request**: `GET /api/v1/users?page=2&per_page=50`
- **Response**:
  ```json
  {
    "data": [ ... ],
    "meta": {
      "total": 450,
      "page": 2,
      "per_page": 50,
      "total_pages": 9
    }
  }
  ```

---

## 5. Filtering, Sorting, and Field Projections

- **Filtering**: Use explicit query parameters:
  `GET /api/v1/orders?status=shipped&created_after=2026-01-01T00:00:00Z`
- **Sorting**: Multi-field comma-separated sorting with `-` prefix for descending order:
  `GET /api/v1/products?sort=-price,created_at`
- **Sparse Fieldsets / Projections**:
  `GET /api/v1/users?fields=id,name,email`

---

## 6. Reliability & Production Hardening

### 🔒 Idempotency Keys (Stripe Pattern)
For mutating operations (financial transactions, webhook dispatches, state transitions):
- Client sends header: `Idempotency-Key: <unique-uuid>`.
- Server checks distributed cache (Redis / DB) with 24h TTL.
- If request is currently processing: return `409 Conflict` or wait.
- If already completed: replay cached response code and body with `Idempotent-Replay: true`.

### ⏱️ Rate Limiting Headers
Every API response should supply rate-limiting telemetry:
- `X-RateLimit-Limit`: Maximum requests permitted per window.
- `X-RateLimit-Remaining`: Remaining request quota.
- `X-RateLimit-Reset`: UTC epoch timestamp when current window resets.
- `Retry-After`: Seconds to wait before retrying (on `429 Too Many Requests`).

### 📦 Upstream Service Invariants
- **Connection pooling & singleton clients**: Never re-create API client instances per request.
- **Timeout budgets**: Hard timeouts on all outbound calls (e.g., 3000ms connect, 5000ms read).
- **Graceful degradation**: Fall back to stale cache or return candidate list on ambiguous inputs; never crash the caller.
