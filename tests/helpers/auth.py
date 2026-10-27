"""Helpers for API tests that need a signed-in client."""

import httpx

PASSWORD = "correct horse battery"


async def signup(
    client: httpx.AsyncClient,
    email: str = "alice@example.com",
    password: str = PASSWORD,
    display_name: str = "Alice",
) -> dict:
    """Register through the API; the client keeps the session cookies. Returns SessionOut."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "display_name": display_name},
    )
    assert response.status_code == 201, response.text
    return response.json()


def csrf(session: dict) -> dict[str, str]:
    return {"X-CSRF-Token": session["csrf_token"]}
