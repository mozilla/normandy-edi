import json

import click

from .cache import cache
from .log import log


TRACE = 5


META_SERVER_KEY = __name__ + ".server"

SERVERS = {
    "prod": {
        "public": "https://normandy.cdn.mozilla.net",
        "admin": "https://prod-admin.normandy.prod.cloudops.mozgcp.net",
    },
    "stage": {
        "public": "https://stage.normandy.nonprod.cloudops.mozgcp.net",
        "admin": "https://stage-admin.normandy.nonprod.cloudops.mozgcp.net",
    },
    "dev": {
        "public": "https://dev.normandy.nonprod.cloudops.mozgcp.net",
        "admin": "https://dev-admin.normandy.nonprod.cloudops.mozgcp.net",
    },
    "prod-admin": {
        "public": "https://prod-admin.normandy.prod.cloudops.mozgcp.net",
        "admin": "https://prod-admin.normandy.prod.cloudops.mozgcp.net",
    },
    "stage-admin": {
        "public": "https://stage-admin.normandy.nonprod.cloudops.mozgcp.net",
        "admin": "https://stage-admin.normandy.nonprod.cloudops.mozgcp.net",
    },
    "dev-admin": {
        "public": "https://dev-ademin.normandy.nonprod.cloudops.mozgcp.net",
        "admin": "https://dev-admin.normandy.nonprod.cloudops.mozgcp.net",
    },
    "local": {"public": "https://localhost:8000", "admin": "https://localhost:8000"},
    "requestbin": {
        "public": "http://requestbin.net/r/sb02oasb",
        "admin": "http://requestbin.net/r/sb02oasb",
    },
}


async def api_request(
    session, verb, endpoint, *, opts=None, data=None, version=3, admin=False, headers=None
):
    """
    :param session: The asyncio session to work in
    :param verb:  GET, POST, etc.
    :param endpoint: The address to query, either only an API path
        ("recipe/42/history") or a full url (https://example.com/...).
    :opts: Optional. Data to include in the query string. Cannot be used if
        endpoint includes a protocol.
    :param data: Optional. Data to pass in the request. Not allowed for GET or HEAD.
    :param version: The version of the API to query.
    :param admin: Whether an admin server is required.
    """
    if endpoint.startswith("http"):
        if opts is not None:
            raise click.ClickException("Can't pass opts with a full url")
        url = endpoint
    else:
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]
        if endpoint.endswith("/"):
            endpoint = endpoint[:-1]
        if "?" in endpoint:
            raise click.ClickException("Don't include query string in endpoint")
        qs = "?" + "&".join(f"{k}={v}" for k, v in opts.items()) if opts else ""

        server_url = click.get_current_context().meta[META_SERVER_KEY][
            "admin" if admin else "public"
        ]
        url = f"{server_url}/api/v{version}/{endpoint}/{qs}"

    if verb in ["GET", "HEAD"] and data is not None:
        raise Exception("Can't include a body for {verb} requests")

    if data:
        log.log(TRACE, f"{verb} {url} {data}")
    else:
        log.log(TRACE, f"{verb} {url}")

    resp_data = None
    cache_key = None
    if cache and verb in ["GET", "HEAD"]:
        cache_key = f"api_request:{verb}:{url}"
        resp_data = cache.get(cache_key)
        if resp_data is not None:
            log.log(TRACE, f"Cache HIT  {cache_key}")
        else:
            log.log(TRACE, f"Cache MISS {cache_key}")

    if resp_data is None:
        verb_method = getattr(session, verb.lower())
        async with verb_method(url, json=data, headers=headers) as resp:
            resp_data = {"status": resp.status, "text": await resp.text()}
            if verb in ["GET", "HEAD"] and cache and cache_key:
                # TODO use real cache timeout / ignore uncacheable
                cache.set(cache_key, resp_data, timeout=300)

    assert resp_data is not None

    if resp_data["status"] >= 400:
        raise click.ClickException(f"HTTP Error Code {resp_data['status']}: {resp_data['text']}")
    if resp_data["status"] == 204:
        return None
    try:
        return json.loads(resp_data["text"])
    except json.decoder.JSONDecodeError:
        log.error(resp_data["text"])
        raise


async def api_fetch(session, endpoint, opts=None, *, version=3, admin=False, headers=None):
    return await api_request(
        session, "GET", endpoint, opts=opts, version=version, admin=admin, headers=headers
    )
