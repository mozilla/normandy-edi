import asyncio
import functools
import json
import logging
import re
import sys
from collections import Counter

import aiohttp
import click
import iso8601
import pyjq
from tqdm import tqdm
from pypeln import asyncio_task as pipeline


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
}


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # noqa
            self.handleError(record)


log = logging.getLogger(__name__)
log.addHandler(TqdmLoggingHandler())
log.setLevel(logging.INFO)


def compose(*functions):
    def compose2(f, g):
        return lambda x: f(g(x))

    return functools.reduce(compose2, functions, lambda x: x)


async def api_request(
    session, verb, endpoint, *, opts=None, data=None, version=3, admin=False
):
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
        log.debug(f"{verb} {url} {data}")
    else:
        log.debug(f"{verb} {url}")
    verb_method = getattr(session, verb.lower())
    async with verb_method(url, data=data) as resp:
        if resp.status >= 400:
            raise click.ClickException(
                f"HTTP Error Code {resp.status}: {await resp.text()}"
            )
        if resp.status == 204:
            return None
        try:
            return json.loads(await resp.text())
        except json.decoder.JSONDecodeError:
            log.debug(await resp.text())
            raise


async def api_fetch(session, endpoint, opts=None, *, version=3, admin=False):
    return await api_request(
        session, "GET", endpoint, opts=opts, version=version, admin=admin
    )


class Iso8601Param(click.ParamType):
    def convert(self, value, param, ctx):
        try:
            return iso8601.parse_date(value)
        except iso8601.ParseError as e:
            self.fail(str(e), param, ctx)


def with_session(async_func):
    @click_compatible_wraps(async_func)
    async def inner(*args, **kwargs):
        async with aiohttp.ClientSession() as session:
            return await async_func(*args, session=session, **kwargs)

    return inner


def with_authed_session(async_func):
    @click.option("-A", "--auth", required=True, allow_from_autoenv=True, prompt=True)
    @click_compatible_wraps(async_func)
    async def inner(auth, *args, **kwargs):
        if not auth.lower().startswith("bearer "):
            auth = "Bearer " + auth
        headers = {"Authorization": auth}
        async with aiohttp.ClientSession(headers=headers) as authed_session:
            return await async_func(*args, authed_session=authed_session, **kwargs)

    return inner


def click_compatible_wraps(wrapped_func):
    assignments = [*functools.WRAPPER_ASSIGNMENTS, "__click_params__"]
    return functools.wraps(wrapped_func, assignments)


def filter_options(func):
    @click.option("--text", "-t")
    @click.option("--enabled", "-e", is_flag=True, default=None)
    @click.option("--action", "-a")
    @click.option("--creator", "-c")
    @click_compatible_wraps(func)
    def filter_parser(text, enabled, action, creator, *args, **kwargs):
        filters = {
            "text": text,
            "enabled": enabled,
            "action": action,
            "creator": creator,
        }
        return func(*args, filters=filters, **kwargs)

    return filter_parser


def logging_options(func):
    @click.option("--verbose", "-v", is_flag=True)
    @click_compatible_wraps(func)
    def logging_wrapper(*args, verbose=False, **kwargs):
        if verbose:
            log.setLevel(logging.DEBUG)
        return func(*args, **kwargs)

    return logging_wrapper


def server_options(func):
    @click.option(
        "--server",
        "-s",
        type=click.Choice(
            ["prod", "stage", "dev", "local", "prod-admin", "stage-admin", "dev-admin"]
        ),
        default="prod",
    )
    @click.pass_context
    @click_compatible_wraps(func)
    def logging_wrapper(context, *args, server, verbose=False, **kwargs):
        server_url = SERVERS[server]
        context.meta[META_SERVER_KEY] = server_url
        return func(*args, **kwargs)

    return logging_wrapper


@click.group()
@logging_options
@server_options
def cli():
    pass


def async_trampoline(async_func):
    @click_compatible_wraps(async_func)
    def sync_func(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        loop.run_until_complete(async_func(*args, **kwargs))

    return sync_func


async def paginated_fetch(
    session, endpoint, *, query=None, admin=False, desc="fetch", limit=None
):
    count = 0
    with tqdm(desc=desc, total=limit) as progress_bar:
        data = await api_fetch(session, endpoint, query, admin=admin)
        if limit is None:
            progress_bar.total = data["count"]
        next_url = data["next"]
        for v in data["results"]:
            yield v
            count += 1
            progress_bar.update(1)
            if limit is not None and count >= limit:
                return

        while next_url:
            data = await api_fetch(session, next_url, admin=admin)
            if limit is None:
                progress_bar.total = data["count"]
            next_url = data["next"]
            for v in data["results"]:
                yield v
                count += 1
                progress_bar.update(1)
                if limit is not None and count >= limit:
                    return


async def fetch_recipes(
    session, *, text=None, enabled=None, action=None, creator=None, limit=None
):
    query = {}
    if text is not None:
        query["text"] = text
    if enabled is not None:
        query["enabled"] = enabled
    if action is not None:
        query["action"] = action

    def match_creator(recipe):
        if creator is None:
            return True
        else:
            approved_creator = (
                recipe.get("latest_revision", {}).get("creator", {}) or {}
            ).get("email", "")
            latest_creator = (
                recipe.get("latest_revision", {}).get("creator", {}) or {}
            ).get("email", "")
            return creator not in approved_creator and creator not in latest_creator

    async for recipe in paginated_fetch(
        session, "recipe", query=query, desc="fetch recipes", limit=limit
    ):
        if match_creator(recipe):
            yield recipe


@cli.command("filter-inputs")
@filter_options
@logging_options
@server_options
@async_trampoline
@with_session
async def filter_inputs(session, filters):
    """find all filter inputs"""
    all_inputs = Counter()
    input_regex = re.compile(r"\[([^]]*)\]\|(stable|bucket)Sample", re.MULTILINE)

    async for recipe in fetch_recipes(session, **filters):
        exp = recipe["latest_revision"]["filter_expression"]
        for match in input_regex.finditer(exp):
            for input in (inp.strip() for inp in match.group(1).split(", ")):
                all_inputs[input] += 1

    for (input, count) in all_inputs.most_common():
        log.info(f"{count:>3} {input}")


@cli.command("enabled-range")
@filter_options
@logging_options
@server_options
@click.option("--begin", "-b", required=True, type=Iso8601Param(), prompt=True)
@click.option("--end", "-e", required=True, type=Iso8601Param(), prompt=True)
@async_trampoline
@with_session
async def enabled_range(session, filters, begin=None, end=None):
    """Show recipes enabled during a time range"""

    async def potentially_enabled(recipe):
        rev = recipe["latest_revision"]
        if iso8601.parse_date(rev["updated"]) < begin and not rev["enabled"]:
            return False
        return True

    async def fetch_history(recipe):
        history = await api_fetch(session, f"/recipe/{recipe['id']}/history/")
        return history

    histories = (
        fetch_recipes(session, **filters)
        | pipeline.filter(potentially_enabled)
        | pipeline.map(fetch_history)
    )

    _histories = []
    async for h in histories:
        _histories.append(h)
    histories = _histories

    log.info(f"{len(histories)} recipes")


@cli.command("heartbeat-url-scan")
@filter_options
@logging_options
@server_options
@async_trampoline
@with_session
async def heartbeat_url_scan(session, filters):
    """Print all URLs used in heartbeat recipes"""
    if filters.get("action") not in ["show-heartbeat", None]:
        log.error(f"Can only run on action=show-heartbeat, not {filters.get('action')}")
        sys.exit(1)

    filters["action"] = "show-heartbeat"

    def get_urls(recipe):
        log.debug(f"processing recipe {recipe['id']}")

        for key in ["latest_revision", "approved_revision"]:
            if not recipe.get(key):
                continue
            args = recipe[key]["arguments"]
            if args.get("learnMoreUrl"):
                yield args["learnMoreUrl"]
            if args.get("postAnswerUrl"):
                yield args["postAnswerUrl"]

    urls = set()
    async for recipe in fetch_recipes(session, **filters):
        for url in get_urls(recipe):
            if url not in urls:
                log.info(url)
                urls.add(url)


@cli.command("slugs")
@filter_options
@logging_options
@server_options
@async_trampoline
@with_session
async def slugs(session, filters):
    async for recipe in fetch_recipes(session, **filters):
        rev = recipe["latest_revision"]
        slug = rev["arguments"].get("slug") or rev["arguments"].get("name")
        if slug:
            log.info(slug)


@cli.command("groups")
@logging_options
@server_options
@async_trampoline
@with_authed_session
async def groups(authed_session):
    async for group in paginated_fetch(
        authed_session, "group", admin=True, desc="fetch groups"
    ):
        log.info(group)


@cli.command("users")
@logging_options
@server_options
@async_trampoline
@with_authed_session
async def users(authed_session):
    async for user in paginated_fetch(
        authed_session, "user", admin=True, desc="fetch users"
    ):
        groups = [g["name"] for g in user["groups"]]
        log.info(f"{user['id']} - {user['email']} - [{', '.join(groups)}]")


@cli.command("whoami")
@logging_options
@server_options
@async_trampoline
@with_authed_session
async def whoami(authed_session):
    log.info(await api_fetch(authed_session, "user/me", admin=True, version=1))


@cli.command("add-users-to-group")
@logging_options
@server_options
@click.option("--user-id", "-u", required=True, prompt=True, multiple=True)
@click.option("--group-id", "-g", required=True, prompt=True)
@async_trampoline
@with_authed_session
async def add_users_to_group(authed_session, user_id, group_id):
    for uid in user_id:
        await api_request(
            authed_session,
            "POST",
            f"group/{group_id}/add_user",
            data={"user_id": uid},
            admin=True,
        )


@cli.command("add-user")
@logging_options
@server_options
@click.argument("email", required=True)
@click.argument("first_name", required=True)
@click.argument("last_name", required=True)
@async_trampoline
@with_authed_session
async def add_user(authed_session, email, first_name, last_name):
    log.info(
        await api_request(
            authed_session,
            "POST",
            "user",
            data={"email": email, "first_name": first_name, "last_name": last_name},
            admin=True,
        )
    )


@cli.group()
@logging_options
@server_options
def recipes():
    pass


@recipes.command("all")
@filter_options
@logging_options
@server_options
@click.option("--jq-query", "-j")
@click.option("--limit", "-l", type=int)
@async_trampoline
@with_session
async def all_recipes(session, filters, jq_query, limit):
    if jq_query:
        compiled_query = pyjq.compile(jq_query)

    recipes = []
    async for recipe in fetch_recipes(session, **filters, limit=limit):
        recipes.append(recipe)
    if jq_query is not None:
        recipes = compiled_query.all(recipes)

    if not recipes:
        log.info("No results")

    for r in recipes:
        if isinstance(r, str):
            log.info(r)
        else:
            log.info(json.dumps(r, indent=True))


@recipes.command("get")
@filter_options
@logging_options
@server_options
@click.option("--jq-query", "-j")
@click.argument("recipe_id", type=int)
@async_trampoline
@with_session
async def get_recipe(session, filters, jq_query, recipe_id):
    if jq_query:
        compiled_query = pyjq.compile(jq_query)

    recipe = await api_fetch(session, f"recipe/{recipe_id}/")
    if jq_query is not None:
        rv = compiled_query.all(recipe)
    else:
        rv = recipe

    if not rv:
        log.info("No results")

    if isinstance(rv, str):
        log.info(rv)
    else:
        log.info(json.dumps(rv, indent=True))


if __name__ == "__main__":
    cli(auto_envvar_prefix="EDI_")
