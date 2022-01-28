import asyncio
import functools
import json
import logging
import math
import re
import ssl
import sys
from collections import Counter, defaultdict

import aiohttp
import click
import iso8601
import jsonschema
from pypeln import asyncio_task as pipeline
from tqdm import tqdm

from . import jexlutils
from .api import api_fetch, api_request, META_SERVER_KEY, SERVERS
from .log import log
from .cache import cache


def compose(*functions):
    def compose2(f, g):
        return lambda x: f(g(x))

    return functools.reduce(compose2, functions, lambda x: x)


class Iso8601Param(click.ParamType):
    def convert(self, value, param, ctx):
        try:
            return iso8601.parse_date(value)
        except iso8601.ParseError as e:
            self.fail(str(e), param, ctx)


def _make_httpio_session(*, cert=None, **kwargs):
    ssl_context = ssl.create_default_context()
    if cert:
        ssl_context.load_verify_locations(cert)
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    return aiohttp.ClientSession(connector=connector, **kwargs)


def with_session(async_func):
    @click.option("--cert")
    @click_compatible_wraps(async_func)
    async def inner(cert, *args, **kwargs):
        async with _make_httpio_session(cert=cert) as session:
            return await async_func(*args, session=session, **kwargs)

    return inner


def with_authed_session(async_func):
    @click.option("--cert")
    @click.option("-A", "--auth", required=True, prompt=True, envvar="EDI_AUTH")
    @click_compatible_wraps(async_func)
    async def inner(cert, auth, *args, **kwargs):
        if " " not in auth:
            auth = "Bearer " + auth
        headers = {"Authorization": auth}
        async with _make_httpio_session(cert=cert, headers=headers) as authed_session:
            return await async_func(*args, authed_session=authed_session, **kwargs)

    return inner


def click_compatible_wraps(wrapped_func):
    assignments = [*functools.WRAPPER_ASSIGNMENTS, "__click_params__"]
    return functools.wraps(wrapped_func, assignments)


def filter_options(func):
    """
    Common options for filter recipes.
    """

    @click.option("--text", "-t")
    @click.option("--enabled", "-e", is_flag=True, default=None)
    @click.option("--action", "-a")
    @click.option("--creator", "-c")
    @click.option("--enabled-begin", "-b")
    @click.option("--enabled-end", "-e")
    @click_compatible_wraps(func)
    def filter_parser(text, enabled, action, creator, enabled_begin, enabled_end, *args, **kwargs):
        filters = {
            "text": text,
            "enabled": enabled,
            "action": action,
            "creator": creator,
            "enabled_begin": iso8601.parse_date(enabled_begin) if enabled_begin else None,
            "enabled_end": iso8601.parse_date(enabled_end) if enabled_end else None,
        }
        return func(*args, filters=filters, **kwargs)

    return filter_parser


def logging_options(func):
    @click.option("--verbose", "-v", is_flag=True, multiple=True)
    @click_compatible_wraps(func)
    def logging_wrapper(*args, verbose=False, **kwargs):
        if len(verbose) == 1:
            print("log level DEBUG")
            log.setLevel(logging.DEBUG)
        elif len(verbose) > 1:
            print("log level TRACE")
            log.setLevel(logging.DEBUG - 5)
        return func(*args, **kwargs)

    return logging_wrapper


def server_options(func):
    @click.option("--server", "-s", type=click.Choice(SERVERS.keys()), default="prod")
    @click.pass_context
    @click_compatible_wraps(func)
    def server_wrapper(context, *args, server, verbose=False, **kwargs):
        server_url = SERVERS[server]
        context.meta[META_SERVER_KEY] = server_url
        return func(*args, **kwargs)

    return server_wrapper


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


async def paginated_fetch(session, endpoint, *, query=None, admin=False, desc="fetch", limit=None):
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
    session,
    *,
    text=None,
    enabled=None,
    action=None,
    creator=None,
    limit=None,
    enabled_begin=None,
    enabled_end=None,
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
            approved_creator = (recipe.get("latest_revision", {}).get("creator", {}) or {}).get(
                "email", ""
            )
            latest_creator = (recipe.get("latest_revision", {}).get("creator", {}) or {}).get(
                "email", ""
            )
            return creator not in approved_creator and creator not in latest_creator

    history_bar = None

    async def match_enabled_range(recipe):
        if enabled_begin is None and enabled_end is None:
            return True

        nonlocal history_bar
        if history_bar is None:
            history_bar = tqdm(desc="fetching histories", total=0)
        history_bar.total += 1

        rev = recipe["latest_revision"]

        if enabled_begin:
            # If the latest revision is not enabled and was created before the
            # start, then it can't be enabled in the range.
            if iso8601.parse_date(rev["updated"]) < enabled_begin and not rev["enabled"]:
                history_bar.update(1)
                return False

        # Otherwise fetch the history and check if it was enabled
        history = await api_fetch(session, f"/recipe/{recipe['id']}/history/")
        history_bar.update(1)

        for rev in history:
            for enabled_state in rev["enabled_states"]:
                if enabled_state["enabled"]:
                    created = iso8601.parse_date(enabled_state["created"])
                    if (not enabled_begin or created >= enabled_begin) and (
                        not enabled_end or created < enabled_end
                    ):
                        return True

        return False

    # async functions don't support yield from. This is about the same.
    async for recipe in (
        paginated_fetch(session, "recipe", query=query, desc="fetch recipes", limit=limit)
        | pipeline.filter(match_creator, workers=16, maxsize=4)
        | pipeline.filter(match_enabled_range, workers=16, maxsize=4)
    ):
        yield recipe

    if history_bar:
        history_bar.close()


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
    async for group in paginated_fetch(authed_session, "group", admin=True, desc="fetch groups"):
        log.info(group)


@cli.group()
@logging_options
@server_options
def users():
    pass


@users.command("list")
@logging_options
@server_options
@async_trampoline
@with_authed_session
async def list_users(authed_session):
    async for user in paginated_fetch(authed_session, "user", admin=True, desc="fetch users"):
        groups = [g["name"] for g in user["groups"]]
        log.info(f"{user['id']} - {user['email']} - [{', '.join(groups)}]")


@users.command("whoami")
@logging_options
@server_options
@async_trampoline
@with_authed_session
async def whoami(authed_session):
    log.info(await api_fetch(authed_session, "user/me", admin=True, version=1))


@users.command("add-to-group")
@logging_options
@server_options
@click.option("--user-id", "-u", required=True, prompt=True, type=click.INT)
@click.option("--group-id", "-g", required=True, prompt=True)
@async_trampoline
@with_authed_session
async def add_user_to_group(authed_session, user_id, group_id):
    log.debug(f"user_id: {user_id!r}")
    await api_request(
        authed_session, "POST", f"group/{group_id}/add_user", data={"user_id": user_id}, admin=True
    )


@users.command("add")
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


@users.command("delete")
@logging_options
@server_options
@click.option("--user-id", "-u", required=True, prompt=True)
@async_trampoline
@with_authed_session
async def delete_user(authed_session, user_id):
    log.info(await api_request(authed_session, "DELETE", f"user/{user_id}/", admin=True))


@cli.group()
@logging_options
@server_options
def recipes():
    pass


@recipes.command("all")
@filter_options
@logging_options
@server_options
@click.option("--limit", "-l", type=int)
@async_trampoline
@with_session
async def all_recipes(session, filters, limit):
    recipes = []
    async for recipe in fetch_recipes(session, **filters, limit=limit):
        recipes.append(recipe)

    if not recipes:
        log.info("No results")

    for r in recipes:
        if isinstance(r, str):
            log.info(r)
        else:
            log.info(json.dumps(r, indent=True))


@recipes.command("summarize")
@filter_options
@logging_options
@server_options
@async_trampoline
@with_session
async def summarize(session, filters):
    """Show recipes enabled during a time range"""

    recipes = []
    async for recipe in fetch_recipes(session, **filters):
        recipes.append(recipe)

    if not recipes:
        log.info("No results")

    by_action = defaultdict(lambda: [])
    for r in recipes:
        rev = r["approved_revision"] or r["latest_revision"]
        by_action[rev["action"]["name"]].append(r)

    log.info(f"{len(recipes)} recipes active in given range")
    for action_name, recipes in by_action.items():
        log.info(f"  * {action_name} - {len(recipes)}")


@recipes.command("get")
@filter_options
@logging_options
@server_options
@click.argument("recipe_id", type=int)
@async_trampoline
@with_session
async def get_recipe(session, filters, recipe_id):
    recipe = await api_fetch(session, f"recipe/{recipe_id}/")

    if not recipe:
        log.info("No results")

    if isinstance(recipe, str):
        log.info(recipe)
    else:
        log.info(json.dumps(recipe, indent=True))


@recipes.command("empty")
@filter_options
@logging_options
@server_options
@click.option("--limit", "-l", type=int)
@async_trampoline
@with_session
async def empty_recipes(session, filters, limit):
    recipes = []
    async for recipe in fetch_recipes(session, **filters, limit=limit):
        if recipe["latest_revision"] is None:
            recipes.append(recipe)

    if not recipes:
        log.info("No results")

    for recipe in recipes:
        if isinstance(recipe, str):
            log.info(recipe)
        else:
            log.info(json.dumps(recipe, indent=True))


@recipes.command("delete")
@logging_options
@server_options
@click.argument("recipe_ids", type=int, nargs=-1, required=True)  # any number of recipe ids
@async_trampoline
@with_authed_session
async def delete_recipes(authed_session, recipe_ids):
    for recipe_id in recipe_ids:
        await api_request(authed_session, "DELETE", f"recipe/{recipe_id}/", admin=True)


@recipes.command("create")
@logging_options
@server_options
@click.argument("data", type=str, required=True)
@async_trampoline
@with_authed_session
async def create_recipe(authed_session, data):
    data = json.loads(data)
    response = await api_request(
        authed_session,
        "POST",
        "recipe/",
        data=data,
        admin=True,
        headers={"Content-Type": "application/json"},
    )
    log.info(response)


@recipes.command("revise")
@logging_options
@server_options
@click.option("-r", "--recipe", type=int, required=True)
@click.argument("data", type=str, required=True)
@async_trampoline
@with_authed_session
async def revise_recipe(authed_session, recipe, data):
    data = json.loads(data)
    response = await api_request(
        authed_session,
        "PATCH",
        f"recipe/{recipe}/",
        data=data,
        admin=True,
        headers={"Content-Type": "application/json"},
    )
    log.info(response)


@recipes.command("count-filters")
@filter_options
@logging_options
@server_options
@click.option("--limit", "-l", type=int)
@async_trampoline
@with_session
async def count_filter(session, filters, limit):
    """Count the usage of filter object types"""
    recipes = []
    async for recipe in fetch_recipes(session, **filters, limit=limit):
        recipes.append(recipe)

    if not recipes:
        log.info("No results")

    filter_types = Counter()

    for r in recipes:
        if r.get("approved_revision"):
            filter_objects = r["approved_revision"]["filter_object"] or []
            has_extra_filter_expression = bool(r["approved_revision"]["extra_filter_expression"])
        elif r.get("latest_revision"):
            filter_objects = r["latest_revision"]["filter_object"] or []
            has_extra_filter_expression = bool(r["latest_revision"]["extra_filter_expression"])
        else:
            continue

        for filter_object in filter_objects:
            filter_types[filter_object.get("type")] += 1
        if has_extra_filter_expression:
            filter_types["extra"] += 1

        if not (
            has_extra_filter_expression
            or any(fo.get("type") == "bucketSample" for fo in filter_objects)
            or any(fo.get("type") == "stableSample" for fo in filter_objects)
        ):
            filter_types["<no sampling>"] += 1

        if not filter_objects and has_extra_filter_expression:
            filter_types["<only extra>"] += 1

    log.info(f"{len(recipes)} total recipes")
    for filter_type, count in filter_types.most_common():
        percent = round(count / len(recipes) * 1000) / 10
        log.info(f"  - {filter_type:<12} {count:>3} ({percent}%)")


@recipes.command("classify-filters")
@filter_options
@logging_options
@server_options
@click.option("--limit", "-l", type=int)
@async_trampoline
@with_session
async def classify_recipe_filters(session, limit, filters):
    """
    Analyze filter expressions to look for common patterns
    """
    recipes = []
    async for recipe in fetch_recipes(session, **filters, limit=limit):
        recipes.append(recipe)

    filter_classifications = Counter()

    count = 0
    for recipe in recipes:
        rev = recipe["approved_revision"] or recipe["latest_revision"]
        if not rev.get("extra_filter_expression"):
            continue
        count += 1
        for classification in jexlutils.classify_filter(rev["extra_filter_expression"]):
            filter_classifications[classification] += 1

    if not recipes:
        log.info("No recipes to analyze")
        return

    percent = round(count / len(recipes) * 1000) / 10
    log.info(f"Of {len(recipes)} recipes, {count} ({percent}%)have extra filter expression")

    max_classification_width = max(len(c) for c in filter_classifications.keys())
    max_digit_width = math.ceil(
        max(math.log(c) for c in filter_classifications.values()) / math.log(10)
    )
    format_string = " - {:<%s}  {:>%s} ({:>4}%%)" % (max_classification_width, max_digit_width)

    for classification, count in sorted(
        filter_classifications.most_common(), key=lambda v: (-v[1], v[0])
    ):
        percent = round(count / len(recipes) * 1000) / 10
        log.info(format_string.format(classification, count, percent))


@recipes.command("check-argument-schemas")
@filter_options
@logging_options
@server_options
@click.option("--limit", "-l", type=int)
@async_trampoline
@with_session
async def recipes_check_argument_schemas(session, limit, filters):
    """
    Check if recipe arguments match their schemas
    """
    stats = Counter()

    async for recipe in fetch_recipes(session, **filters, limit=limit):
        stats["total"] += 1
        revision = recipe["approved_revision"] or recipe["latest_revision"]
        arguments = revision["arguments"]
        schema = revision["action"]["arguments_schema"]
        action_name = revision["action"]["name"]

        try:
            jsonschema.validate(arguments, schema)
            stats["ok"] += 1
            stats[f"ok::{action_name}"] += 1
        except jsonschema.exceptions.ValidationError as err:
            stats["error"] += 1
            stats[f"error::{action_name}"] += 1
            log.warn(f"{action_name} recipe {recipe['id']} doesn't match schema")
            log.debug(f"\t{err.message}")

    keys = sorted([k for k in stats.keys() if k != "total"])
    keys.insert(0, "total")
    max_key_length = max(len(k) for k in keys)
    max_val_length = max(len(str(v)) for v in stats.values())
    format_str = "{key:<%i}  {val:>%i}" % (max_key_length, max_val_length)

    for key in keys:
        log.info(format_str.format(key=key, val=stats[key]))


@cli.group()
@logging_options
@server_options
def extensions():
    pass


@extensions.command("all")
@logging_options
@server_options
@click.option("--limit", "-l", type=int)
@async_trampoline
@with_session
async def all_extensions(session, limit):
    extensions = []

    async for extension in paginated_fetch(
        session, "extension", desc="fetch extensions", limit=limit
    ):
        extensions.append(extension)

    if not extensions:
        log.info("No results")

    for extension in extensions:
        if isinstance(extension, str):
            log.info(extension)
        else:
            log.info(json.dumps(extension, indent=True))


@extensions.command("delete")
@logging_options
@server_options
@click.argument("extension_id", type=int)
@async_trampoline
@with_authed_session
async def delete_extension(authed_session, extension_id):
    await api_request(authed_session, "DELETE", f"extension/{extension_id}/", admin=True)


@cli.command("check-extensions")
@server_options
@logging_options
@async_trampoline
@with_session
async def check_extensions(session):
    """
    Check for revisions that reference extension XPIs that don't correspond
    to any uploaded extension.
    """
    extension_filenames = set()
    async for extension in paginated_fetch(session, "extension", desc="fetch extensions"):
        extension_filenames.add(extension["xpi"].split("/")[-1])

    bad_revisions = []
    async for revision in paginated_fetch(session, "recipe_revision", desc="fetch revisions"):
        try:
            if revision["action"]["name"] != "opt-out-study":
                continue
        except Exception:
            log.info("revision=", revision)

        addon_filename = revision["arguments"]["addonUrl"].split("/")[-1]
        if addon_filename not in extension_filenames:
            bad_revisions.append(revision)

    if not bad_revisions:
        log.info("No bad revisions")

    for r in bad_revisions:
        if isinstance(r, str):
            log.info(r)
        else:
            log.info(json.dumps(r, indent=True))


@cli.command("convert-pref-exp")
@click.argument("old_recipe_json", type=click.File("r"))
@click.option("--public-name", required=True, prompt=True)
@click.option("--public-description", required=True, prompt=True)
def convert_pref_exp(old_recipe_json, public_name, public_description):
    old_recipe = json.load(old_recipe_json)

    new_recipe = {
        "branches": [],
        "experimentDocumentUrl": old_recipe["experimentDocumentUrl"],
        "slug": old_recipe["slug"],
        "userFacingDescription": public_description,
        "userFacingName": public_name,
    }

    for branch in old_recipe["branches"]:
        new_recipe["branches"].append(
            {
                "ratio": branch["ratio"],
                "slug": branch["slug"],
                "preferences": {
                    old_recipe["preferenceName"]: {
                        "preferenceBranchType": old_recipe["preferenceBranchType"],
                        "preferenceType": old_recipe["preferenceType"],
                        "preferenceValue": branch["value"],
                    }
                },
            }
        )

    log.info(("=" * 80) + "\n" + json.dumps(new_recipe, indent=2))


@cli.command("fix-is-enrollment-paused")
@logging_options
@server_options
@async_trampoline
@with_authed_session
async def revise_recipe_arguments(authed_session):
    async def patch(recipe, base_revision):
        new_arguments = {**base_revision["arguments"], "isEnrollmentPaused": False}
        log.info(f"Recipe {recipe['id']} will be patched")
        await api_request(
            authed_session,
            "PATCH",
            f"recipe/{recipe['id']}/",
            data={
                "arguments": new_arguments,
                "comment": "mass edit to fix isEnrollmentPaused missing",
            },
            admin=True,
            headers={"Content-Type": "application/json"},
        )

    service_info = await api_request(authed_session, "GET", "service_info/", admin=True)
    peer_approval_enforced = service_info["peer_approval_enforced"]

    async for recipe in fetch_recipes(authed_session, action="multi-preference-experiment"):
        approved = recipe["approved_revision"]
        latest = recipe["latest_revision"]

        if approved and approved["id"] == latest["id"]:
            # Approved is latest, this is easy
            if "isEnrollmentPaused" in approved["arguments"]:
                # Nothing to do here
                log.debug(f"Recipe {recipe['id']} has isEnrollmentPaused")
                continue
            else:
                await patch(recipe, approved)

        elif not approved:
            # Not approved, easy
            patch(recipe, latest)
        else:
            # This will be harder
            if "isEnrollmentPaused" in latest["arguments"]:
                if "isEnrollmentPaused" in approved["arguments"]:
                    log.debug(f"Recipe {recipe['id']} has isEnrollmentPaused")
                else:
                    approval_id = None
                    if latest["approval_request"]:
                        approval_id = latest["approval_request"]["id"]
                        if peer_approval_enforced:
                            log.info(
                                f"Recipe {recipe['id']} needs review: ext+normandy://prod/recipes/{recipe['id']}"
                            )
                    else:
                        approval_request = await api_request(
                            authed_session,
                            "POST",
                            f"recipe_revision/{latest['id']}/request_approval/",
                            admin=True,
                        )
                        approval_id = approval_request["id"]
                        if peer_approval_enforced:
                            log.info(
                                f"Recipe {recipe['id']} requested review: ext+normandy://prod/recipes/{recipe['id']}"
                            )

                    if not peer_approval_enforced:
                        await api_request(
                            authed_session,
                            "POST",
                            f"approval_request/{approval_id}/approve/",
                            admin=True,
                            data={"comment": "mass approved for isEnrollmentPaused fix"},
                            headers={"Content-Type": "application/json"},
                        )
                        log.info(f"Recipe {recipe['id']} self-approved")

            else:
                if "isEnrollmentPaused" in approved["arguments"]:
                    log.warn(
                        f"Recipe {recipe['id']} has draft that regress isEnrollmentPaused and needs manual work ext+normandy://prod/recipes/{recipe['id']}"
                    )
                else:
                    log.info(
                        f"Recipe {recipe['id']} has a draft and needs manual work ext+normandy://prod/recipes/{recipe['id']}"
                    )


def main():
    try:
        cli()
    finally:
        log.debug(f"Cache info: {cache.stats.most_common()}")
        cache.write()


if __name__ == "__main__":
    main()
