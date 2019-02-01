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
from tqdm import tqdm
from pypeln import asyncio_task as pipeline


SERVER_URL = "https://normandy.cdn.mozilla.net"


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


async def api_fetch(session, endpoint, opts=None, *, version=3):
    if endpoint.startswith("http"):
        if opts is not None:
            raise Exception("Can't pass opts with a full url")
        url = endpoint
    else:
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]
        if endpoint.endswith("/"):
            endpoint = endpoint[:-1]
        if "?" in endpoint:
            raise Exception("Don't include query string in endpoint")
        qs = "?" + "&".join(f"{k}={v}" for k, v in opts.items()) if opts else ""
        url = f"{SERVER_URL}/api/v{version}/{endpoint}/{qs}"

    log.debug(f"GET {url}")
    async with session.get(url) as resp:
        return json.loads(await resp.text())


class Iso8601Param(click.ParamType):
    def convert(self, value, param, ctx):
        try:
            return iso8601.parse_date(value)
        except iso8601.ParseError as e:
            self.fail(str(e), param, ctx)


@click.group()
def cli():
    pass


def filter_options(func):
    @click.option("--text", "-t")
    @click.option("--enabled", "-e", is_flag=True, default=None)
    @click.option("--action", "-a")
    @click_compatible_wraps(func)
    def filter_parser(text, enabled, action, *args, **kwargs):
        filters = {"text": text, "enabled": enabled, "action": action}
        return func(filters, *args, **kwargs)

    return filter_parser


def logging_options(func):
    @click.option("--verbose", "-v", is_flag=True)
    @click_compatible_wraps(func)
    def logging_wrapper(*args, verbose=False, **kwargs):
        if verbose:
            log.setLevel(logging.DEBUG)
        return func(*args, **kwargs)

    return logging_wrapper


def click_compatible_wraps(wrapped_func):
    assignments = [*functools.WRAPPER_ASSIGNMENTS, "__click_params__"]
    return functools.wraps(wrapped_func, assignments)


def async_trampoline(async_func):
    @click_compatible_wraps(async_func)
    def sync_func(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        loop.run_until_complete(async_func(*args, **kwargs))

    return sync_func


async def fetch_recipes(session, *, text=None, enabled=None, action=None):
    query = {}
    if text is not None:
        query["text"] = text
    if enabled is not None:
        query["enabled"] = enabled
    if action is not None:
        query["action"] = action

    with tqdm(desc="fetch recipes") as progress_bar:
        data = await api_fetch(session, "recipe", query)
        progress_bar.total = data["count"]
        next_url = data["next"]
        for recipe in data["results"]:
            yield recipe
        progress_bar.update(len(data["results"]))

        while next_url:
            data = await api_fetch(session, next_url)
            next_url = data["next"]
            for recipe in data["results"]:
                yield recipe
            progress_bar.update(len(data["results"]))


@cli.command("filter-inputs")
@filter_options
@logging_options
@async_trampoline
async def filter_inputs(filters):
    """find all filter inputs"""
    all_inputs = Counter()
    input_regex = re.compile(r"\[([^]]*)\]\|(stable|bucket)Sample", re.MULTILINE)

    async with aiohttp.ClientSession() as session:
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
@click.option("--begin", "-b", required=True, type=Iso8601Param(), prompt=True)
@click.option("--end", "-e", required=True, type=Iso8601Param(), prompt=True)
@async_trampoline
async def enabled_range(filters, begin=None, end=None):
    """Find all recipes enabled during a certain time range"""
    async with aiohttp.ClientSession() as session:

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
@async_trampoline
async def heartbeat_url_scan(filters):
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

    async with aiohttp.ClientSession() as session:
        urls = set()
        async for recipe in fetch_recipes(session, **filters):
            for url in get_urls(recipe):
                if url not in urls:
                    log.info(url)
                    urls.add(url)


if __name__ == "__main__":
    cli()
