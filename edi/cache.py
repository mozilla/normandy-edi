import json
import logging
import os
from collections import Counter
from datetime import datetime, timedelta, timezone

import iso8601

from .log import TqdmLoggingHandler


CACHE_PATH = "./cache.json"

log = logging.getLogger(__name__)
log.addHandler(TqdmLoggingHandler())
log.setLevel(logging.INFO)


class Cache(object):
    def __init__(self):
        self._cache_data = None
        self.stats = Counter()

    @property
    def cache_data(self):
        if self._cache_data is None:
            try:
                with open(CACHE_PATH) as f:
                    self._cache_data = json.load(f)
            except json.JSONDecodeError as e:
                log.warn(f"Cache file not valid JSON: {e}")
                try:
                    os.unlink(CACHE_PATH)
                except IOError:
                    log.error(f"Could not delete invalid cache")
                self._cache_data = {}
            except IOError as e:
                log.debug(f"cache file could not be opened: {e}")
                self._cache_data = {}

        return self._cache_data

    def write(self):
        if self._cache_data is None:
            return

        for key, entry in self._cache_data.items():
            if is_timed_out(entry):
                del self._cache_data[key]

        with open(CACHE_PATH, "w") as f:
            json.dump(self._cache_data, f)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.write()

    def set(self, key, value, timeout):
        deadline = datetime.now() + timedelta(seconds=timeout)
        deadline = deadline.astimezone(timezone.utc)
        json.dumps(value)  # to check if the value is serializable
        self.cache_data[key] = {"value": value, "deadline": deadline.isoformat()}

    def get(self, key):
        self.stats["query"] += 1
        entry = self.cache_data.get(key)
        if entry:
            if is_timed_out(entry):
                del self._cache_data[key]
                self.stats["miss"] += 1
                return None
            else:
                self.stats["hit"] += 1
                return entry["value"]
        else:
            self.stats["miss"] += 1


def is_timed_out(entry):
    deadline = iso8601.parse_date(entry["deadline"])
    now = datetime.utcnow()
    now = now.astimezone(timezone.utc)
    return now > deadline


cache = Cache()
