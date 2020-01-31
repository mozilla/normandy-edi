import logging

from tqdm import tqdm


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


log = logging.getLogger("edi")
log.addHandler(TqdmLoggingHandler())
log.setLevel(logging.INFO)
