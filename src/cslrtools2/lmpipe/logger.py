import logging

lmpipe_logger = logging.getLogger("lmpipe")

lmpipe_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s (%(pathname)s:%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
