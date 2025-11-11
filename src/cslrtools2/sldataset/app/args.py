from clipar import namespace, mixin
from .plugins import loader

plugins = loader()

@namespace
class CliArgs(mixin.ReprMixin, mixin.CommandMixin): pass
for pl_info in plugins.values():
    CliArgs.add_wrapper(pl_info["name"], pl_info["nswrapper"])

