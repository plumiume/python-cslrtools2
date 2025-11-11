from typing import Literal

type ExecutorMode = Literal["batch", "frames"] | None
"Type alias for executor modes."
type ExecutorType = Literal["thread", "process"] | None
"Type alias for executor types."

type ExistRule = Literal["skip", "overwrite", "suffix", "error"]
"Type alias for file existence handling rules."
