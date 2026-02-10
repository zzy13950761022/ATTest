"""
Color theme and style configurations.
"""
from rich.theme import Theme

# TestAgent color theme
TESTAGENT_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "tool.readonly": "blue",
    "tool.write": "magenta bold",
    "prompt": "bold cyan",
    "model": "green italic",
    "stage.current": "bold yellow",
    "stage.completed": "green",
    "stage.pending": "dim white",
})

# Style constants
STYLE_HEADER = "bold cyan"
STYLE_SUBHEADER = "cyan"
STYLE_KEYWORD = "bold"
STYLE_VALUE = "green"
STYLE_ERROR = "bold red"
STYLE_SUCCESS = "bold green"
STYLE_DIM = "dim"
