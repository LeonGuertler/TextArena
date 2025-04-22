# from textarena.wrappers.RenderWrappers.terminal_render_wrapper import TerminalRenderWrapper
from textarena.wrappers.RenderWrappers.CursesRenderWrapper.render import CursesRenderWrapper
from textarena.wrappers.RenderWrappers.SimpleRenderWrapper.render import SimpleRenderWrapper

__all__ = ["SimpleRenderWrapper", "CursesRenderWrapper"]