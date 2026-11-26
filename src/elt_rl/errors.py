"""Failure categories shared by environment, tools and the trainer."""


class ActionError(ValueError):
    """A model-controlled configuration/action can be corrected in this episode."""


class EpisodeLimit(ActionError):
    """The declared episode execution budget expired."""


class InfrastructureError(RuntimeError):
    """A host/service failure invalidates collection, rather than earning policy zero."""


class IntegrityError(RuntimeError):
    """Trusted inputs, frozen artifacts or evidence have changed."""


class OutcomeUnknown(InfrastructureError):
    """A dispatched mutation may have completed; reconcile before attempting another."""


class TokenLimitExceeded(InfrastructureError):
    """A predeclared sampling or training token limit prevents another dispatch."""
