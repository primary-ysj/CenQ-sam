class _DebugLogger:
    """Small compatibility logger for legacy debug hooks."""

    def __init__(self):
        self._max_values = {}

    @staticmethod
    def _to_scalar(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu()
        if hasattr(value, 'item'):
            try:
                return value.item()
            except ValueError:
                pass
        return value

    def max_log(self, name, value, log_func=print):
        scalar = self._to_scalar(value)
        previous = self._max_values.get(name)
        if previous is None or scalar > previous:
            self._max_values[name] = scalar
            log_func(f'{name}: {scalar}')


logger = _DebugLogger()
