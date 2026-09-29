class ProviderError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False, code: str = "provider_error"):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.code = code
