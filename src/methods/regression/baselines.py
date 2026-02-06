from ...base import BaseMethod, FitResult
class AdaCoop(BaseMethod):
    def fit(self, X, y):
        # Placeholder
        return FitResult(selected_features={k: list(range(5)) for k in X})
class MSGLasso(BaseMethod):
    def fit(self, X, y):
        # Placeholder
        return FitResult(selected_features={k: list(range(5)) for k in X})
