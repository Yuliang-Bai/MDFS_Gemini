from ...base import BaseMethod, FitResult
class MCFL(BaseMethod):
    def fit(self, X, y=None):
        return FitResult(selected_features={k: list(range(5)) for k in X})
class MRAG(BaseMethod):
    def fit(self, X, y=None):
        return FitResult(selected_features={k: list(range(5)) for k in X})
