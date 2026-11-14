"""One-factor switches for catalog experiments in the local AppWorld environment."""

from clarifysafe.agent.policy import PolicyEngine
from clarifysafe.environment.appworld_tools import AppWorldTools
from clarifysafe.shared.schemas import VerificationResult


class NoRiskFeaturesPolicy(PolicyEngine):
    """Preserve argument, permission and scope checks while removing risk-based confirmation."""

    def assess(self, action, tools):
        assessment = super().assess(action, tools)
        if assessment.decision == "confirm":
            return assessment.model_copy(update={"decision": "act", "reasons": []})
        return assessment


class CatalogAblationTools(AppWorldTools):
    def __init__(self, world, api_docs, controller):
        super().__init__(world, api_docs)
        self.controller = controller

    def policy(self):
        policy = super().policy()
        return NoRiskFeaturesPolicy(policy.config) if self.controller == "no-risk" else policy

    def verify_operation(self, state, result):
        if self.controller == "no-verifier":
            return VerificationResult(passed=True, checks={"verification_disabled": True})
        return super().verify_operation(state, result)

    def verify_plan_state(self, actions, expected):
        if self.controller == "no-verifier":
            return True
        return super().verify_plan_state(actions, expected)
