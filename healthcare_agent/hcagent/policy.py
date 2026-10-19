"""Deterministic access control. The LLM never decides whether access is granted.

Rules
  R0 role_tool        the user's role must be listed in the tool's allowed_roles
  R1 ops_no_phi       operations users never reach patient-identifiable tools/intents
  R2 patient_self     a patient may only access their own record
  R3 clinician_scope  a clinician with a care-team list may only access patients on it
  R4 target_required  patient-scoped access needs a resolved patient ID
"""

from __future__ import annotations

from .schemas import PATIENT_SCOPED_INTENTS, Intent, PermissionDecision, Role, UserContext
from .tools import TOOLS, ToolSpec


class PolicyEngine:
    def allowed_tools(self, user: UserContext) -> list[str]:
        return [name for name, spec in TOOLS.items() if user.role in spec.allowed_roles]

    def check_patient_access(self, user: UserContext, patient_id: str | None) -> PermissionDecision:
        if user.role == Role.OPERATIONS:
            return PermissionDecision(
                allowed=False, rule="R1 ops_no_phi", patient_id=patient_id,
                reason="Operations users cannot access patient-identifiable clinical records.",
            )
        if patient_id is None:
            return PermissionDecision(
                allowed=False, rule="R4 target_required",
                reason="No patient was identified for this request.",
            )
        if user.role == Role.PATIENT:
            if user.patient_id is None or patient_id != user.patient_id:
                return PermissionDecision(
                    allowed=False, rule="R2 patient_self", patient_id=patient_id,
                    reason="Patients can only access their own records.",
                )
            return PermissionDecision(
                allowed=True, rule="R2 patient_self", patient_id=patient_id,
                reason="Authenticated patient accessing own record.",
            )
        if user.authorized_patients is not None and patient_id not in user.authorized_patients:
            return PermissionDecision(
                allowed=False, rule="R3 clinician_scope", patient_id=patient_id,
                reason="Patient is not in this clinician's care-team scope.",
            )
        return PermissionDecision(
            allowed=True, rule="R3 clinician_scope", patient_id=patient_id,
            reason="Clinician authorised for this patient.",
        )

    def check_tool(self, user: UserContext, spec: ToolSpec, args: dict) -> PermissionDecision:
        if user.role not in spec.allowed_roles:
            rule = "R1 ops_no_phi" if spec.scope == "patient" else "R0 role_tool"
            return PermissionDecision(
                allowed=False, rule=rule, tool=spec.name, patient_id=args.get("patient_id"),
                reason=f"Role '{user.role.value}' may not use {spec.name}.",
            )
        if spec.scope == "patient":
            decision = self.check_patient_access(user, args.get("patient_id"))
            return decision.model_copy(update={"tool": spec.name})
        return PermissionDecision(
            allowed=True, rule="R0 role_tool", tool=spec.name,
            reason="Non-identifiable operational data.",
        )

    def check_request(
        self, user: UserContext, intents: list[Intent], patient_id: str | None
    ) -> PermissionDecision:
        """Request-level gate, evaluated before routing or any tool execution."""
        if any(i in PATIENT_SCOPED_INTENTS for i in intents):
            return self.check_patient_access(user, patient_id)
        return PermissionDecision(allowed=True, rule="R0 role_tool", reason="No patient-identifiable data requested.")
