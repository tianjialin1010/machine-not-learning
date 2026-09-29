from __future__ import annotations

from .models import ActionRecommendation, DecisionRequest, Evidence


class TemplateComposer:
    """Safe offline composer; replace with a generative adapter in production."""

    def compose(self, request: DecisionRequest, action: ActionRecommendation, evidence: list[Evidence]) -> str:
        if "prediction_only_no_action_policy" in action.reason_codes:
            return "本次仅返回结构化判断结果，尚未配置业务动作策略。"
        if action.kind in {"retrieve_evidence", "retrieve_history"}:
            return "我先检索相关证据，再给你一个可核对的回答。"
        if action.kind == "ask_clarification":
            return "我还不确定你希望我先回答哪一部分，可以再具体一点吗？"
        if action.kind == "human_review":
            return "这条消息需要更谨慎地处理，我先暂停自动判断。"
        if action.kind == "stop":
            return "我先暂停处理，避免在信息不足时继续猜测。"
        if action.kind == "abstain":
            return "当前信息不足以可靠判断，我先不强行给出结论。"
        if action.kind == "draft_suggestion":
            return "我可以先整理一个行动建议，确认后再继续。"
        if evidence:
            return "我根据当前对话和已检索到的内容来回答。"
        return "我先根据当前这句话回答，如果理解不对你可以纠正我。"
