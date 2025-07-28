from pydantic import BaseModel, Field


# Treated as a tool, to be invoked by the primary assistant
# which will become part of the AIMessage=>tool_name, args when invoked by the primary assistant
class ToFlightAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""

    request: str = Field(
        description="Any necessary followup questions the update flight assistant should clarify before proceeding."
    )

class ToTaxiAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle taxi related information."""

    request: str = Field(
        description="Any necessary followup questions the taxi assistant should clarify before proceeding."
    )


class WorkerCompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the primary assistant,
    who can re-route the dialog based on the user's needs.
    example 1: "cancel": True, "reason": "User changed their mind about the current task.",
    example 2: "cancel": True, "reason": "I have fully completed the task.",
    example 3: "cancel": False, "reason": "I need to search the user's emails or calendar for more information.",
    """

    cancel: bool = True
    reason: str

    # class Config:
    #     json_schema_extra = {
    #         "examples": [
    #             {
    #                 "cancel": True,
    #                 "reason": "User changed their mind about the current task.",
    #             },
    #             {
    #                 "cancel": True,
    #                 "reason": "I have fully completed the task.",
    #             },
    #             {
    #                 "cancel": False,
    #                 "reason": "I need to search the user's emails or calendar for more information.",
    #             },
    #         ]
    #     }
