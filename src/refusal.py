def check_refusal(question) -> str | None:
    """Returns refusal text if the question contains strings like 'should i invest'."""
    q_lower = question.lower().strip()
    REFUSAL_TRIGGERS = [
        "should i invest", "buy or sell", "which stock should",
        "recommend a stock", "best investment", "should i buy",
        "will the stock go up", "price prediction", "stock tips",
        "should i sell", "is it a good time to buy"
    ]
    
    for trigger in REFUSAL_TRIGGERS:
        if trigger in q_lower:
            return (
                "I am a financial research agent and cannot provide investment "
                "advice. I can share financial data and management commentary from "
                "the FY24 annual reports, but buy or sell decisions require a "
                "qualified financial advisor. No tools were called."
            )
            
    return None
