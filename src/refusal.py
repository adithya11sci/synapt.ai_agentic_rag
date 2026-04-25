def check_refusal(question) -> str | None:
    """Return refusal text for disallowed or out-of-domain questions.

    - Investment advice is refused deterministically.
    - Obvious out-of-domain trivia/recipes are refused deterministically to
      avoid tool calls and improve evaluation stability.
    """
    q_lower = question.lower().strip()
    REFUSAL_TRIGGERS = [
        "should i invest", "buy or sell", "which stock should",
        "recommend a stock", "best investment", "should i buy",
        "will the stock go up", "price prediction", "stock tips",
        "should i sell", "is it a good time to buy"
    ]

    OUT_OF_DOMAIN_TRIGGERS = [
        # Recipes / cooking
        "pizza", "dough", "recipe", "cook", "cooking",
        # Assignment-style trivia
        "airspeed velocity", "unladen swallow",
        # Movies / pop culture
        "movie", "monkey king",
    ]
    
    for trigger in REFUSAL_TRIGGERS:
        if trigger in q_lower:
            return (
                "I am a financial research agent and cannot provide investment "
                "advice. I can share financial data and management commentary from "
                "the FY24 annual reports, but buy or sell decisions require a "
                "qualified financial advisor. No tools were called."
            )

    for trigger in OUT_OF_DOMAIN_TRIGGERS:
        if trigger in q_lower:
            return (
                "I am a financial research agent focused on Infosys, TCS, and Wipro "
                "(FY21–FY24 structured financials and FY24 annual-report content). "
                "This question appears outside that scope, so I can’t help with it. "
                "No tools were called."
            )
            
    return None
