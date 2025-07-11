import logging

logger = logging.getLogger(__name__)

def mixture_of_agents(system_prompt: str, initial_query: str, client, model: str) -> str:
    logger.info(f"Starting mixture_of_agents function with model: {model}")
    moa_completion_tokens = 0
    completions = []

    logger.debug(f"Generating initial completions for query: {initial_query}")
    
    try:
        # Try to generate 3 completions in a single API call using n parameter
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_query}
            ],
            max_tokens=4096,
            n=3,
            temperature=1
        )
        completions = [choice.message.content for choice in response.choices]
        moa_completion_tokens += response.usage.completion_tokens
        logger.info(f"Generated {len(completions)} initial completions using n parameter. Tokens used: {response.usage.completion_tokens}")
        
    except Exception as e:
        logger.warning(f"n parameter not supported by provider: {str(e)}")
        logger.info("Falling back to generating 3 completions one by one")
        
        # Fallback: Generate 3 completions one by one in a loop
        completions = []
        for i in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": initial_query}
                    ],
                    max_tokens=4096,
                    temperature=1
                )
                completions.append(response.choices[0].message.content)
                moa_completion_tokens += response.usage.completion_tokens
                logger.debug(f"Generated completion {i+1}/3")
                
            except Exception as fallback_error:
                logger.error(f"Error generating completion {i+1}: {str(fallback_error)}")
                continue
        
        if not completions:
            logger.error("Failed to generate any completions")
            return "Error: Could not generate any completions", 0
        
        logger.info(f"Generated {len(completions)} completions using fallback method. Total tokens used: {moa_completion_tokens}")
    
    # Handle case where fewer than 3 completions were generated
    if len(completions) < 3:
        original_count = len(completions)
        # Pad with the first completion to ensure we have 3
        while len(completions) < 3:
            completions.append(completions[0])
        logger.warning(f"Only generated {original_count} unique completions, padded to 3 for critique")
    
    logger.debug("Preparing critique prompt")
    critique_prompt = f"""
    Original query: {initial_query}

    I will present you with three candidate responses to the original query. Please analyze and critique each response, discussing their strengths and weaknesses. Provide your analysis for each candidate separately.

    Candidate 1:
    {completions[0]}

    Candidate 2:
    {completions[1]}

    Candidate 3:
    {completions[2]}

    Please provide your critique for each candidate:
    """

    logger.debug("Generating critiques")
    critique_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": critique_prompt}
        ],
        max_tokens=512,
        n=1,
        temperature=0.1
    )
    critiques = critique_response.choices[0].message.content
    moa_completion_tokens += critique_response.usage.completion_tokens
    logger.info(f"Generated critiques. Tokens used: {critique_response.usage.completion_tokens}")
    
    logger.debug("Preparing final prompt")
    final_prompt = f"""
    Original query: {initial_query}

    Based on the following candidate responses and their critiques, generate a final response to the original query.

    Candidate 1:
    {completions[0]}

    Candidate 2:
    {completions[1]}

    Candidate 3:
    {completions[2]}

    Critiques of all candidates:
    {critiques}

    Please provide a final, optimized response to the original query:
    """

    logger.debug("Generating final response")
    final_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}
        ],
        max_tokens=8192,
        n=1,
        temperature=0.1
    )
    moa_completion_tokens += final_response.usage.completion_tokens
    logger.info(f"Generated final response. Tokens used: {final_response.usage.completion_tokens}")
    
    logger.info(f"Total completion tokens used: {moa_completion_tokens}")
    return final_response.choices[0].message.content, moa_completion_tokens